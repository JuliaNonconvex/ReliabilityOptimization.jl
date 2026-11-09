using ReliabilityOptimization, Test, FiniteDifferences, Zygote, ChainRulesCore
using Distributions, Random

const GROUP = get(ENV, "GROUP", "All")

if GROUP in ("All", "Tests")
    Random.seed!(1)

    @testset "RIA on a linear limit state" begin
        # Regression test: with the limit-state value c a free variable, the MPP
        # used to land off the limit-state surface (beta = sqrt(2) for g = p - 2
        # with p ~ N(0, 1)). The Hasofer-Lind solution is p = 2, beta = 2.
        f = (x, p) -> p[1] - 2.0
        p = MvNormal(zeros(1), ones(1, 1))
        p0 = ReliabilityOptimization.getp0(f, [0.0], p, FORM(RIA()))
        @test p0 ≈ [2.0] atol = 1e-4
    end

    @testset "_jacobian rrule" begin
        # Unit test for the ChainRules rrule of the finite-difference Jacobian:
        # the pullback must reproduce the second-order terms (d2f/dxdp and
        # d2f/dp2), which reverse-mode AD through the stencil cannot obtain.
        fs = [
            (x, p) -> p[1] * x[1]^2 + p[2] * x[2] + 1.0,        # scalar output
            (x, p) -> [p[1] * x[1], p[2] * x[2] + p[1] * x[1]], # vector output
        ]
        for f in fs
            x = [1.3, -0.7]
            p = [0.4, 1.1]
            J = ReliabilityOptimization._jacobian(f, x, p)
            Δm = rand(Float64, size(J))
            val, pb = ChainRulesCore.rrule(ReliabilityOptimization._jacobian, f, x, p)
            @test val ≈ J
            _, _, x̄, p̄ = pb(Δm)
            loss = (x, p) -> dot(Δm, ReliabilityOptimization._jacobian(f, x, p))
            x̄_fd = FiniteDifferences.grad(central_fdm(5, 1), x -> loss(x, p), x)[1]
            p̄_fd = FiniteDifferences.grad(central_fdm(5, 1), p -> loss(x, p), p)[1]
            @test x̄ ≈ x̄_fd rtol = 1e-6 atol = 1e-8
            @test p̄ ≈ p̄_fd rtol = 1e-6 atol = 1e-8
            # Zygote routes through the rrule and must agree with the FDs too.
            x̄_zy = Zygote.gradient(x -> loss(x, p), x)[1]
            p̄_zy = Zygote.gradient(p -> loss(x, p), p)[1]
            @test x̄_zy ≈ x̄_fd rtol = 1e-6 atol = 1e-8
            @test p̄_zy ≈ p̄_fd rtol = 1e-6 atol = 1e-8
        end
    end

    @testset "RIA with non-standard Gaussian parameters" begin
        # Non-standard parameters are handled in standard normal space:
        # u = (p - μ)/σ, so β = (x1 - μ)/σ and the MPP is p* = x1.
        f = (x, p) -> p[1] - x[1]
        p = MvNormal([1.0], [4.0;;])
        x = [2.0]
        @test getp0(f, x, p) ≈ [2.0] atol = 1e-4
        @test reliability_index(f, x, p) ≈ 0.5 atol = 1e-4
        @test failure_probability(f, x, p) ≈ cdf(Normal(), -0.5) atol = 1e-6
        g_ad = Zygote.gradient(x -> reliability_index(f, x, p), x)[1]
        g_fd = FiniteDifferences.grad(
            central_fdm(5, 1), x -> reliability_index(f, x, p), x
        )[1]
        @test g_ad ≈ g_fd rtol = 1e-5
    end

    @testset "Vector-valued limit states" begin
        f = (x, p) -> [p[1] - 2.0 - x[1], p[2] - 1.0 - x[2]]
        p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        x = [0.5, -0.3]
        @test getp0(f, x, p) ≈ [2.5, 0.7] atol = 1e-4
        β = reliability_index(f, x, p)
        @test β ≈ sqrt(2.5^2 + 0.7^2) atol = 1e-4
        g_ad = Zygote.gradient(x -> reliability_index(f, x, p), x)[1]
        g_fd = FiniteDifferences.grad(
            central_fdm(5, 1), x -> reliability_index(f, x, p), x
        )[1]
        @test g_ad ≈ g_fd rtol = 1e-5 atol = 1e-6
    end

    @testset "Univariate Normal parameters" begin
        f = (x, p) -> p - 2.0 - x[1]
        p = Normal(0.0, 1.0)
        x = [0.0]
        @test getp0(f, x, p) ≈ 2.0 atol = 1e-4
        @test reliability_index(f, x, p) ≈ 2.0 atol = 1e-4
        rf = RandomFunction(f, p)
        d = rf(x)
        @test mean(d)[1] ≈ -2.0 atol = 1e-4
        @test cov(d)[1, 1] ≈ 1.0 atol = 1e-4
        g_ad = Zygote.gradient(x -> mean(rf(x))[1] + 2 * sqrt(cov(rf(x))[1, 1]), x)[1]
        g_fd = FiniteDifferences.grad(
            central_fdm(5, 1), x -> mean(rf(x))[1] + 2 * sqrt(cov(rf(x))[1, 1]), x
        )[1]
        @test g_ad ≈ g_fd rtol = 1e-5 atol = 1e-6
    end

    @testset "PMA performance measure" begin
        # margin convention: g ≥ 0 is safe; g* = 3 + x1 - β_t√2 on the sphere
        g = (x, p) -> 3.0 + x[1] - p[1] - p[2]
        p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        method = FORM(PMA(β_target=2.0))
        x = [0.0]
        @test performance_measure(g, x, p; method=method) ≈ 3.0 - 2sqrt(2) atol = 1e-4
        g_ad = Zygote.gradient(x -> performance_measure(g, x, p; method=method), x)[1]
        g_fd = FiniteDifferences.grad(
            central_fdm(5, 1), x -> performance_measure(g, x, p; method=method), x
        )[1]
        @test g_ad ≈ g_fd rtol = 1e-5 atol = 1e-6
        @test g_ad ≈ [1.0] rtol = 1e-4
    end

    @testset "reliability_constraint" begin
        g = (x, p) -> 3.0 + x[1] - p[1] - p[2] # margin convention
        p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        c_pma = reliability_constraint(g, p, 2.0) # default: PMA
        c_ria = reliability_constraint(g, p, 2.0; method=FORM(RIA()))
        # the probabilistic_constraint alias
        @test probabilistic_constraint(g, p, 2.0)([0.0]) ≈ c_pma([0.0])
        @test c_pma([-0.3]) > 0 # unreliable side (β = 2.7/√2 < 2)
        @test c_pma([0.3]) < 0  # reliable side
        # both formulations agree on the boundary x1 = 2√2 - 3
        xb = [2sqrt(2) - 3]
        @test c_ria(xb) ≈ 0.0 atol = 1e-3
        @test c_pma(xb) ≈ 0.0 atol = 1e-3
        g_ad = Zygote.gradient(c_ria, [0.0])[1]
        g_fd = FiniteDifferences.grad(central_fdm(5, 1), c_ria, [0.0])[1]
        @test g_ad ≈ g_fd rtol = 1e-4
        g_ad2 = Zygote.gradient(c_pma, [0.0])[1]
        g_fd2 = FiniteDifferences.grad(central_fdm(5, 1), c_pma, [0.0])[1]
        @test g_ad2 ≈ g_fd2 rtol = 1e-4
    end

    @testset "SORM curvature correction" begin
        # linear limit state: zero curvature ⇒ SORM == FORM (also covers n = 1)
        g1 = (x, p) -> p[1] - 2.0
        p1 = MvNormal(zeros(1), ones(1, 1))
        @test reliability_index(g1, [0.0], p1; method=SORM()) ≈ 2.0 atol = 1e-4
        # curved limit state checked against a 1-D quadrature reference
        βt, κc = 3.5, 0.2
        g = (x, p) -> p[2] - (βt + κc / 2 * p[1]^2) # failure: g > 0
        p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        pf_sorm = failure_probability(g, [0.0], p; method=SORM())
        # E[Φ(-βt - κc/2 q²)] by trapezoid quadrature over q ~ N(0, 1)
        qs = range(-8.0, 8.0; length=20001)
        h = step(qs)
        vals = [cdf(Normal(), -βt - κc / 2 * q^2) * pdf(Normal(), q) for q in qs]
        pf_quad = h * (sum(vals) - (first(vals) + last(vals)) / 2)
        @test pf_sorm ≈ pf_quad rtol = 0.15
        # the surface curves away from the origin, so the failure region is
        # smaller than the linearized half-space
        @test pf_sorm < cdf(Normal(), -βt)
        # through a univariate transform: exercises the scalar-parameter FD
        # Hessian and the FD gradient/Hessian branches of the custom-transform
        # path (n = 1 has no tangent space, so SORM == FORM)
        pln = LogNormal(0.0, 0.5)
        gu = (x, p) -> p - 3.0
        @test reliability_index(gu, [0.0], pln; method=SORM()) ≈ log(3) / 0.5 atol = 1e-4
        # undefined Breitung correction (1 + βκ ≤ 0): falls back to FORM (a
        # warning is emitted)
        gb = (x, p) -> p[2] + 0.5 * p[1]^2 - 2.0
        pb = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        @test reliability_index(gb, [0.0], pb; method=SORM()) ≈ 2.0 atol = 1e-4
    end

    @testset "Custom standard-normal transform" begin
        # lognormal-like parameters: p = exp(u) with u standard normal
        tr = StandardNormalTransform(
            u -> exp.(u), p -> log.(p); μ=[exp(0.5)], Σ=[exp(1) * (exp(1) - 1);;]
        )
        f = (x, p) -> p[1] - 3.0
        @test reliability_index(f, [0.0], tr) ≈ log(3) atol = 1e-4
        @test getp0(f, [0.0], tr) ≈ [3.0] atol = 1e-4
        # PMA through a custom transform (finite-difference-gradient conditions):
        # min over |u| = 2 of exp(u) − x1 is attained at u = −2. nstarts = 2
        # because the default start (+β_target) is a KKT point of the sphere
        # problem but the wrong one (the maximum).
        g = (x, p) -> p[1] - x[1]
        method = FORM(PMA(β_target=2.0, nstarts=2))
        @test performance_measure(g, [2.0], tr; method=method) ≈ exp(-2.0) - 2.0 atol = 1e-4
        g_ad = Zygote.gradient(x -> performance_measure(g, x, tr; method=method), [2.0])[1]
        @test g_ad ≈ [-1.0] rtol = 1e-4
    end

    @testset "Custom jacobian" begin
        f = (x, p) -> [p[1] * x[1], p[2] - x[2]]
        jac = (f, x, p) -> [x[1] 0.0; 0.0 1.0]
        p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        x = [1.5, 2.0]
        @test reliability_index(f, x, p; jacobian=jac) ≈ 2.0 atol = 1e-4
        g_ad = Zygote.gradient(x -> reliability_index(f, x, p; jacobian=jac), x)[1]
        g_fd = FiniteDifferences.grad(
            central_fdm(5, 1), x -> reliability_index(f, x, p; jacobian=jac), x
        )[1]
        @test g_ad ≈ g_fd rtol = 1e-5 atol = 1e-6
        # the custom jacobian must also be differentiable (it is, being linear)
        rf = RandomFunction(f, p; jacobian=jac)
        d = rf(x)
        # p* = [0, x2], J = diag(x1, 1): muf = [0, −x2], covf = diag(x1², 1)
        @test mean(d) ≈ [0.0, -2.0] atol = 1e-4
        @test cov(d) ≈ [2.25 0.0; 0.0 1.0] atol = 1e-4
        g_ad2 = Zygote.gradient(x -> sum(mean(rf(x))), x)[1]
        g_fd2 = FiniteDifferences.grad(central_fdm(5, 1), x -> sum(mean(rf(x))), x)[1]
        @test g_ad2 ≈ g_fd2 rtol = 1e-5 atol = 1e-6
    end

    @testset "Multistart RIA" begin
        f = (x, p) -> p[1] - 2.0
        p = MvNormal(zeros(1), ones(1, 1))
        @test reliability_index(f, [0.0], p; method=FORM(RIA(nstarts=3))) ≈ 2.0 atol = 1e-4
    end

    @testset "Textbook distribution transformations" begin
        # The classic reliability-analysis table: u = Φ⁻¹(F_X(x)). Each transform
        # must match the textbook formula, round-trip, and satisfy Φ(u) = F_X(x).
        Φ(p) = cdf(Normal(), p)
        Φ⁻¹(p) = quantile(Normal(), p)
        cases = [
            (Normal(1.5, 0.7), x -> (x - 1.5) / 0.7),
            (LogNormal(0.3, 0.5), x -> (log(x) - 0.3) / 0.5),
            (Uniform(-2.0, 5.0), x -> Φ⁻¹((x + 2.0) / 7.0)),
            (Exponential(1.3), x -> Φ⁻¹(1 - exp(-x / 1.3))),
            (Gumbel(2.0, 0.8), x -> Φ⁻¹(exp(-exp(-(x - 2.0) / 0.8)))),
            (Weibull(2.5, 1.1), x -> Φ⁻¹(1 - exp(-(x / 1.1)^2.5))),
            (Frechet(3.0, 1.4), x -> Φ⁻¹(exp(-(x / 1.4)^(-3.0)))),
            (Gamma(2.0, 1.5), x -> Φ⁻¹(cdf(Gamma(2.0, 1.5), x))),
        ]
        for (d, formula) in cases
            tr = StandardNormalTransform(d)
            for x in quantile.(d, [0.05, 0.3, 0.7, 0.95])
                @test tr.to_u(x)[1] ≈ formula(x) atol = 1e-8
                @test tr.from_u([tr.to_u(x)[1]]) ≈ x rtol = 1e-8
                @test Φ(tr.to_u(x)[1]) ≈ cdf(d, x) atol = 1e-8
            end
        end
        # Rayleigh(σ) ≡ Weibull(shape = 2, scale = √2 σ): F = 1 − exp(−x²/(2σ²))
        σr = 1.7
        tr = StandardNormalTransform(Weibull(2.0, sqrt(2) * σr))
        @test tr.to_u(2.1)[1] ≈ Φ⁻¹(1 - exp(-2.1^2 / (2σr^2))) atol = 1e-8
        # univariate transforms plug into the reliability analysis
        g = (x, p) -> p - 3.0 - x[1]
        pln = LogNormal(0.0, 1.0)
        @test reliability_index(g, [0.0], pln) ≈ log(3) atol = 1e-5
        @test getp0(g, [0.0], pln) ≈ 3.0 atol = 1e-5
        g_ad = Zygote.gradient(x -> reliability_index(g, x, pln), [0.0])[1]
        g_fd = FiniteDifferences.grad(
            central_fdm(5, 1), x -> reliability_index(g, x, pln), [0.0]
        )[1]
        @test g_ad ≈ g_fd rtol = 1e-5
        # the linearized output distribution is differentiable for the
        # closed-form transforms (Uniform here)
        pu = Uniform(-1.0, 3.0)
        f = (x, p) -> p * x[1] + 1.0
        rfu = RandomFunction(f, pu)
        g_ad2 = Zygote.gradient(x -> mean(rfu(x))[1] + 2 * sqrt(cov(rfu(x))[1, 1]), [1.5])[1]
        g_fd2 = FiniteDifferences.grad(
            central_fdm(5, 1), x -> mean(rfu(x))[1] + 2 * sqrt(cov(rfu(x))[1, 1]), [1.5]
        )[1]
        @test g_ad2 ≈ g_fd2 rtol = 1e-5
    end

    @testset "Nataf transform" begin
        # normal marginals: the Nataf correlation equals the original one
        nt = nataf_transform([Normal(0.0, 1.0), Normal(0.0, 1.0)], [1.0 0.6; 0.6 1.0])
        L = cholesky([1.0 0.6; 0.6 1.0]).L
        @test nt.from_u([1.0, 0.5]) ≈ L * [1.0, 0.5] atol = 1e-8
        @test nt.to_u([1.0, 0.5]) ≈ L \ [1.0, 0.5] atol = 1e-8
        # uniform marginals: closed form ρ = 2 sin(π ρ₀ / 6)
        ρ0_target = 0.4
        ρ_true = 2 * sin(pi * ρ0_target / 6)
        nt2 = nataf_transform([Uniform(0, 1), Uniform(0, 1)], [1.0 ρ_true; ρ_true 1.0])
        z = [0.5, -0.9]
        @test nt2.to_u(nt2.from_u(z)) ≈ z atol = 1e-6
        # the solved equivalent-normal correlation must satisfy the Nataf
        # equation: u ~ N(0, I) through from_u must give parameters with the
        # target correlation ρ_true
        Random.seed!(7)
        U = rand(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)), 200_000)
        P = map(eachcol(U)) do u
            nt2.from_u(u)
        end
        P = reduce(hcat, P)
        @test cor(P[1, :], P[2, :]) ≈ ρ_true rtol = 0.02
        # reliability analysis through the Nataf transform
        g = (x, p) -> p[1] + p[2] - x[1]
        β = reliability_index(g, [3.0], nt)
        @test β ≈ 3.0 / sqrt(2 + 2 * 0.6) atol = 1e-4
        # the linearized output distribution through the Nataf transform
        # (closed-form quantiles keep the from_u mapping differentiable)
        rfn = RandomFunction(g, nt)
        d = rfn([3.0])
        @test mean(d)[1] ≈ -3.0 atol = 1e-4
        @test cov(d)[1, 1] ≈ 3.2 atol = 1e-4
        g_ad = Zygote.gradient(x -> mean(rfn(x))[1] + 2 * sqrt(cov(rfn(x))[1, 1]), [3.0])[1]
        g_fd = FiniteDifferences.grad(
            central_fdm(5, 1), x -> mean(rfn(x))[1] + 2 * sqrt(cov(rfn(x))[1, 1]), [3.0]
        )[1]
        @test g_ad ≈ g_fd rtol = 1e-5
    end

    @testset "Error paths and edge cases" begin
        p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        # PMA requires a scalar limit state
        @test_throws ArgumentError performance_measure(
            (x, p) -> [p[1] - 1.0, p[2] - 1.0], [0.0], p; method=FORM(PMA(β_target=2.0))
        )
        # wrong method combinations
        @test_throws ArgumentError reliability_index(
            (x, p) -> p[1] - 1.0, [0.0], p; method=FORM(PMA(β_target=2.0))
        )
        @test_throws ArgumentError performance_measure(
            (x, p) -> p[1] - 1.0, [0.0], p; method=FORM(RIA())
        )
        @test_throws ArgumentError reliability_constraint(
            (x, p) -> p[1] - 1.0,
            p,
            2.0;
            method=RIA(), # not wrapped in FORM
        )
        # RandomFunction does not accept SORM methods
        @test_throws ArgumentError RandomFunction((x, p) -> p[1] - 1.0, p, SORM())
        # unsupported multivariate distributions
        @test_throws ArgumentError RandomFunction(
            (x, p) -> p[1] - 1.0, MvLogNormal(zeros(2), Matrix{Float64}(I, 2, 2))
        )
        # Nataf correlation clamping: mixed marginals cannot reach ρ = 0.95
        # (max ≈ 0.866), so the equivalent-normal correlation is clamped and the
        # transform still round-trips
        ntc = nataf_transform([Uniform(0.0, 1.0), Exponential(1.0)], [1.0 0.95; 0.95 1.0])
        z = [0.3, -0.8]
        @test ntc.to_u(ntc.from_u(z)) ≈ z atol = 1e-6
    end

    @testset "ReliabilityOptimization.jl" begin
        # test function - y is random. The linearization Jacobian is xᵀ at the
        # MPP y* = 2x/‖x‖², so the problem is well-posed (a vanishing Jacobian,
        # as with a ‖x + y‖² limit state, makes the linearized covariance
        # singular and finite-difference checks meaningless).
        for pol in [(x, y) -> [sum(x .* y) - 2.0], (x, y) -> sum(x .* y) - 2.0]
            # input value to be used in example
            x = [2.0, 3.0, 6.0]
            # wrap original function in RandomFunction struct
            y = MvNormal(zeros(3), Diagonal(ones(3)))
            rf = RandomFunction(pol, y)
            rf = RandomFunction(pol, y, FORM(RIA()))
            # call wrapper with example input
            d = rf(x)
            function obj(x)
                dist = rf(x)
                mean(dist)[1] + 2 * sqrt(cov(dist)[1, 1])
            end
            obj(x)
            g1 = FiniteDifferences.grad(central_fdm(5, 1), obj, x)[1]
            g2 = Zygote.gradient(obj, x)[1]
            @test norm(g1 - g2) < 1e-7
        end
    end

    @testset "Multi-material TopOpt example" begin
        include("example.jl")
    end
end # GROUP in ("All", "Tests")

if GROUP in ("All", "Aqua_Tests")
    @testset "Aqua" begin
        include("aqua.jl")
    end
end
