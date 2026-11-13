# Coverage-focused tests for ReliabilityOptimization internals and API edge
# paths that are not exercised by the main test suite (TestItems style).
# Each @testitem is self-contained and only runs cheap, deterministic checks.

@testitem "transforms: constructors, validation errors and helpers" begin
    using ReliabilityOptimization, Test, LinearAlgebra, Distributions

    # StandardNormalTransform: positional-n constructor validation errors
    @test_throws ArgumentError StandardNormalTransform(
        u -> u, p -> p, 2; μ=[1.0], Σ=Matrix{Float64}(I, 2, 2)
    )
    @test_throws ArgumentError StandardNormalTransform(
        u -> u, p -> p, 2; μ=[1.0, 2.0], Σ=ones(1, 1)
    )

    # positional-n constructor happy path
    tr = StandardNormalTransform(
        u -> 2.0 .* u, p -> p ./ 2.0, 2; μ=zeros(2), Σ=Matrix{Float64}(I, 2, 2)
    )
    @test ReliabilityOptimization.dim(tr) ≈ 2
    @test tr.from_u([1.0, -1.0]) ≈ [2.0, -2.0]
    @test tr.to_u([2.0, -2.0]) ≈ [1.0, -1.0]

    # custom μ/Σ overrides for the distribution-based constructors
    trn = StandardNormalTransform(Normal(1.0, 2.0); μ=[0.5], Σ=reshape([4.5], 1, 1))
    @test trn.μ ≈ [0.5]
    @test trn.Σ ≈ reshape([4.5], 1, 1)
    @test trn.n ≈ 1
    @test trn.to_u(3.0) ≈ [(3.0 - 1.0) / 2.0]
    trln = StandardNormalTransform(LogNormal(0.0, 0.5); μ=[2.0], Σ=reshape([7.0], 1, 1))
    @test trln.μ ≈ [2.0]
    @test trln.Σ ≈ reshape([7.0], 1, 1)
    @test trln.to_u(3.0) ≈ [log(3.0) / 0.5]

    # generic (non-closed-form) univariate distribution path: Beta
    trb = StandardNormalTransform(Beta(2.0, 3.0))
    @test trb.to_u(trb.from_u([0.7])) ≈ [0.7] atol = 1e-8
    @test cdf(Normal(), trb.to_u(0.4)[1]) ≈ cdf(Beta(2.0, 3.0), 0.4) atol = 1e-8

    # AffineTransform: direct construction, both univariate branches
    tra = ReliabilityOptimization.AffineTransform(
        [1.0], reshape([2.0], 1, 1), reshape([4.0], 1, 1), true
    )
    @test ReliabilityOptimization.dim(tra) ≈ 1
    @test ReliabilityOptimization.from_u(tra, [0.5]) === 2.0
    @test ReliabilityOptimization.to_u(tra, 3.0) ≈ [1.0]
    trm = ReliabilityOptimization.AffineTransform(
        [1.0, 2.0], [1.0 0.0; 0.5 1.0], [1.0 0.5; 0.5 1.25], false
    )
    @test ReliabilityOptimization.from_u(trm, [1.0, 0.0]) ≈ [2.0, 2.5]
    @test ReliabilityOptimization.to_u(trm, [2.0, 2.5]) ≈ [1.0, 0.0] atol = 1e-12

    # standard-normal CDF helpers
    @test ReliabilityOptimization._normcdf(0.0) ≈ 0.5
    @test ReliabilityOptimization._norminvcdf(0.5) ≈ 0.0 atol = 1e-12
    @test ReliabilityOptimization._norminvcdf(ReliabilityOptimization._normcdf(0.7)) ≈ 0.7 atol =
        1e-10
    @test ReliabilityOptimization._clampcdf(0.0) ≈ 1e-15
    @test ReliabilityOptimization._clampcdf(1.0) ≈ 1 - 1e-15
    @test ReliabilityOptimization._clampcdf(0.3) ≈ 0.3

    # Gauss-Hermite nodes/weights via Golub-Welsch
    nodes, weights = ReliabilityOptimization._gauss_hermite(10)
    @test length(nodes) ≈ 10 && length(weights) == 10
    @test issorted(nodes)
    @test all(weights .> 0)
    @test sum(weights) ≈ 1.0
    @test abs(sum(weights .* nodes)) < 1e-12
    @test sum(weights .* nodes .^ 2) ≈ 1.0
    n1, w1 = ReliabilityOptimization._gauss_hermite(1)
    @test n1 ≈ [0.0] && w1 == [1.0]
end

@testitem "nataf_transform: options, dimensions and clamping" begin
    using ReliabilityOptimization, Test, LinearAlgebra, Distributions

    # wrong correlation-matrix size
    @test_throws ArgumentError nataf_transform(
        [Normal(0.0, 1.0), Normal(0.0, 1.0)], [1.0 0.3]
    )

    # single marginal
    nt1 = nataf_transform([Normal(1.0, 2.0)], reshape([1.0], 1, 1))
    @test nt1.from_u([0.7]) ≈ [0.7 * 2.0 + 1.0]
    @test nt1.to_u([2.4]) ≈ [0.7] atol = 1e-8

    # gh_n option; normal marginals solve ρ₀ = ρ exactly
    nt = nataf_transform([Normal(0.0, 1.0), Normal(0.0, 1.0)], [1.0 0.6; 0.6 1.0]; gh_n=8)
    L = cholesky([1.0 0.6; 0.6 1.0]).L
    @test nt.from_u([0.3, -0.4]) ≈ L * [0.3, -0.4] atol = 1e-8
    @test nt.to_u(nt.from_u([0.3, -0.4])) ≈ [0.3, -0.4] atol = 1e-8

    # three marginals: exercises all off-diagonal pairs
    R = [1.0 0.3 0.2; 0.3 1.0 0.1; 0.2 0.1 1.0]
    nt3 = nataf_transform([Normal(0.0, 1.0) for _ in 1:3], R)
    z = [0.5, -0.7, 0.9]
    @test nt3.to_u(nt3.from_u(z)) ≈ z atol = 1e-8

    # mean/covariance of the joint parameters: μ = means, Σ = D ρ D
    σs = [2.0, sqrt(1 / 12)]
    ntm = nataf_transform([Normal(1.0, 2.0), Uniform(0.0, 1.0)], [1.0 0.3; 0.3 1.0])
    @test ntm.μ ≈ [1.0, 0.5]
    @test ntm.Σ ≈ [σs[1]^2 σs[1] * σs[2] * 0.3; σs[1] * σs[2] * 0.3 σs[2]^2]

    # target correlation below the achievable range: warns and clamps
    ntc = @test_logs (:warn, r"outside the achievable range") match_mode = :any nataf_transform(
        [Uniform(0.0, 1.0), Exponential(1.0)], [1.0 -0.95; -0.95 1.0]
    )
    z2 = [0.3, -0.8]
    @test ntc.to_u(ntc.from_u(z2)) ≈ z2 atol = 1e-6
end

@testitem "finite-difference jacobian and hessian helpers" begin
    using ReliabilityOptimization,
        Test, LinearAlgebra, FiniteDifferences, ChainRulesCore, Zygote, ForwardDiff

    # _vec
    @test ReliabilityOptimization._vec(1.5) ≈ [1.5]
    v = [1.0, 2.0]
    @test ReliabilityOptimization._vec(v) === v

    # _resolve_jac
    @test ReliabilityOptimization._resolve_jac(nothing) ===
        ReliabilityOptimization._jacobian
    jm = (f, x, p) -> ones(1, 1)
    @test ReliabilityOptimization._resolve_jac(jm) === jm

    # _scalar_limit_state
    @test ReliabilityOptimization._scalar_limit_state(2.0) ≈ 2.0
    @test ReliabilityOptimization._scalar_limit_state([2.0]) ≈ 2.0
    @test_throws ArgumentError ReliabilityOptimization._scalar_limit_state([1.0, 2.0])

    # _jacobian shape branches: scalar output / scalar param -> 1×1
    J1 = ReliabilityOptimization._jacobian((x, p) -> p * x[1]^2, [2.0], 1.5)
    @test size(J1) == (1, 1)
    @test J1[1, 1] ≈ 4.0 atol = 1e-8
    # vector output / scalar param -> m×1 (reshape branch)
    J2 = ReliabilityOptimization._jacobian((x, p) -> [p * x[1], p^2], [2.0], 1.5)
    @test size(J2) == (2, 1)
    @test J2 ≈ [2.0; 3.0] atol = 1e-8
    # scalar output / vector param
    J3 = ReliabilityOptimization._jacobian((x, p) -> p[1] * x[1] + 2p[2], [2.0], [1.0, 3.0])
    @test size(J3) == (1, 2)
    @test J3 ≈ [2.0 2.0] atol = 1e-8

    # rrule with a scalar parameter: the parameter cotangent stays a scalar
    fs = (x, p) -> p * x[1]^2 + p^2 * x[2]
    x = [1.5, 0.5]
    p0 = 0.7
    J = ReliabilityOptimization._jacobian(fs, x, p0)
    Δm = [0.6;;]
    val, pb = ChainRulesCore.rrule(ReliabilityOptimization._jacobian, fs, x, p0)
    @test val ≈ J
    _, _, x̄, p̄ = pb(Δm)
    @test p̄ isa Real
    loss = (xx, pp) -> dot(Δm, ReliabilityOptimization._jacobian(fs, xx, pp))
    x̄_fd = FiniteDifferences.grad(central_fdm(5, 1), xx -> loss(xx, p0), x)[1]
    p̄_fd = FiniteDifferences.grad(central_fdm(5, 1), pp -> loss(x, pp), p0)[1]
    @test x̄ ≈ x̄_fd rtol = 1e-6 atol = 1e-8
    @test p̄ ≈ p̄_fd rtol = 1e-6 atol = 1e-8

    # a scalar cotangent is filled to the Jacobian shape
    _, _, x̄2, p̄2 = pb(0.6)
    _, _, x̄1, p̄1 = pb([0.6;;])
    @test x̄2 ≈ x̄1 && p̄2 ≈ p̄1

    # scalar design: the rrule must keep the design cotangent scalar (it
    # used to be indexed like a vector, raising a BoundsError) and agree
    # with finite differences and Zygote
    fsc = (x, p) -> p[1] * x^2
    xsc = 1.5
    psc = [0.7]
    Jsc = ReliabilityOptimization._jacobian(fsc, xsc, psc)
    Δsc = [0.6;;]
    valsc, pbsc = ChainRulesCore.rrule(ReliabilityOptimization._jacobian, fsc, xsc, psc)
    @test valsc ≈ Jsc
    _, _, x̄sc, p̄sc = pbsc(Δsc)
    @test x̄sc isa Real
    lossc = (xx, pp) -> dot(Δsc, ReliabilityOptimization._jacobian(fsc, xx, pp))
    x̄sc_fd = FiniteDifferences.grad(central_fdm(5, 1), xx -> lossc(xx, psc), xsc)[1]
    x̄sc_zy = Zygote.gradient(xx -> lossc(xx, psc), xsc)[1]
    @test x̄sc ≈ x̄sc_fd rtol = 1e-6 atol = 1e-8
    @test x̄sc_zy ≈ x̄sc_fd rtol = 1e-6 atol = 1e-8

    # eltype-preserving cotangent buffers: with Float32-valued duals the
    # buffers used to be `zeros(float(eltype(x1)))`, silently promoting the
    # cotangent eltype to Dual{Float64} and truncating nested derivatives
    fdu = (x, p) -> p[1]^2 * x[1]^2 + p[1] * x[2]
    xdu = ForwardDiff.Dual{:τ,Float32,1}[1.3, -0.7]
    pdu = Float32[0.4]
    Jdu = ReliabilityOptimization._jacobian(fdu, xdu, pdu)
    @test Jdu[1] ≈ 2 * 0.4 * 1.69 - 0.7 atol = 1e-3
    Δdu = Float32[0.6;;]
    valdu, pbdu = ChainRulesCore.rrule(ReliabilityOptimization._jacobian, fdu, xdu, pdu)
    @test valdu ≈ Jdu
    _, _, x̄du, p̄du = pbdu(Δdu)
    # The design cotangent keeps the dual eltype of the primals; the
    # parameter cotangent must be able to hold the duals that flow through
    # the dual-valued design (promoted, not truncated to Float64).
    @test eltype(x̄du) === eltype(xdu)
    @test eltype(p̄du) <: ForwardDiff.Dual
    # J = 2 p x[1]^2 + x[2], so ∂(dot(Δ, J))/∂x = Δ * [4 p x[1], 1]
    @test x̄du ≈ [0.6 * 4 * 0.4 * 1.3, 0.6] atol = 1e-3
    # ∂(dot(Δ, J))/∂p = Δ * 2 x[1]^2 = 0.6 * 2 * 1.69
    @test p̄du[1] ≈ 0.6 * 2 * 1.69 atol = 1e-3

    # _hessian_fd: scalar and vector parameter branches
    H1 = ReliabilityOptimization._hessian_fd(q -> q^3 + 2q, 0.5)
    @test size(H1) == (1, 1)
    @test H1[1, 1] ≈ 3.0 atol = 1e-6
    H2 = ReliabilityOptimization._hessian_fd(
        p -> p[1]^2 + p[1] * p[2] + 3p[2]^2, [1.0, -2.0]
    )
    @test H2 ≈ [2.0 1.0; 1.0 6.0] atol = 1e-6

    # _hessian wrapper with scalar parameters
    Hw = ReliabilityOptimization._hessian((x, p) -> p^3, [0.0], 0.5)
    @test Hw[1, 1] ≈ 3.0 atol = 1e-6
end

@testitem "multistart internals, nested solver and constructors" begin
    using ReliabilityOptimization, Test, LinearAlgebra

    # _starts: warm-start copy, axis-aligned starts, sign alternation, radius growth
    warm = [1.0, 2.0]
    s1 = ReliabilityOptimization._starts(warm, 1; radius=2.0)
    @test length(s1) ≈ 1 && s1[1] == warm && s1[1] !== warm
    s0 = ReliabilityOptimization._starts(warm, 0; radius=2.0)
    @test length(s0) ≈ 1 && s0[1] == warm
    s5 = ReliabilityOptimization._starts(warm, 5; radius=2.0)
    @test length(s5) ≈ 5
    @test s5[2] ≈ [-2.0, 0.0]
    @test s5[3] ≈ [0.0, -2.0]
    @test s5[4] ≈ [2.0, 0.0]
    @test s5[5] ≈ [0.0, 2.0]
    s6 = ReliabilityOptimization._starts(warm, 6; radius=2.0)
    @test s6[6] ≈ [-4.0, 0.0] # radius doubles after one pass over the axes
    # n = 1: alternating signs on the single axis
    su = ReliabilityOptimization._starts([0.0], 3; radius=1.0)
    @test su[2] ≈ [-1.0] && su[3] == [1.0]

    # _best_candidate selection rules
    cands = [(viol=0.5, u=[1.0], g=1.0), (viol=0.0, u=[2.0], g=2.0)]
    @test ReliabilityOptimization._best_candidate(cands, c -> dot(c.u, c.u)).u ≈ [2.0]
    cands2 = [(viol=0.0, u=[3.0], g=0.0), (viol=0.0, u=[1.0], g=0.0)]
    @test ReliabilityOptimization._best_candidate(cands2, c -> dot(c.u, c.u)).u ≈ [1.0]
    cands3 = [(viol=0.9, u=[1.0], g=1.0), (viol=0.2, u=[2.0], g=2.0)]
    @test ReliabilityOptimization._best_candidate(cands3, c -> c.g).u ≈ [2.0]
    # exact ties keep the first candidate
    cands4 = [(viol=0.0, u=[1.0], g=1.0), (viol=0.0, u=[1.0], g=1.0)]
    @test ReliabilityOptimization._best_candidate(cands4, c -> c.g).u ≈ [1.0]

    # _solve!: workspaces with and without an `x0` field
    struct MockWS
        x0::Vector{Float64}
        called::Ref{Bool}
    end
    function ReliabilityOptimization.optimize!(ws::MockWS)
        ws.called[] = true
        return ws.x0[1] + ws.x0[2]
    end
    struct NoX0WS end
    ReliabilityOptimization.optimize!(::NoX0WS) = 42.0

    ws = MockWS([0.0, 0.0], Ref(false))
    s = ReliabilityOptimization.NestedSolver(ws, nothing, nothing, [9.0, 9.0])
    @test ReliabilityOptimization._solve!(s, [1.0, 2.0]) ≈ 3.0
    @test ws.called[] && ws.x0 ≈ [1.0, 2.0]
    s2 = ReliabilityOptimization.NestedSolver(NoX0WS(), nothing, nothing, [0.0, 0.0])
    @test ReliabilityOptimization._solve!(s2, [3.0, 4.0]) ≈ 42.0

    # positional RIA constructor and struct fields
    alg = ReliabilityOptimization.IpoptAlg()
    opts = ReliabilityOptimization.IpoptOptions(; print_level=0)
    m = RIA(alg, opts; nstarts=2)
    @test m.optim_alg === alg
    @test m.optim_options === opts
    @test m.nstarts ≈ 2
    # keyword constructors and field layout
    @test RIA().nstarts ≈ 1
    pm = PMA(β_target=2.5)
    @test pm.β_target ≈ 2.5 && pm.nstarts == 1
    @test FORM(pm).method === pm
    srm = SORM(FORM(RIA()))
    @test srm.hessian === nothing
end

@testitem "SORM: constructors, hessian options and degenerate cases" begin
    using ReliabilityOptimization, Test, LinearAlgebra, Distributions

    p1 = MvNormal(zeros(1), ones(1, 1))

    # SORM(method) constructor; limit state through the origin (β = 0)
    @test reliability_index((x, p) -> p[1], [0.0], p1; method=SORM(FORM(RIA()))) ≈ 0.0 atol =
        1e-8

    # zero gradient at the MPP: a custom jacobian returning exactly zero makes
    # the Breitung correction undefined, so SORM falls back to the FORM index
    jz = (f, x, p) -> zeros(1, 1)
    @test reliability_index((x, p) -> p[1] - 2.0, [0.0], p1; method=SORM(), jacobian=jz) ≈
        2.0 atol = 1e-4

    # user Hessian in p-space (AffineTransform branch)
    hp = (f, x, p) -> zeros(1, 1)
    @test reliability_index((x, p) -> p[1] - 2.0, [0.0], p1; method=SORM(hessian=hp)) ≈ 2.0 atol =
        1e-4

    # univariate Normal parameters: scalar-parameter FD Hessian branch, with
    # and without a user Hessian (which receives a scalar p)
    @test reliability_index((x, p) -> p - 2.0, [0.0], Normal(0.0, 1.0); method=SORM()) ≈ 2.0 atol =
        1e-4
    hn = (f, x, p) -> reshape([0.0], 1, 1)
    @test reliability_index(
        (x, p) -> p - 2.0, [0.0], Normal(0.0, 1.0); method=SORM(hessian=hn)
    ) ≈ 2.0 atol = 1e-4

    # user Hessian in u-space (StandardNormalTransform branch)
    tr = StandardNormalTransform(
        u -> exp.(u), p -> log.(p); μ=[exp(0.5)], Σ=reshape([exp(1) * (exp(1) - 1)], 1, 1)
    )
    hu = (f, x, u) -> zeros(1, 1)
    @test reliability_index((x, p) -> p[1] - 3.0, [0.0], tr; method=SORM(hessian=hu)) ≈
        log(3.0) atol = 1e-4
end

@testitem "API: option handling and error paths" begin
    using ReliabilityOptimization, Test, LinearAlgebra, Distributions, Statistics, Zygote

    p2 = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
    g = (x, p) -> 3.0 + x[1] - p[1] - p[2]

    # performance_measure on an RIA-based RandomFunction throws
    rf_ria = RandomFunction(
        (x, p) -> p[1] - 1.0, MvNormal(zeros(1), ones(1, 1)), FORM(RIA())
    )
    @test_throws ArgumentError performance_measure(rf_ria, [0.0])
    # reliability_index on a PMA-based RandomFunction throws
    rf_pma = RandomFunction(
        (x, p) -> p[1] - 1.0, MvNormal(zeros(1), ones(1, 1)), FORM(PMA(β_target=2.0))
    )
    @test_throws ArgumentError reliability_index(rf_pma, [0.0])
    # the same for a PMA method combined with univariate parameters
    @test_throws ArgumentError reliability_index(
        (x, p) -> p - 1.0, [0.0], Normal(0.0, 1.0); method=FORM(PMA(β_target=2.0))
    )
    # SORM cannot be a RandomFunction method nor a constraint method
    @test_throws ArgumentError RandomFunction(
        (x, p) -> p[1] - 1.0, MvNormal(zeros(1), ones(1, 1)), SORM()
    )
    @test_throws ArgumentError performance_measure(g, [0.0], p2; method=SORM())
    @test_throws ArgumentError reliability_constraint(g, p2, 2.0; method=SORM())

    # default PMA method (β_target = 3.0)
    @test performance_measure(g, [0.0], p2) ≈ 3.0 - 3sqrt(2) atol = 1e-4

    # explicit PMA method: the constraint uses the method's β_target, not the
    # positional argument
    c3 = reliability_constraint(g, p2, 2.0; method=FORM(PMA(β_target=3.0)))
    @test c3([0.0]) ≈ 3sqrt(2) - 3.0 atol = 1e-4
    c2 = reliability_constraint(g, p2, 2.0)
    @test c2([0.0]) ≈ 2sqrt(2) - 3.0 atol = 1e-4

    # custom jacobian through the PMA path
    jac = (f, x, p) -> reshape([-1.0, -1.0], 1, 2)
    @test performance_measure(g, [0.0], p2; method=FORM(PMA(β_target=2.0)), jacobian=jac) ≈
        3.0 - 2sqrt(2) atol = 1e-4
    cj = reliability_constraint(g, p2, 2.0; jacobian=jac)
    @test cj([0.0]) ≈ 2sqrt(2) - 3.0 atol = 1e-4

    # PMA-based RandomFunction linearizes at the performance point
    rfp = RandomFunction(g, p2, FORM(PMA(β_target=2.0)); jacobian=jac)
    @test getp0(rfp, [0.0]) ≈ [sqrt(2), sqrt(2)] atol = 1e-4
    d = rfp([0.0])
    @test mean(d)[1] ≈ 3.0 atol = 1e-4
    @test cov(d)[1, 1] ≈ 2.0 atol = 1e-4

    # 4-argument getp0 with a custom jacobian
    @test getp0(
        (x, p) -> p[1] - 2.0,
        [0.0],
        MvNormal(zeros(1), ones(1, 1)),
        FORM(RIA());
        jacobian=(f, x, p) -> reshape([1.0], 1, 1),
    ) ≈ [2.0] atol = 1e-4

    # reliability_index on a RandomFunction built with a custom jacobian
    rfr = RandomFunction(
        (x, p) -> p[1] - 2.0,
        MvNormal(zeros(1), ones(1, 1));
        jacobian=(f, x, p) -> reshape([1.0], 1, 1),
    )
    @test reliability_index(rfr, [0.0]) ≈ 2.0 atol = 1e-4

    # PMA performance measure is differentiable through a custom jacobian
    g_ad = Zygote.gradient(
        x -> performance_measure(g, x, p2; method=FORM(PMA(β_target=2.0)), jacobian=jac),
        [0.0],
    )[1]
    @test g_ad ≈ [1.0] rtol = 1e-4
end

@testitem "RandomFunction chain-rule internals" begin
    using ReliabilityOptimization,
        Test,
        LinearAlgebra,
        Distributions,
        ChainRulesCore,
        Zygote,
        FiniteDifferences,
        Statistics

    # _unthunk on plain values, thunks, tuples, named tuples and tangents
    @test ReliabilityOptimization._unthunk(1.0) === 1.0
    th = ChainRulesCore.@thunk 2.0
    @test ReliabilityOptimization._unthunk(th) ≈ 2.0
    @test ReliabilityOptimization._unthunk((th, 3.0)) == (2.0, 3.0)
    @test ReliabilityOptimization._unthunk((a=th, b=3.0)) == (a=2.0, b=3.0)
    tg = ChainRulesCore.Tangent{ComplexF64}(re=1.0, im=th)
    ut = ReliabilityOptimization._unthunk(tg)
    @test ut isa ChainRulesCore.Tangent && ut.re ≈ 1.0 && ut.im == 2.0

    # the RandomFunction rrule accepts an explicit Tangent cotangent and
    # matches the Zygote-accumulated and finite-difference gradients
    f = (x, p) -> p[1] - x[1]
    p = MvNormal(zeros(1), ones(1, 1))
    rf = RandomFunction(f, p)
    x = [1.0]
    y, back = ChainRulesCore.rrule(rf, x)
    @test y isa MvNormal
    Δ = ChainRulesCore.Tangent{typeof(y)}(μ=[1.0], Σ=ChainRulesCore.ZeroTangent())
    ∂x_direct = back(Δ)[2]
    ∂x_zygote = Zygote.gradient(x -> mean(rf(x))[1], x)[1]
    @test ∂x_direct ≈ ∂x_zygote rtol = 1e-6 atol = 1e-8
    ∂x_fd = FiniteDifferences.grad(central_fdm(5, 1), x -> mean(rf(x))[1], x)[1]
    @test ∂x_zygote ≈ ∂x_fd rtol = 1e-6 atol = 1e-8
end
