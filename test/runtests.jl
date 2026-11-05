using ReliabilityOptimization, Test, FiniteDifferences, Zygote
using Distributions

@testset "RIA on a linear limit state" begin
    # Regression test: with the limit-state value c a free variable, the MPP
    # used to land off the limit-state surface (beta = sqrt(2) for g = p - 2
    # with p ~ N(0, 1)). The Hasofer-Lind solution is p = 2, beta = 2.
    f = (x, p) -> p[1] - 2.0
    p = MvNormal(zeros(1), ones(1, 1))
    p0 = ReliabilityOptimization.getp0(f, [0.0], p, FORM(RIA()))
    @test p0 ≈ [2.0] atol = 1e-4
end

@testset "ReliabilityOptimization.jl" begin
    # test function - y is random
    for pol in [(x, y) -> [norm(x + y)^2], (x, y) -> norm(x + y)^2]
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
