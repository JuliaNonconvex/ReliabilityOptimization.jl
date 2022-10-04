using Uncertainty, Test, FiniteDifferences, Zygote

@testset "Uncertainty.jl" begin
    # test function - y is random
    pol(x, y) = [norm(x + y)^2]
    # input value to be used in example
    x = [2.0, 3.0, 6.0]
    # wrap original function in RandomFunction struct
    y = MvNormal(zeros(3), Diagonal(ones(3)))
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
