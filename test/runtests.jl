using Uncertainty, Test

# @testset "Uncertainty.jl" begin
    # function with at least one random input
    pol(x, y) = norm(x)^2 + norm(y)^2
    # define deterministic inputs, in case there are any
    const y = [4; 5; 8]
    # input value to be used in example
    _x = [2; 3; 6]
    # wrap original function in RandomFunction struct
    rf = RandomFunction(
        pol, MvNormal(_x, Diagonal(ones(3))), method = FORM(:RIA)
    )
    # call wrapper with example input
    d = rf(_x)
    function obj(x)
        dist = rf(x)
        mean(dist) + 2 * std(dist)
    end
    
# end