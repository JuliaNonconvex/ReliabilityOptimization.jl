using Uncertainty, Test, ImplicitDifferentiation, Zygote, TopOpt, ChainRulesCore, UnPack

@testset "Uncertainty.jl" begin
    ## 1) ImplicitDifferentiation.jl -> ensure differentiability
    ## 2) solve optimization -> linear approximation around p0 (MPP)
    ## 3) linear approximation around p0 and base distribution of parameters -> RandomFunction struct
    ## 4) RandomFunction struct -> mean and cov functions
    ## 5) obj(x) to be used in actual application (e.g. TO)
    function myFunc(x; p)
    end
    function g()
    end
    rf = RandomFunction(f, MvNormal(zeros(3), I(3)), g, method = FORM(RIA(g)))
    function obj(x)
        dist = rf(x)
        mean(dist) + 2 * std(dist)
    end
end