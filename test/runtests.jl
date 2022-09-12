using Uncertainty, Test, ImplicitDifferentiation, Zygote, TopOpt, ChainRulesCore

@testset "Uncertainty.jl" begin
    ## 1) ImplicitDifferentiation.jl -> ensure differentiability
    zero_gradient(x, y) = 2(y - x) # place-holders from ImplicitDifferentiation.jl docs
    implicit = ImplicitFunction(???, zero_gradient)
    x = rand(3)
    implicit(x)
    Zygote.jacobian(implicit, x)[1]
    ## 2) solve optimization -> linear approximation around p0 (MPP)
    # ???
    ## linear approximation around p0 and base distribution of parameters -> RandomFunction struct
    E_p(f(x; p)) = f(x; p = p0) + df(x; p = p0)/dp * mu_p - df(x; p = p0)/dp * p0
    p ~ MvNormal(mu_p, cov_p)
    mu(x) = f(x; p = p0(x)) + df(x; p = p0(x))/dp * mu_p - df(x; p = p0(x))/dp * p0
    cov(x) = (df(x; p = p0)/dp) * cov_p * (df(x; p = p0)/dp)'
    f(x; p) ~ MvNormal(mu(x), cov(x))
    rf = RandomFunction(f, MvNormal(zeros(3), I(3)), method = FORM(RIA(g)))
    ## RandomFunction struct -> mean and cov functions
    function obj(x)
        dist = rf(x)
        mean(dist) + 2 * std(dist)
    end
    ## obj(x) to be used in actual application (e.g. TO)
    # TopOpt.jl code
end