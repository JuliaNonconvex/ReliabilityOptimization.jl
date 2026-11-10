@testitem "Multi-material TopOpt example" begin
    using Test, Pkg
    # TopOpt's dependency chain still targets the pre-Core-2 `Nonconvex`
    # metapackage, so it cannot be resolved next to NonconvexCore 2. Skip the
    # example until the upstream TopOpt release that supports Core 2 merges;
    # re-add TopOpt to the test environment then.
    if !haskey(Pkg.project().dependencies, "TopOpt")
        @test_skip "TopOpt does not yet resolve against NonconvexCore 2 (upstream registration pending)"
    else
        using ReliabilityOptimization, NonconvexTOBS, ChainRulesCore
        using TopOpt, Zygote, FiniteDifferences

        include(joinpath(@__DIR__, "example.jl"))
    end
end
