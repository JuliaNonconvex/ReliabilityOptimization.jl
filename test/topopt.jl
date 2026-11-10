@testitem "Multi-material TopOpt example" begin
    using Test, Pkg
    # TopOpt's dependency chain still targets the pre-Core-2 `Nonconvex`
    # metapackage, and NonconvexTOBS 0.2.3 (Core 2 support) is not yet
    # registered, so neither resolves next to NonconvexCore 2. Skip the
    # example until the upstream releases land; re-add TopOpt and
    # NonconvexTOBS to the test environment then.
    deps = Pkg.project().dependencies
    if !(haskey(deps, "TopOpt") && haskey(deps, "NonconvexTOBS"))
        @test_skip "TopOpt/NonconvexTOBS do not yet resolve against NonconvexCore 2 (upstream registration pending)"
    else
        using ReliabilityOptimization, NonconvexTOBS, ChainRulesCore
        using TopOpt, Zygote, FiniteDifferences

        include(joinpath(@__DIR__, "example.jl"))
    end
end
