@testitem "Multi-material TopOpt example" begin
    using Test, ReliabilityOptimization, NonconvexTOBS, ChainRulesCore
    using TopOpt, Zygote, FiniteDifferences

    include(joinpath(@__DIR__, "example.jl"))
end
