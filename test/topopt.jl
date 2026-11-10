@testitem "Multi-material TopOpt example" begin
    using ReliabilityOptimization, Test, NonconvexTOBS, ChainRulesCore
    using TopOpt, Zygote, FiniteDifferences

    include("example.jl")
end
