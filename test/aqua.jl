@testitem "Aqua" begin
    using Aqua, ReliabilityOptimization

    # All Aqua quality gates are enabled. No tests are skipped.
    Aqua.test_all(ReliabilityOptimization)
end
