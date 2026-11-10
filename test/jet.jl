@testitem "JET" begin
    using JET, ReliabilityOptimization
    JET.test_package(ReliabilityOptimization; target_modules=(ReliabilityOptimization,))
end
