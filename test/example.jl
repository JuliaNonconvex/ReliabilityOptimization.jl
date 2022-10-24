using ReliabilityOptimization, Test, TopOpt, NonconvexTOBS, ChainRulesCore

const densities = [0.0, 0.5, 1.0] # for mass calculation
const nmats = 3 # number of materials
const v = 0.3 # Poisson’s ratio
const f = 1.0 # downward force
const problemSize = (160, 40) # size of rectangular mesh
const elSize = (1.0, 1.0) # size of QUAD4 elements
# Point load cantilever problem to be solved
problem = PointLoadCantilever(Val{:Linear}, problemSize, elSize, 1.0, v, f)
ncells = TopOpt.getncells(problem) # Number of elements in mesh
solver = FEASolver(Direct, problem; xmin = 0.0)
filter = DensityFilter(solver; rmin = 3.0) # filter to avoid checkerboarding
M = 1 / nmats / 2 # mass fraction
comp = Compliance(solver) # function that returns compliance
penalty = TopOpt.PowerPenalty(3.0) # SIMP penalty
# Young’s modulii of air (void) + 2 materials
const avgEs = [1e-6, 0.5, 2.0]
# since first E is close to zero,
# can the deviation make the final value negative?
Es = MvNormal(avgEs, Diagonal(0.1 .* avgEs))
# 'Original' function. At least one input is random.
# In this example, Es is the random input.
function multMatVar(x, nmats)
  out = []
  @ignore_derivatives out = MultiMaterialVariables(x, nmats)
  return out
end
function uncertainComp(Es, x)
  # interpolation of properties between materials
  interp = MaterialInterpolation(Es, penalty)
  [multMatVar(x, nmats) |> interp |> filter |> comp]
end
# wrap original function in RandomFunction struct
rf = RandomFunction(uncertainComp, Es, FORM(RIA()))
# initial homogeneous distribution of pseudo-densities
x0 = fill(M, ncells * (length(Es) - 1))
# call wrapper with example input
d = rf(x0)