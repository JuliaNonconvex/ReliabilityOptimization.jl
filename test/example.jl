using ReliabilityOptimization,
    Test, NonconvexTOBS, ChainRulesCore, TopOpt, Zygote, FiniteDifferences

const densities = [0.0, 0.5, 1.0] # for mass calculation
const nmats = 3 # number of materials
const v = 0.3 # Poisson’s ratio
const f = 1.0 # downward force
const problemSize = (4, 4) # size of rectangular mesh
const elSize = (1.0, 1.0) # size of QUAD4 elements
# Point load cantilever problem to be solved
problem = PointLoadCantilever(problemSize, elSize, 1.0, v, f)
ncells = TopOpt.getncells(problem) # Number of elements in mesh
solver = FEASolver(DirectSolver, problem; xmin=0.0)
filter = DensityFilterFun(solver; rmin=3.0) # filter to avoid checkerboarding
M = 1 / nmats / 2 # mass fraction
comp = ComplianceFun(solver) # function that returns compliance
penalty = PowerPenaltyFun(3.0) # SIMP penalty
# Young’s modulii of air (void) + 2 materials
const avgEs = [1e-6, 0.5, 2.0]
# since first E is close to zero,
# can the deviation make the final value negative?
logEs = MvNormal(log.(avgEs), Matrix(Diagonal(0.1 .* abs.(log.(avgEs)))))
# 'Original' function. At least one input is random.
# In this example, Es is the random input.
function uncertainComp(x, logEs)
    Es = exp.(logEs)
    # interpolation of properties between materials
    interp = MaterialInterpolationFun(Es, penalty)
    return comp(interp(filter(tounit(MultiMaterialVariablesFun(x, nmats)))))
    # return sum(x) + sum(Es)
end
# wrap original function in RandomFunction struct
rf = RandomFunction(uncertainComp, logEs, FORM(RIA()))
# initial homogeneous distribution of pseudo-densities
x0 = fill(M, ncells * (length(logEs) - 1))
# call wrapper with example input
# (Returns propability distribution of the objective for current point)
d = rf(x0)
# mass constraint
constr_penalty = PowerPenaltyFun(1.0)
constr_interp = MaterialInterpolationFun(densities, constr_penalty)
constr = x -> begin
    ρs = constr_interp(MultiMaterialVariablesFun(x, nmats))
    return sum(ρs.x) / ncells - 0.3 # elements have unit volumes
end
function obj(x) # objective for TO problem
    dist = rf(x)
    return mean(dist)[1] + 2 * sqrt(cov(dist)[1, 1])
end
obj(x0)
g1 = Zygote.gradient(obj, x0)[1]
g2 = FiniteDifferences.grad(central_fdm(5, 1), obj, x0)[1]
norm(g1 - g2)

cg1 = Zygote.gradient(constr, x0)[1]
cg2 = FiniteDifferences.grad(central_fdm(5, 1), constr, x0)[1]
norm(cg1 - cg2)

m = Model(obj) # create optimization model
addvar!(m, zeros(length(x0)), ones(length(x0))) # setup optimization variables
add_ineq_constraint!(m, constr) # setup volume inequality constraint

@time r = optimize(m, TOBSAlg(), x0; options=TOBSOptions())
# @time r = Nonconvex.optimize(m, IpoptAlg(), x0; options = IpoptOptions())
# @time r = Nonconvex.optimize(m, MMA87(), x0; options = MMAOptions())
