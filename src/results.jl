using Nonconvex
Nonconvex.@load Ipopt

f(x) = sqrt(x[2])
g(x, a, b) = (a * x[1] + b)^3 - x[2]

model = Model(f)
addvar!(model, [0.0, 0.0], [10.0, 10.0])
add_ineq_constraint!(model, x -> g(x, 2, 0))
add_ineq_constraint!(model, x -> g(x, -1, 1))

alg = IpoptAlg()
options = IpoptOptions()
r = optimize(model, alg, [1.0, 1.0], options = options)
propertynames(r)
println(propertynames(r.problem))
typeof(r.problem.intermediate)
propertynames(r.problem.intermediate)
r.problem
