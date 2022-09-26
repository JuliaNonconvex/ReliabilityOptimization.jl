module Uncertainty

function FORM()

end

function RIA()

end

function getp0(randFunc::RandomFunction, x)
  @unpack f, p, conditions, method = randFunc
	lag_multipliers = 0.0
	function forward(x)
		obj = get_obj(x) # gets an objective function of p
		constr = get_constr(x) # gets the constraint on p
		# solve the RIA problem to find p0
		innerOptModel = Nonconvex.Model(obj)
		add_eq_constraint!(innerOptModel, constr)
		result = optimize(innerOptModel, IpoptAlg(), mean.(p), options = IpoptOptions())
		p0 = result.minimizer
		# get the Lagrangian multipliers and store them in lag_multipliers
		grads = Zygote.gradient((x, p0) -> f(x, p), [x, p0])[1]
		jac = Zygote.jacobian((x, p0) -> conditions(x, p), [x, p0])[1]
		lag_multipliers = grads*inv(jac)
		return p0
	end
	function kkt_conditions(x, p)
		obj = get_obj(x) # gets an objective function of p
		constr = get_constr(x) # gets the constraint on p
		return Zygote.gradient(obj, p) + Zygote.jacobian(constr, p)' * lag_multipliers
	end
	implicit_f = ImplicitFunction(forward, kkt_conditions)
	return implicit_f(x)
end

struct RandomFunction{F,P}
  f::F
  p::P
	conditions::Array{F}
	method::F
end

function (f::RandomFunction)(x)
	mup = mean(f.p)
	covp = cov(f.p)
	p0 = getp0(f, x) # should use ImplicitFunctions to be differentiable and efficient
	dfdp0 = Zygote.jacobian(p -> f.f(x, p), p0)[1]
	fp0 = f.f(x, p0)
	muf = fp0 + dfdp0 * (mup - p0)
	covf = dfdp0 * covp * dfdp0'
	return MvNormal(muf, covf)
end

end