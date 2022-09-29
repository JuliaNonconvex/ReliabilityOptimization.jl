module Uncertainty

using ImplicitDifferentiation, Zygote, TopOpt, ChainRulesCore, LinearAlgebra
using UnPack, Nonconvex, Statistics, GLMakie, Distributions, Reexport
Nonconvex.@load Ipopt
@reexport using LinearAlgebra

struct RandomFunction{F,P}
  f::F
  p::P
	method
end

function getp0(randFunc::RandomFunction, x)
  @unpack f, p, method = randFunc
	lag_multipliers = 0.0
	function forward(x)
		obj = p -> p'*p # gets an objective function of p
		constr = p -> f(x, p) # gets the constraint on p
		# solve the RIA problem to find p0
		innerOptModel = Nonconvex.Model(obj)
		addvar!(innerOptModel, zeros(size(p)), fill(10.0, size(p)))
		add_eq_constraint!(innerOptModel, constr)
		result = optimize(innerOptModel, IpoptAlg(), mean(p), options = IpoptOptions())
		lag_multipliers = result # get the Lagrangian multipliers
		return result.minimizer
	end
	function kkt_conditions(x, p)
		constr = p -> f(x, p) # gets the constraint on p
		return 2 * p + Zygote.gradient(constr, p)[1] * lag_multipliers
	end
	implicit_f = ImplicitFunction(forward, kkt_conditions)
	return implicit_f(x)
end

struct FORM{M}
	method::M
end
struct RIA end

function RandomFunction(f, p)
	return RandomFunction(f, p, FORM(RIA()))
end
RandomFunction(f, p; method) = RandomFunction(f, p, method)

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

export RandomFunction
export FORM
export MvNormal

end