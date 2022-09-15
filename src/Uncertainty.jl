module Uncertainty

function FORM()

end

function RIA()

end

function getp0(randFunc::RandomFunction, x)
  @unpack f, p, conditions, method = randFunc
	innerOptModel = Nonconvex.Model(f) # Wrap original function f in Nonconvex.jl optimization model
	KKT = [p_ -> Zygote.gradient(p -> cond(x, p), p_)[1] for cond in conditions] # formulate KKT conditions
	[add_eq_constraint!(innerOptModel, KKTᵢ) for KKTᵢ in KKT] # add KKTs to optimization model
	implicit = ImplicitFunction( # implicit wrapper to ensure differentiability
		optimize(innerOptModel, IpoptAlg(), mean.(p), options = IpoptOptions()),
		KKT, method)
	return implicit(x)
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