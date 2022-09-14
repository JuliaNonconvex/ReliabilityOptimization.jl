module Uncertainty

function FORM()

end

function RIA()

end

function getp0(f, x)
  
	@unPack forward, params, conds, method = f
  return ImplicitFunction(forward, (x, p) -> conds, method)(x, params)

end

struct RandomFunction{F,P}
  f::F
  p::P
	conditions::F
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