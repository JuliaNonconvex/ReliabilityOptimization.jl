module Uncertainty

using ImplicitDifferentiation, Zygote, LinearAlgebra, ChainRulesCore, SparseArrays
using UnPack, NonconvexIpopt, Statistics, Distributions, Reexport, DistributionsAD
@reexport using LinearAlgebra
@reexport using Statistics
export RandomFunction, FORM, RIA, MvNormal

struct RandomFunction{F,P,M}
    f::F
    p::P
    method::M
end

struct FORM{M}
    method::M
end
struct RIA end

function get_forward(f, p, ::FORM{<:RIA})
    function forward(x)
        # gets an objective function of p
        obj = pc -> begin
            _p = pc[1:end-1]
            c = pc[end]
            return _p' * _p + c^2
        end
        # gets the constraint on p
        constr = pc -> begin
            _p = pc[1:end-1]
            c = pc[end]
            f(x, _p) .- c
        end
        # solve the RIA problem to find p0
        # should use and reuse the VecModel
        innerOptModel = Model(obj)
        n = size(p)[1]
        addvar!(innerOptModel, fill(-Inf, n + 1), fill(Inf, n + 1))
        add_eq_constraint!(innerOptModel, constr)
        result = optimize(
            innerOptModel,
            IpoptAlg(),
            [mean(p); 0.0],
            options = IpoptOptions(print_level = 0),
        )
        return vcat(result.minimizer, result.problem.mult_g[1])
    end
    return forward
end

function get_conditions(f, ::FORM{<:RIA})
    function kkt_conditions(x, pcmult)
        p = pcmult[1:end-2]
        c = pcmult[end-1]
        mult = pcmult[end]
        return vcat(
            2 * p + Zygote.pullback(p -> f(x, p), p)[2](mult)[1],
            2c - mult,
            f(x, p) .- c,
        )
    end
end

function get_implicit(f, p, method)
    forward = get_forward(f, p, method)
    kkt_conditions = get_conditions(f, method)
    return ImplicitFunction(forward, kkt_conditions)
end

function getp0(f, x, p, method::FORM{<:RIA})
    implicit_f = get_implicit(f, p, method)
    return implicit_f(x)[1:size(p)[1]]
end

function RandomFunction(f, p; method = FORM(RIA()))
    return RandomFunction(f, p, method)
end

_vec(x::Real) = [x]
_vec(x) = x

function _jacobian(f, x)
    val, pb = Zygote.pullback(f, x)
    M = length(val)
    vecs = [Vector(sparsevec([i], [true], M)) for i in 1:M]
    Jt = reduce(hcat, first.(pb.(vecs)))
    return copy(Jt')
end

function (f::RandomFunction)(x)
    mup = mean(f.p)
    covp = cov(f.p)
    p0 = getp0(f.f, x, f.p, f.method)
    dfdp0 = _jacobian(p -> f.f(x, p), p0)
    fp0 = f.f(x, p0)
    muf = _vec(fp0) + dfdp0 * (mup - p0)
    covf = dfdp0 * covp * dfdp0'
    return MvNormal(muf, covf)
end

end
