module ReliabilityOptimization

using ImplicitDifferentiation, Zygote, LinearAlgebra, ChainRulesCore, SparseArrays
using UnPack, NonconvexIpopt, Statistics, Distributions, Reexport
using FiniteDifferences, StaticArraysCore
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
struct RIA{A,O}
    optim_alg::A
    optim_options::O
end
RIA() = RIA(IpoptAlg(), IpoptOptions(print_level = 0, max_wall_time = 1.0))

function get_forward(f, p, method::FORM{<:RIA})
    alg, options = method.method.optim_alg, method.method.optim_options
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
        result = optimize(innerOptModel, alg, [mean(p); 0.0], options = options)
        return vcat(result.minimizer, result.problem.mult_g[1])
    end
    return forward
end

function get_conditions(f, ::FORM{<:RIA})
    function kkt_conditions(x, pcmult)
        p = pcmult[1:end-2]
        c = pcmult[end-1]
        mult = pcmult[end]
        return vcat(2 * p + vec(_jacobian(f, x, p)) * mult, 2c - mult, f(x, p) .- c)
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

const fdm = FiniteDifferences.central_fdm(5, 1)

function _jacobian(f, x1, x2)
    ẏs = map(eachindex(x2)) do n
        return fdm(zero(eltype(x2))) do ε
            xn = x2[n]
            xcopy = vcat(x2[1:n-1], xn + ε, x2[n+1:end])
            ret = copy(f(x1, xcopy))  # copy required incase `f(x)` returns something that aliases `x`
            return ret
        end
    end
    return reduce(hcat, ẏs)
end

function _mean_cov(f::RandomFunction, x)
    mup = mean(f.p)
    covp = cov(f.p)
    p0 = getp0(f.f, x, f.p, f.method)
    dfdp0 = _jacobian(f.f, x, p0)
    fp0 = f.f(x, p0)
    muf = _vec(fp0) .+ dfdp0 * (mup - p0)
    covf = dfdp0 * covp * dfdp0'
    return muf, covf
end

function (f::RandomFunction)(x)
    muf, covf = _mean_cov(f, x)
    return MvNormal(muf, covf)
end

_unthunk(x) = x
_unthunk(x::ChainRulesCore.AbstractThunk) = ChainRulesCore.unthunk(x)
_unthunk(x::ChainRulesCore.Tangent) = begin
    P = typeof(x).parameters[1]
    return ChainRulesCore.Tangent{P}(; map(_unthunk, ChainRulesCore.backing(x))...)
end
_unthunk(x::NamedTuple) = map(_unthunk, x)
_unthunk(x::Tuple) = map(_unthunk, x)

# Zygote 0.7 cannot differentiate through the `MvNormal`/`PDMat` struct
# constructors (`Jnew` rejects thunked cotangents), so provide a rrule that
# only differentiates the mean/covariance computation.
function ChainRulesCore.rrule(f::RandomFunction, x)
    muf, covf = _mean_cov(f, x)
    dist = MvNormal(muf, covf)
    function pullback(Δ)
        Δu = _unthunk(Δ)
        backing = ChainRulesCore.backing(Δu)
        Δm = get(backing, :μ, ChainRulesCore.ZeroTangent())
        ΔS = get(backing, :Σ, ChainRulesCore.ZeroTangent())
        _, back = Zygote.pullback(_mean_cov, f, x)
        ∂f, ∂x = back((_unthunk(Δm), _unthunk(ΔS)))
        return NoTangent(), ∂x
    end
    return dist, pullback
end

function ChainRulesCore.ProjectTo(x::StaticArraysCore.SVector{N, Vector{Float64}}) where {N}
    return ChainRulesCore.ProjectTo{SArray}(;
        element = ChainRulesCore.ProjectTo(zero(first(x))),
        axes = axes(x),
        size = Size(x),
    )
end

end
