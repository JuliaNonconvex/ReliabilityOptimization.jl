module ReliabilityOptimization

using ImplicitDifferentiation, Zygote, LinearAlgebra, ChainRulesCore, SparseArrays
using UnPack, NonconvexIpopt, Statistics, Distributions, Reexport, DistributionsAD
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

function (f::RandomFunction)(x)
    mup = mean(f.p)
    covp = cov(f.p)
    p0 = getp0(f.f, x, f.p, f.method)
    dfdp0 = _jacobian(f.f, x, p0)
    fp0 = f.f(x, p0)
    muf = _vec(fp0) .+ dfdp0 * (mup - p0)
    covf = dfdp0 * covp * dfdp0'
    return MvNormal(muf, covf)
end

# necessary type piracy FiniteDifferences._estimate_magnitudes uses this constructor which Zygote struggles to differentiate on its own
function ChainRulesCore.rrule(
    ::typeof(StaticArraysCore.SVector{3}),
    x1::T,
    x2::T,
    x3::T,
) where {T}
    StaticArraysCore.SVector{3}(x1, x2, x3), Δ -> (NoTangent(), Δ[1], Δ[2], Δ[3])
end

function ChainRulesCore._eltype_projectto(::Type{T}) where {T<:AbstractVector{<:Number}}
    return ChainRulesCore.ProjectTo(zero(T))
end

end
