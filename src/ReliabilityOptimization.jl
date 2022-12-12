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
struct RIA{A, O}
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
        result = optimize(
            innerOptModel,
            alg,
            [mean(p); 0.0],
            options = options,
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
            2 * p + vec(_jacobian(f, x, p)) * mult,
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

function get_identity_vecs(M)
    return [Vector(sparsevec([i], [1.0], M)) for i in 1:M]
end
function ChainRulesCore.rrule(::typeof(get_identity_vecs), M::Int)
    get_identity_vecs(M), _ -> (NoTangent(), NoTangent())
end
reduce_hcat(vs) = reduce(hcat, vs)
# function ChainRulesCore.rrule(::typeof(reduce_hcat), vs::Vector{<:Vector})
#     return reduce_hcat(vs), Δ -> begin
#         return NoTangent(), [Δ[:, i] for i in 1:size(Δ, 2)]
#     end
# end

const fdm = FiniteDifferences.central_fdm(5, 1)

function _jacobian(f, x1, x2)
    # val, pb = Zygote.pullback(f, x1, x2)
    # if val isa Vector
    #     M = length(val)
    #     vecs = get_identity_vecs(M)
    #     cotangents = map(pb, vecs)
    #     Jt = reduce_hcat(map(last, cotangents))
    #     return copy(Jt')
    # elseif val isa Real
    #     Jt = last(pb(1.0))
    #     return copy(Jt')
    # else
    #     throw(ArgumentError("Output type not supported."))
    # end
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
# function ChainRulesCore.rrule(::typeof(_jacobian), f, x1, x2)
#     (val, pb), _pb_pb = Zygote.pullback(Zygote.pullback, f, x1, x2)
#     M = length(val)
#     if val isa Vector
#         vecs = get_identity_vecs(M)
#         _pb = (pb, v) -> last(pb(v))
#         co1, pb_pb = Zygote.pullback(_pb, pb, first(vecs))
#         cotangents = vcat([co1], last.(map(pb, @view(vecs[2:end]))))
#         Jt, hcat_pb = Zygote.pullback(reduce_hcat, cotangents)
#         return copy(Jt'), Δ -> begin
#             temp = hcat_pb(Δ')[1]
#             co_pb = map(temp) do t
#                 first(pb_pb(t))
#             end
#             co_f_x = _pb_pb.(tuple.(Ref(nothing), co_pb))
#             co_f = sum(getindex.(co_f_x, 1))
#             co_x1 = sum(getindex.(co_f_x, 2))
#             co_x2 = sum(getindex.(co_f_x, 3))
#             return NoTangent(), co_f, co_x1, co_x2
#         end
#     elseif val isa Real
#         println(1)
#         _pb = (pb, v) -> pb(v)[end]
#         println(2)
#         @show _pb(pb, 1.0)
#         Jt, pb_pb = Zygote.pullback(_pb, pb, 1.0)
#         println(3)
#         return copy(Jt'), Δ -> begin
#             println(4)
#             @show vec(Δ)
#             @show Δ
#             @show pb_pb(Δ')
#             co_pb = first(pb_pb(vec(Δ)))
#             co_f_x = _pb_pb((nothing, co_pb))
#             co_f = co_f_x[1]
#             co_x1 = co_f_x[2]
#             co_x2 = co_f_x[3]
#             return NoTangent(), co_f, co_x1, co_x2
#         end
#     else
#         throw(ArgumentError("Output type not supported."))
#     end
# end

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
function ChainRulesCore.rrule(::typeof(StaticArraysCore.SVector{3}), x1::T, x2::T, x3::T) where {T}
    StaticArraysCore.SVector{3}(x1, x2, x3), Δ -> (NoTangent(), Δ[1], Δ[2], Δ[3])
end

end