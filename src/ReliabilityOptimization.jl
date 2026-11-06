module ReliabilityOptimization

using ImplicitDifferentiation, Zygote, LinearAlgebra, ChainRulesCore, SparseArrays
using UnPack, NonconvexIpopt, Statistics, Distributions, Reexport
using FiniteDifferences, StaticArraysCore, ADTypes
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
RIA() = RIA(IpoptAlg(), IpoptOptions(print_level = 0, max_wall_time = 30.0, tol = 1e-12))

function get_forward(f, p, method::FORM{<:RIA})
    alg, options = method.method.optim_alg, method.method.optim_options
    prev_minimizer = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    function forward(x)
        # Reliability index analysis (Hasofer-Lind): minimize ||p|| subject to
        # the limit state g(x, p) = 0. The auxiliary variable c is fixed at 0
        # so that the optimum lies on the limit-state surface.
        obj = pc -> begin
            _p = pc[1:end-1]
            return _p' * _p
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
        addvar!(innerOptModel, [fill(-Inf, n); 0.0], [fill(Inf, n); 0.0])
        add_eq_constraint!(innerOptModel, constr)
        # Warm-start from the previous MPP when available.
        x0 = prev_minimizer[] === nothing ? [mean(p); 0.0] : prev_minimizer[]
        result = optimize(innerOptModel, alg, x0, options = options)
        prev_minimizer[] = result.minimizer
        # ImplicitDifferentiation ≥ 0.9 requires the solver to return
        # (solution, byproduct); there is no meaningful byproduct here.
        return vcat(result.minimizer, vec(result.problem.mult_g)), nothing
    end
    return forward
end

function get_conditions(f, ::FORM{<:RIA})
    # ImplicitDifferentiation ≥ 0.9 passes the solver byproduct as the third
    # argument of the conditions function.
    function kkt_conditions(x, pcmult, z)
        p = pcmult[1:end-2]
        c = pcmult[end-1]
        mult = pcmult[end]
        # The auxiliary variable c is fixed at 0 in the forward problem, so its
        # optimality condition is c = 0 rather than 2c - mult = 0.
        return vcat(2 * p + vec(_jacobian(f, x, p)) * mult, c, f(x, p) .- c)
    end
end

function get_implicit(f, p, method)
    forward = get_forward(f, p, method)
    kkt_conditions = get_conditions(f, method)
    # The conditions embed a finite-difference Jacobian (see `_jacobian`),
    # whose derivative with respect to the parameters cannot be obtained by
    # reverse-mode AD. Differentiate the conditions with finite differences
    # as well, so the curvature term mult * d²f/dp² of the KKT Jacobian is
    # included instead of being silently dropped.
    backends = (; x=AutoFiniteDifferences(fdm), y=AutoFiniteDifferences(fdm))
    return ImplicitFunction(forward, kkt_conditions; backends=backends)
end

function getp0(f, x, p, method::FORM{<:RIA})
    implicit_f = get_implicit(f, p, method)
    # ImplicitDifferentiation ≥ 0.9 returns (solution, byproduct).
    y, _ = implicit_f(x)
    return y[1:size(p)[1]]
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
    J = reduce(hcat, ẏs)
    # reduce over a single element returns the element itself, so a scalar
    # output with one random parameter would collapse to a scalar; restore the
    # 1×1 matrix shape that hcat produces for larger inputs.
    return J isa Real ? fill(J, 1, 1) : J
end

# The finite-difference Jacobian `_jacobian(f, x1, x2)` cannot be
# differentiated through by reverse-mode AD (the stencil evaluation points
# depend on `x1` and `x2` in a way Zygote silently drops, which halved the
# gradient). Provide a rrule that computes the second-order terms
# (∂²f/∂x₁∂x₂ and ∂²f/∂x₂²) with central finite differences of the
# Jacobian itself, so the chain remains correct without requiring `f` to
# be AD-friendly.
function ChainRulesCore.rrule(::typeof(_jacobian), f, x1, x2)
    J = _jacobian(f, x1, x2)
    function _jacobian_pullback(ΔJ)
        ΔJ = ChainRulesCore.unthunk(ΔJ)
        Δm = ΔJ isa AbstractMatrix ? Matrix(ΔJ) : fill(ΔJ, size(J))
        h = cbrt(eps(float(real(eltype(x2)))))
        x̄1 = zero(x1)
        x̄2 = zero(x2)
        e = zero(x1)
        for i in eachindex(x1)
            e[i] = one(eltype(x1))
            x̄1[i] = dot(Δm, (_jacobian(f, x1 + h * e, x2) - _jacobian(f, x1 - h * e, x2)) / (2h))
            e[i] = zero(eltype(x1))
        end
        e = zero(x2)
        for i in eachindex(x2)
            e[i] = one(eltype(x2))
            x̄2[i] = dot(Δm, (_jacobian(f, x1, x2 + h * e) - _jacobian(f, x1, x2 - h * e)) / (2h))
            e[i] = zero(eltype(x2))
        end
        return NoTangent(), NoTangent(), x̄1, x̄2
    end
    return J, _jacobian_pullback
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
