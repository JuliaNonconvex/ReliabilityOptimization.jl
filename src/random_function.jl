# RandomFunction: turns a deterministic function with random parameters into
# a random function with a (first-order) Gaussian output distribution.

"""
    RandomFunction(f, p; method = FORM(RIA()), jacobian = nothing)
    RandomFunction(f, p, method; jacobian = nothing)

Wrap a deterministic function `f(x, p)` of the design variables `x` and the
random parameters `p` into a callable that returns the first-order Gaussian
approximation of the output distribution:

- `p`: the random-parameter distribution (`MvNormal`, `Normal`, or any
  univariate distribution, which is transformed by its Rosenblatt
  transformation) or a [`StandardNormalTransform`](@ref) (e.g.
  [`nataf_transform`](@ref) for correlated non-Gaussian parameters).
- `method`: the reliability method ([`FORM`](@ref) with [`RIA`](@ref) or
  [`PMA`](@ref)); determines the point around which `f` is linearized (the
  MPP for RIA, the minimum-performance point for PMA).
- `jacobian`: optional user function `(f, x, p) -> J` replacing the default
  central finite-difference Jacobian (e.g. a Zygote-based one).

Calling `rf(x)` returns `MvNormal(muf, covf)` with

    muf  = f(x, p0) + ∇ₚf(x, p0) (μ − p0)
    covf = ∇ₚf(x, p0) Σ ∇ₚf(x, p0)ᵀ

where `μ` and `Σ` are the mean and covariance of `p` and `p0` is the
point selected by `method`. The returned object is differentiable w.r.t.
`x` (Zygote/ChainRules).

Note: for PMA methods the linearization point is the performance-measure
point on the `β_target` sphere, not the MPP; use
[`performance_measure`](@ref)/[`reliability_constraint`](@ref) with PMA and
`rf(x)` with RIA.
"""
struct RandomFunction{F,T,M,I,J}
    f::F
    tr::T
    method::M
    implicit::I
    jac::J
end

function RandomFunction(f, p; method=FORM(RIA()), jacobian=nothing)
    tr = _transform(p)
    jac = _resolve_jac(jacobian)
    return RandomFunction(f, tr, method, get_implicit(f, tr, method, jac), jac)
end
function RandomFunction(f, p, method; jacobian=nothing)
    return RandomFunction(f, p; method=method, jacobian=jacobian)
end

# Most probable point in standard normal space.
function _mpp(rf::RandomFunction, x)
    y, _ = rf.implicit(x)
    return y[1:dim(rf.tr)]
end

"""
    getp0(rf::RandomFunction, x)
    getp0(f, x, p, method = FORM(RIA()); jacobian = nothing)

The most probable point (MPP) `p0` in the original parameter space: the
point on the limit-state surface closest to the origin of the standard
normal space (RIA), or the performance-measure point on the `β_target`
sphere (PMA).
"""
getp0(rf::RandomFunction, x) = from_u(rf.tr, _mpp(rf, x))
function getp0(
    f,
    x,
    p::Union{Distribution,AbstractTransform},
    method::FORM=FORM(RIA());
    jacobian=nothing,
)
    return getp0(RandomFunction(f, p, method; jacobian=jacobian), x)
end

function _mean_cov(f::RandomFunction, x)
    p0 = getp0(f, x)
    dfdp0 = f.jac(f.f, x, p0)
    fp0 = f.f(x, p0)
    muf = _vec(fp0) .+ dfdp0 * (f.tr.μ - _vec(p0))
    covf = dfdp0 * f.tr.Σ * dfdp0'
    return muf, covf
end

function (f::RandomFunction)(x)
    muf, covf = _mean_cov(f, x)
    return MvNormal(muf, covf)
end

_unthunk(x) = x
_unthunk(x::ChainRulesCore.AbstractThunk) = ChainRulesCore.unthunk(x)
function _unthunk(x::ChainRulesCore.Tangent)
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
        # Materialize sparse or zero cotangents (`ZeroTangent`, `OneElement`)
        # into ordinary arrays matching the primal shapes: the Zygote pullback
        # of `_mean_cov` mis-handles `ZeroTangent` cotangents through the
        # broadcast in the mean update.
        Δm = _unthunk(Δm)
        ΔS = _unthunk(ΔS)
        Δm = Δm isa ChainRulesCore.AbstractZero ? zeros(length(muf)) : Array(Δm)
        ΔS = ΔS isa ChainRulesCore.AbstractZero ? zeros(size(covf)) : Array(ΔS)
        _, back = Zygote.pullback(_mean_cov, f, x)
        # The cotangent of the last argument (`x`) is the last element of
        # the tuple returned by Zygote's public pullback; depending on the
        # Zygote version the non-differentiable `f` argument cotangent is
        # either kept (2-tuple `(∂f, ∂x)`) or dropped (1-tuple `(∂x,)`), so
        # destructuring into `∂f, ∂x` (BoundsError) or indexing `[1]`
        # (`nothing`) are both wrong.
        ∂x = back((Δm, ΔS))[end]
        return NoTangent(), ∂x
    end
    return dist, pullback
end
