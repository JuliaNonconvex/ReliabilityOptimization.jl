# Reliability analysis and probabilistic-constraint API.

"""
    reliability_index(f, x, p; method = FORM(RIA()), jacobian = nothing)
    reliability_index(rf::RandomFunction, x)

The first-order reliability index `β` of the limit state `g(x, p) = 0` at
the design `x`: the distance from the origin to the limit-state surface in
standard normal space. With `method = SORM(...)` the second-order
(Breitung-corrected) index is returned instead. The index does not depend
on which side of the limit state is the failure side.
"""
function reliability_index(f, x, p; method=FORM(RIA()), jacobian=nothing)
    method isa SORM && return _sorm(f, x, _transform(p), method, jacobian)[2]
    rf = RandomFunction(f, p, method; jacobian=jacobian)
    return reliability_index(rf, x)
end
function reliability_index(rf::RandomFunction, x)
    rf.method isa FORM{<:RIA} || throw(
        ArgumentError(
            "reliability_index is defined for RIA-based RandomFunctions; " *
            "use performance_measure for PMA-based ones.",
        ),
    )
    return norm(_mpp(rf, x))
end

"""
    failure_probability(f, x, p; method = FORM(RIA()), jacobian = nothing)

The first-order approximation `Φ(−β)` of the probability of failure
(`SORM`-corrected with `method = SORM(...)`). The index is **signed**:
`β < 0` when the limit state is negative at the mean point `(x, μ)` (the
mean design lies in the failure region), in which case the failure
probability is near 1 and is returned as `Φ(β)`, with a warning. The SORM
correction assumes the mean design is in the safe region.
"""
function failure_probability(f, x, p; method=FORM(RIA()), jacobian=nothing)
    if method isa SORM
        return cdf(Normal(), -reliability_index(f, x, p; method=method, jacobian=jacobian))
    end
    rf = RandomFunction(f, p, method; jacobian=jacobian)
    β = reliability_index(rf, x)
    g0 = _scalar_limit_state(rf.f(x, rf.tr.μ))
    if g0 < 0
        @warn "The limit state is negative at the mean point `(x, μ)`: the " *
            "mean design is in the failure region, so the first-order failure " *
            "probability is near 1 (returned as `Φ(β)` instead of `Φ(−β)`)." maxlog =
            10
        return cdf(Normal(), β)
    end
    return cdf(Normal(), -β)
end

"""
    performance_measure(f, x, p; method = FORM(PMA()), jacobian = nothing)
    performance_measure(rf::RandomFunction, x)

The PMA performance measure: the minimum of the limit state over the
`β_target` sphere in standard normal space. The limit state must be
scalar-valued and expressed with the **margin convention** (positive
values safe, `g = 0` the failure surface); then `g* ≥ 0` if and only if the
reliability index is at least `β_target` (for well-behaved limit states).
"""
function performance_measure(f, x, p; method=FORM(PMA()), jacobian=nothing)
    method isa FORM{<:PMA} || throw(
        ArgumentError("performance_measure requires a PMA method; got $(typeof(method))."),
    )
    rf = RandomFunction(f, p, method; jacobian=jacobian)
    return performance_measure(rf, x)
end
function performance_measure(rf::RandomFunction, x)
    rf.method isa FORM{<:PMA} || throw(
        ArgumentError(
            "performance_measure is defined for PMA-based RandomFunctions; " *
            "use reliability_index for RIA-based ones.",
        ),
    )
    p0 = getp0(rf, x)
    return _scalar_limit_state(rf.f(x, p0))
end

"""
    reliability_constraint(f, p, β_target; method = nothing, jacobian = nothing)

Return a constraint function `x -> c(x)` for use with Nonconvex models,
with `c(x) ≤ 0` if and only if the reliability requirement is satisfied.
The limit state must follow the **margin convention** (positive values
safe, `g = 0` the failure surface); the requirement is then
`P(g(x, p) ≥ 0) ≥ Φ(β_target)` (failure probability at most
`Φ(−β_target)`).

- Default (`method = nothing`): PMA, returning `c(x) = −g*(x)` with the
  margin convention (`g ≥ 0` safe).
- `method = FORM(RIA(...))`: returns `c(x) = β_target − β(x)` with the
  **signed** index convention: `β < 0` when the limit state is negative at
  the mean point `(x, μ)` (the mean design lies in the failure region), so
  the requirement is never reported satisfied there.
- `method = FORM(PMA(β_target = ...))`: as the default, with custom inner
  solver settings.

The returned closure keeps a single reusable inner-optimization workspace
and is differentiable w.r.t. `x` (first-order methods).
"""
function reliability_constraint(f, p, β_target::Real; method=nothing, jacobian=nothing)
    m = method === nothing ? FORM(PMA(; β_target=β_target)) : method
    if m isa FORM{<:PMA}
        rf = RandomFunction(f, p, m; jacobian=jacobian)
        return x -> -performance_measure(rf, x)
    elseif m isa FORM{<:RIA}
        rf = RandomFunction(f, p, m; jacobian=jacobian)
        return x -> begin
            β = reliability_index(rf, x)
            # Signed index: negative when the mean point is in the failure
            # region, so the requirement stays violated there.
            g0 = _scalar_limit_state(rf.f(x, rf.tr.μ))
            return β_target - (g0 < 0 ? -β : β)
        end
    end
    return throw(
        ArgumentError(
            "reliability_constraint requires a FORM(RIA(...)) or " *
            "FORM(PMA(...)) method; got $(typeof(m)).",
        ),
    )
end

"""
    probabilistic_constraint(f, p, β_target; kwargs...)

Alias for [`reliability_constraint`](@ref).
"""
probabilistic_constraint(args...; kwargs...) = reliability_constraint(args...; kwargs...)
