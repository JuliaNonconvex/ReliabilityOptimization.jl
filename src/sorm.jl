# Second-order reliability method (SORM).

"""
    SORM(method = FORM(RIA()); hessian = nothing)

Second-order reliability method: Breitung's curvature correction applied to
the MPP of the inner first-order method `method` (an RIA-based
[`FORM`](@ref)). The limit state must be scalar-valued.

- `hessian`: optional user function `(f, x, p) -> H` returning the Hessian
  of the limit state w.r.t. the parameters (or, for custom
  [`StandardNormalTransform`](@ref)s, `(f, x, u) -> H` in u-space). Defaults
  to central finite differences.

SORM indices are **evaluation-only**: differentiating them (e.g. with
Zygote) is not supported; use [`FORM`](@ref) methods inside optimization
objectives or constraints.
"""
struct SORM{M,H}
    method::M
    hessian::H
end
SORM(method::M) where {M} = SORM(method, nothing)
SORM(; method=FORM(RIA()), hessian=nothing) = SORM(method, hessian)

# SORM is a reliability-analysis method, not a RandomFunction method.
function get_forward(f, tr, method::SORM, jac)
    return throw(
        ArgumentError(
            "SORM is a reliability-analysis method and cannot be used as the " *
            "RandomFunction method; use FORM(RIA()) or FORM(PMA()), or pass " *
            "SORM to reliability_index/failure_probability.",
        ),
    )
end

# Gradient of the scalar limit state in u-space.
function _grad_u(f, x, tr::AffineTransform, u, jac)
    p = from_u(tr, u)
    J = jac(f, x, p)
    return tr.L' * vec(J)
end
function _grad_u(f, x, tr::StandardNormalTransform, u, jac)
    return FiniteDifferences.grad(fdm, q -> _scalar_limit_state(f(x, tr.from_u(q))), u)[1]
end

# Hessian of the scalar limit state in u-space.
function _hessian_u(f, x, tr::AffineTransform, u, hess)
    p = from_u(tr, u)
    Hp = hess === nothing ? _hessian(f, x, p) : hess(f, x, p)
    return tr.L' * Hp * tr.L
end
function _hessian_u(f, x, tr::StandardNormalTransform, u, hess)
    hess === nothing || return hess(f, x, u)
    return _hessian_fd(q -> _scalar_limit_state(f(x, tr.from_u(q))), u)
end

# Breitung's asymptotic formula: P_f ≈ Φ(−β) ∏ᵢ (1 + β κᵢ)^(−1/2), where the
# κᵢ are the principal curvatures of the limit-state surface at the MPP,
# positive when the surface curves away from the origin.
function _sorm(f, x, tr, method::SORM, jac)
    inner = method.method
    jac = _resolve_jac(jac)
    rf = RandomFunction(f, tr, inner, get_implicit(f, tr, inner, jac), jac)
    u0 = _mpp(rf, x)
    β = norm(u0)
    (β > 0) || return 0.0, 0.0
    n = dim(tr)
    ∇u = _grad_u(f, x, tr, u0, jac)
    gmag = norm(∇u)
    (gmag > 0) || return β, β
    Hu = _hessian_u(f, x, tr, u0, method.hessian)
    α = u0 ./ β
    P = Matrix{Float64}(I, n, n) - α * α'
    Q = Symmetric(-P * (Hu ./ gmag) * P)
    κ = eigvals(Q)
    terms = 1 .+ β .* κ
    if any(t -> t ≤ 0, terms)
        @warn "SORM curvature correction is undefined (1 + βκ ≤ 0); " *
            "returning the FORM reliability index."
        return β, β
    end
    pf = cdf(Normal(), -β) * prod(t -> 1 / sqrt(t), terms)
    βs = -quantile(Normal(), pf)
    return β, βs
end
