# Finite-difference machinery for limit-state Jacobians and Hessians.

_vec(x::Real) = [x]
_vec(x) = x

const fdm = FiniteDifferences.central_fdm(5, 1)

"""
    _jacobian(f, x1, x2)

Jacobian `∂f(x1, x2)/∂x2` computed with central finite differences on the
second argument. Always returns a matrix (even for scalar `f` outputs and
scalar `x2` inputs).
"""
function _jacobian(f, x1, x2)
    x2v = _vec(x2)
    scalarx2 = x2 isa Real
    ẏs = map(eachindex(x2v)) do n
        return fdm(zero(eltype(x2v))) do ε
            xcopy = vcat(x2v[1:n-1], x2v[n] + ε, x2v[n+1:end])
            # copy required incase `f(x)` returns something that aliases `x`
            return copy(f(x1, scalarx2 ? xcopy[1] : xcopy))
        end
    end
    J = reduce(hcat, ẏs)
    # reduce over a single element returns the element itself, so a scalar
    # output with one random parameter would collapse to a scalar; restore
    # the matrix shape that hcat produces for larger inputs.
    J isa Real && return fill(J, 1, 1)
    J isa AbstractVector && return reshape(J, length(J), 1)
    return J
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
        x̄1 = zeros(float(eltype(x1)), length(x1))
        e = zeros(eltype(x1), length(x1))
        for i in eachindex(x1)
            e[i] = one(eltype(x1))
            x̄1[i] = dot(Δm, (_jacobian(f, x1 + h * e, x2) - _jacobian(f, x1 - h * e, x2)) / (2h))
            e[i] = zero(eltype(x1))
        end
        # scalar parameters are perturbed directly; vector parameters per
        # component (the limit state may only accept a scalar `p`)
        x2v = _vec(x2)
        x̄2v = zeros(float(eltype(x2v)), length(x2v))
        if x2 isa Real
            x̄2v[1] = dot(Δm, (_jacobian(f, x1, x2 + h) - _jacobian(f, x1, x2 - h)) / (2h))
        else
            e = zeros(eltype(x2), length(x2v))
            for i in eachindex(x2v)
                e[i] = one(eltype(x2))
                x̄2v[i] = dot(
                    Δm,
                    (_jacobian(f, x1, x2 + h * e) - _jacobian(f, x1, x2 - h * e)) / (2h),
                )
                e[i] = zero(eltype(x2))
            end
        end
        x̄2 = x2 isa Real ? x̄2v[1] : x̄2v
        return NoTangent(), NoTangent(), x̄1, x̄2
    end
    return J, _jacobian_pullback
end

"""
    _hessian_fd(g, p)

Central finite-difference Hessian of a scalar-valued function `g(p)`.
"""
function _hessian_fd(g, p::Real)
    h = eps(Float64)^(1 / 4)
    g0 = g(p)
    return reshape([(g(p + h) - 2g0 + g(p - h)) / h^2], 1, 1)
end
function _hessian_fd(g, p::AbstractVector)
    n = length(p)
    H = zeros(float(eltype(p)), n, n)
    h = eps(float(eltype(p)))^(1 / 4)
    g0 = g(p)
    for i in 1:n
        ei = zeros(eltype(p), n)
        ei[i] = h
        H[i, i] = (g(p + ei) - 2g0 + g(p - ei)) / h^2
        for j in (i + 1):n
            ej = zeros(eltype(p), n)
            ej[j] = h
            H[i, j] =
                (g(p + ei + ej) - g(p + ei - ej) - g(p - ei + ej) + g(p - ei - ej)) /
                (4h^2)
            H[j, i] = H[i, j]
        end
    end
    return H
end

# Hessian of the (scalar) limit state w.r.t. the parameters.
_hessian(f, x, p) = _hessian_fd(q -> _scalar_limit_state(f(x, q)), p)

# User-supplied Jacobian or the finite-difference default; the choice is
# resolved here so that no runtime branch on `jac === nothing` is left in
# Zygote-traced code (Zygote cannot differentiate through such ternaries).
_resolve_jac(jac) = jac === nothing ? _jacobian : jac

function _scalar_limit_state(g)
    gv = _vec(g)
    length(gv) == 1 || throw(
        ArgumentError(
            "this method requires a scalar-valued limit-state function; got " *
            "$(length(gv)) outputs.",
        ),
    )
    return gv[1]
end
