"""
    AbstractTransform

Supertype of parameter-space transformations between the original (physical)
random-parameter space and the standard normal space `u ~ N(0, I)`. All the
reliability methods in this package are formulated in the standard normal
space, where the Hasofer-Lind reliability index is simply the Euclidean norm
of the most probable point.
"""
abstract type AbstractTransform end

"""
    AffineTransform(μ, L, Σ; univariate = false)

Affine mapping `p = μ + L u` with `u ~ N(0, I)` for Gaussian parameters,
where `L` is the lower-triangular Cholesky factor of the covariance matrix
(`Σ = L Lᵀ`). `univariate = true` means the original parameter is a scalar
(a univariate `Normal` distribution); `from_u` then returns a scalar and
`to_u` accepts a scalar.

Constructed automatically from a `Distributions.MvNormal` or
`Distributions.Normal`; users usually do not need to build one directly.
"""
struct AffineTransform{M<:AbstractMatrix} <: AbstractTransform
    μ::Vector{Float64}
    L::M
    Σ::AbstractMatrix{Float64}
    univariate::Bool
end

function AffineTransform(d::MvNormal)
    μ = vec(mean(d))
    Σ = Matrix(cov(d))
    L = Matrix(cholesky(Symmetric(Σ)).L)
    return AffineTransform(μ, L, Σ, false)
end
function AffineTransform(d::Normal)
    μ = [mean(d)]
    L = reshape([std(d)], 1, 1)
    Σ = reshape([var(d)], 1, 1)
    return AffineTransform(μ, L, Σ, true)
end

dim(tr::AffineTransform) = length(tr.μ)
function from_u(tr::AffineTransform, u::AbstractVector)
    return tr.univariate ? tr.μ[1] + tr.L[1, 1] * u[1] : tr.μ + tr.L * u
end
function to_u(tr::AffineTransform, p)
    return tr.univariate ? [(p - tr.μ[1]) / tr.L[1, 1]] : tr.L \ (vec(p) - tr.μ)
end

"""
    StandardNormalTransform(from_u, to_u, [n]; μ, Σ)

A custom, possibly non-affine bijection between the standard normal space
`u ∈ ℝⁿ` and the original parameter space: `p = from_u(u)` and
`u = to_u(p)`. Use it to model non-Gaussian parameters, e.g. via a
Rosenblatt or Nataf transformation implemented by the user. `n` defaults
to `length(μ)`.

`μ` and `Σ` are the mean vector and covariance matrix of the *original*
parameters; they are used only for the first-order linearization of the
output distribution returned by [`RandomFunction`](@ref) and may therefore
be approximate.
"""
struct StandardNormalTransform{F,T,M<:AbstractMatrix} <: AbstractTransform
    from_u::F
    to_u::T
    μ::Vector{Float64}
    Σ::M
    n::Int
end
function StandardNormalTransform(from_u::F, to_u::T; μ, Σ) where {F,T}
    return StandardNormalTransform(from_u, to_u, length(μ); μ=μ, Σ=Σ)
end
function StandardNormalTransform(from_u::F, to_u::T, n::Integer; μ, Σ) where {F,T}
    μv = collect(float.(μ))
    Σm = Matrix(float.(Σ))
    length(μv) == n || throw(ArgumentError("μ must have length $n"))
    size(Σm) == (n, n) || throw(ArgumentError("Σ must be a $n×$n matrix"))
    return StandardNormalTransform(from_u, to_u, μv, Σm, Int(n))
end

dim(tr::StandardNormalTransform) = tr.n
from_u(tr::StandardNormalTransform, u) = tr.from_u(u)
to_u(tr::StandardNormalTransform, p) = tr.to_u(p)

# Standard-normal CDF and inverse CDF in terms of `erf`/`erfinv`, which have
# ChainRules support (via SpecialFunctions), so these are differentiable.
_normcdf(u) = (1 + erf(u / sqrt(2))) / 2
_norminvcdf(p) = sqrt(2) * erfinv(2p - 1)

# Quantile functions with closed, Zygote-differentiable forms where available
# (used by the Rosenblatt and Nataf transformations so that gradients of the
# linearized output distribution work for the textbook distributions).
_quantile_ad(d::Normal, p) = mean(d) + std(d) * _norminvcdf(p)
_quantile_ad(d::LogNormal, p) = exp(d.μ + d.σ * _norminvcdf(p))
_quantile_ad(d::Uniform, p) = d.a + (d.b - d.a) * p
_quantile_ad(d::Exponential, p) = -d.θ * log(1 - p)
_quantile_ad(d::Gumbel, p) = d.μ - d.θ * log(-log(p))
_quantile_ad(d::Weibull, p) = d.θ * (-log(1 - p))^(1 / d.α)
_quantile_ad(d::Frechet, p) = d.θ * (-log(p))^(-1 / d.α)
_quantile_ad(d::UnivariateDistribution, p) = quantile(d, p)

"""
    StandardNormalTransform(d::UnivariateDistribution; μ = [mean(d)], Σ = [var(d);;])

The exact (Rosenblatt) transformation of a univariate distribution `d` to
standard normal space: `p = quantile(d, Φ(u))`, `u = Φ⁻¹(cdf(d, p))`. Works
for any distribution with `cdf` and `quantile` defined (Beta, Gamma, ...).

The classic textbook table (e.g. Nowak & Collins, *Reliability of
Structures*, Table 5.1) is recovered as special cases:

| distribution | transformation |
|---|---|
| `Normal(μ, σ)` | `u = (p − μ) / σ` |
| `LogNormal(λ, ζ)` | `u = (log p − λ) / ζ` |
| `Uniform(a, b)` | `u = Φ⁻¹((p − a) / (b − a))` |
| `Exponential(θ)` | `u = Φ⁻¹(1 − exp(−p / θ))` |
| `Gumbel(μ, θ)` (max) | `u = Φ⁻¹(exp(−exp(−(p − μ) / θ)))` |
| `Weibull(k, λ)` (min) | `u = Φ⁻¹(1 − exp(−(p / λ)^k))` |
| `Frechet(α, σ)` (max) | `u = Φ⁻¹(exp(−(p / σ)^(−α)))` |
| `Gamma(k, θ)` | `u = Φ⁻¹(γ(k, p/θ) / Γ(k))` |

Closed-form, Zygote-differentiable transformations are used for
`LogNormal`, `Uniform`, `Exponential`, `Gumbel`, `Weibull` and `Frechet`;
the generic `cdf`/`quantile` form is used otherwise (reliability indices
and constraints remain differentiable; only the linearized output
distribution of `RandomFunction` requires the transformation itself to be
differentiable).
"""
function StandardNormalTransform(d::UnivariateDistribution; μ=nothing, Σ=nothing)
    μv = μ === nothing ? [float(mean(d))] : collect(float.(μ))
    Σm = Σ === nothing ? reshape([float(var(d))], 1, 1) : Matrix(float.(Σ))
    from_u = u -> _quantile_ad(d, _normcdf(u[1]))
    to_u = p -> [_norminvcdf(cdf(d, p))]
    return StandardNormalTransform(from_u, to_u, 1; μ=μv, Σ=Σm)
end

# Closed-form differentiable special cases of the textbook table.

function StandardNormalTransform(d::LogNormal; μ=nothing, Σ=nothing)
    μv = μ === nothing ? [float(mean(d))] : collect(float.(μ))
    Σm = Σ === nothing ? reshape([float(var(d))], 1, 1) : Matrix(float.(Σ))
    λ, ζ = d.μ, d.σ
    from_u = u -> exp(λ + ζ * u[1])
    to_u = p -> [(log(p) - λ) / ζ]
    return StandardNormalTransform(from_u, to_u, 1; μ=μv, Σ=Σm)
end

function StandardNormalTransform(d::Uniform; μ=nothing, Σ=nothing)
    μv = μ === nothing ? [float(mean(d))] : collect(float.(μ))
    Σm = Σ === nothing ? reshape([float(var(d))], 1, 1) : Matrix(float.(Σ))
    a, b = d.a, d.b
    from_u = u -> a + (b - a) * _normcdf(u[1])
    to_u = p -> [_norminvcdf((p - a) / (b - a))]
    return StandardNormalTransform(from_u, to_u, 1; μ=μv, Σ=Σm)
end

function StandardNormalTransform(d::Exponential; μ=nothing, Σ=nothing)
    μv = μ === nothing ? [float(mean(d))] : collect(float.(μ))
    Σm = Σ === nothing ? reshape([float(var(d))], 1, 1) : Matrix(float.(Σ))
    θ = d.θ
    from_u = u -> -θ * log(1 - _normcdf(u[1]))
    to_u = p -> [_norminvcdf(1 - exp(-p / θ))]
    return StandardNormalTransform(from_u, to_u, 1; μ=μv, Σ=Σm)
end

function StandardNormalTransform(d::Gumbel; μ=nothing, Σ=nothing)
    μv = μ === nothing ? [float(mean(d))] : collect(float.(μ))
    Σm = Σ === nothing ? reshape([float(var(d))], 1, 1) : Matrix(float.(Σ))
    gm, θ = d.μ, d.θ
    from_u = u -> gm - θ * log(-log(_normcdf(u[1])))
    to_u = p -> [_norminvcdf(exp(-exp(-(p - gm) / θ)))]
    return StandardNormalTransform(from_u, to_u, 1; μ=μv, Σ=Σm)
end

function StandardNormalTransform(d::Weibull; μ=nothing, Σ=nothing)
    μv = μ === nothing ? [float(mean(d))] : collect(float.(μ))
    Σm = Σ === nothing ? reshape([float(var(d))], 1, 1) : Matrix(float.(Σ))
    k, λ = d.α, d.θ
    from_u = u -> λ * (-log(1 - _normcdf(u[1])))^(1 / k)
    to_u = p -> [_norminvcdf(1 - exp(-(p / λ)^k))]
    return StandardNormalTransform(from_u, to_u, 1; μ=μv, Σ=Σm)
end

function StandardNormalTransform(d::Frechet; μ=nothing, Σ=nothing)
    μv = μ === nothing ? [float(mean(d))] : collect(float.(μ))
    Σm = Σ === nothing ? reshape([float(var(d))], 1, 1) : Matrix(float.(Σ))
    α, σ = d.α, d.θ
    from_u = u -> σ * (-log(_normcdf(u[1])))^(-1 / α)
    to_u = p -> [_norminvcdf(exp(-(p / σ)^(-α)))]
    return StandardNormalTransform(from_u, to_u, 1; μ=μv, Σ=Σm)
end

# Gauss-Hermite nodes and weights via the Golub-Welsch algorithm
# (probabilists' convention: Σ wₖ f(xₖ) ≈ ∫ f(z) φ(z) dz, Σ wₖ = 1).
function _gauss_hermite(n::Int)
    J = zeros(n, n)
    for i in 1:(n - 1)
        J[i, i + 1] = J[i + 1, i] = sqrt(i)
    end
    E = eigen(Symmetric(J))
    nodes = E.values
    weights = E.vectors[1, :] .^ 2
    return nodes, weights
end

# The marginal quantile of Φ(z): clamp away from 0/1 so extreme Gauss-Hermite
# nodes (±7.6 for n = 20) do not produce ±Inf parameters.
_clampcdf(p) = clamp(p, 1e-15, 1 - 1e-15)

# Nataf equivalent-normal correlation for one pair: the Gauss-Hermite
# approximation of ρ = E[gᵢ(zᵢ) gⱼ(zⱼ)] with z ~ N(0, [1 ρ₀; ρ₀ 1]).
function _nataf_pair_rho(gi, gj, rho0, nodes, weights)
    b = sqrt(1 - rho0^2)
    s = 0.0
    for (k, zk) in enumerate(nodes), (l, zl) in enumerate(nodes)
        s += weights[k] * weights[l] * gi(zk) * gj(rho0 * zk + b * zl)
    end
    return s
end

# Solve the Nataf integral equation ρ = ρ(ρ₀) for ρ₀ by bisection
# (ρ(ρ₀) is monotone increasing for standard marginals).
function _nataf_rho0(gi, gj, rho, nodes, weights)
    f = r0 -> _nataf_pair_rho(gi, gj, r0, nodes, weights)
    lo, hi = -0.9999, 0.9999
    flo, fhi = f(lo), f(hi)
    if rho < flo || rho > fhi
        @warn "the target correlation $rho is outside the achievable range " *
            "[$flo, $fhi] for this pair of marginals; clamping."
        return rho < flo ? lo : hi
    end
    for _ in 1:60
        mid = (lo + hi) / 2
        fm = f(mid)
        fm < rho ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    nataf_transform(marginals, ρ; gh_n = 20)

The Nataf transformation of correlated non-Gaussian parameters to standard
normal space: the marginal Rosenblatt transformations `zᵢ = Φ⁻¹(Fᵢ(pᵢ))`
with the equivalent-normal correlation matrix `ρ₀` solved from the Nataf
integral equation

    ρᵢⱼ = ∫∫ ((Fᵢ⁻¹(Φ(zᵢ)) − μᵢ)/σᵢ) ((Fⱼ⁻¹(Φ(zⱼ)) − μⱼ)/σⱼ) φ₂(zᵢ, zⱼ; ρ₀ᵢⱼ) dzᵢ dzⱼ

so that the back-transformed parameters have the target correlation matrix
`ρ`. `gh_n` is the number of Gauss-Hermite nodes per dimension (20 by
default). The mean `μ` and covariance `Σ = D ρ D` of the joint parameters
(`D = diag(σᵢ)`) are used for the first-order output linearization of
[`RandomFunction`](@ref).
"""
function nataf_transform(
    marginals::AbstractVector{<:UnivariateDistribution}, ρ::AbstractMatrix; gh_n::Integer=20
)
    n = length(marginals)
    size(ρ) == (n, n) || throw(ArgumentError("ρ must be a $n×$n matrix"))
    μs = [float(mean(d)) for d in marginals]
    σs = [float(std(d)) for d in marginals]
    gs = map(1:n) do i
        d = marginals[i]
        return z -> (_quantile_ad(d, _clampcdf(_normcdf(z))) - μs[i]) / σs[i]
    end
    nodes, weights = _gauss_hermite(Int(gh_n))
    ρ0 = Matrix{Float64}(I, n, n)
    for i in 1:n, j in (i + 1):n
        r0 = _nataf_rho0(gs[i], gs[j], float(ρ[i, j]), nodes, weights)
        ρ0[i, j] = ρ0[j, i] = r0
    end
    L = cholesky(Symmetric(ρ0)).L
    from_u =
        u -> begin
            v = L * u
            return [_quantile_ad(marginals[i], _clampcdf(_normcdf(v[i]))) for i in 1:n]
        end
    to_u = p -> begin
        z = [_norminvcdf(cdf(marginals[i], p[i])) for i in 1:n]
        return L \ z
    end
    μ = μs
    Σ = Diagonal(σs) * Matrix(ρ) * Diagonal(σs)
    return StandardNormalTransform(from_u, to_u, n; μ=μ, Σ=Σ)
end

_transform(d::MvNormal) = AffineTransform(d)
_transform(d::Normal) = AffineTransform(d)
_transform(d::UnivariateDistribution) = StandardNormalTransform(d)
_transform(t::AbstractTransform) = t
function _transform(d::Distribution)
    return throw(
        ArgumentError(
            "unsupported random-parameter distribution $(typeof(d)); use " *
            "MvNormal, Normal, a univariate distribution, or a " *
            "StandardNormalTransform (e.g. nataf_transform for correlated " *
            "non-Gaussian parameters).",
        ),
    )
end
