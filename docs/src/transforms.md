# Distribution transformations

Random parameters are handled in the standard normal space
`u ~ N(0, I)`; the reliability index is then the Euclidean norm of the most
probable point. The mapping between the original parameters and `u` is
selected automatically from the distribution:

- `MvNormal` and `Normal` use the exact affine transformation
  `p = μ + L u` ([`AffineTransform`](@ref ReliabilityOptimization.AffineTransform)),
- any univariate distribution uses its exact Rosenblatt transformation
  `p = F_X^{-1}(Φ(u))` (built as a [`StandardNormalTransform`](@ref)),
- correlated non-Gaussian parameters use [`nataf_transform`](@ref),
- custom mappings can be supplied as a [`StandardNormalTransform`](@ref)
  directly.

## Gaussian parameters

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> # non-standard Gaussians are handled exactly in u-space: u = (p − μ)/σ

julia> g = (x, p) -> p[1] - x[1];

julia> p = MvNormal([1.0], [4.0;;]);

julia> reliability_index(g, [2.0], p); # (2 − 1)/2 = 0.5

julia> # univariate Normal: the limit state receives a scalar p

julia> gu = (x, p) -> p - 2.0 - x[1];

julia> pu = Normal(0.0, 1.0);

julia> reliability_index(gu, [0.0], pu); # 2.0
```

## Univariate distributions

The classic textbook transformation table (cf. Nowak & Collins,
*Reliability of Structures*, Table 5.1) is recovered as special cases:

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
other distributions use the generic `cdf`/`quantile` form. Reliability
indices and constraints remain differentiable for any distribution; only
the linearized output distribution of [`RandomFunction`](@ref) requires the
transformation itself to be differentiable.

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> g = (x, p) -> p - 3.0 - x[1];

julia> # lognormal property: β = (log(3 + x₁) − λ)/ζ

julia> pln = LogNormal(0.0, 1.0);

julia> reliability_index(g, [0.0], pln); # log(3)

julia> # any univariate distribution with cdf/quantile works (e.g. Gamma, Beta)

julia> pg = Gamma(2.0, 1.5);

julia> reliability_index(g, [0.0], pg);
```

## Correlated non-Gaussian parameters

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> marginals = [LogNormal(0.0, 0.4), Gumbel(1.0, 0.3), Weibull(2.0, 1.2)];

julia> ρ = [1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0];

julia> tr = nataf_transform(marginals, ρ);

julia> g = (x, p) -> p[1] + p[2] + p[3] - x[1];

julia> reliability_index(g, [4.0], tr);

julia> rf = RandomFunction(g, tr); # f(x, p) with p a vector
```

For `Normal` marginals the Nataf transformation reduces to the affine
transformation with the Cholesky factor of the correlation matrix.

## Custom transformations

A user-implemented Rosenblatt or Nataf mapping can be supplied directly;
`μ` and `Σ` are used only for the first-order output linearization of
[`RandomFunction`](@ref) and may be approximate:

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> # lognormal-like parameters: p = exp(u) with u standard normal

julia> tr = StandardNormalTransform(
           u -> exp.(u), p -> log.(p);                  # p = from_u(u), u = to_u(p)
           μ = [exp(0.5)], Σ = [exp(1) * (exp(1) - 1);;],
       );

julia> g = (x, p) -> p[1] - 3.0;

julia> reliability_index(g, [0.0], tr); # log(3)
```
