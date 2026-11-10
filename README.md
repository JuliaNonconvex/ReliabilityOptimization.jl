# ReliabilityOptimization

<img src="https://raw.githubusercontent.com/JuliaNonconvex/ReliabilityOptimization.jl/master/docs/src/assets/logo.png" width="120" align="right"/>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/ReliabilityOptimization.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/ReliabilityOptimization.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/ReliabilityOptimization.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/ReliabilityOptimization.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/ReliabilityOptimization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/ReliabilityOptimization.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)


Reliability-based design optimization (RBDO/RBTO) on top of the
[Nonconvex](https://github.com/JuliaNonconvex/Nonconvex.jl) ecosystem.

A deterministic function `f(x, p)` of the design variables `x` and random
parameters `p` is wrapped into a `RandomFunction`, which returns the
first-order Gaussian approximation of the output distribution, linearized at
the most probable point (MPP). The reliability analysis is performed with

- **FORM** first-order reliability methods:
  - `RIA` — the reliability index approach (Hasofer-Lind), and
  - `PMA` — the performance measure approach;
- `SORM` — the second-order (Breitung curvature-corrected) method.

All first-order quantities (`RandomFunction`, the reliability index, the
performance measure, and `reliability_constraint`) are differentiable with
respect to the design variables.

## Quick start

```julia
using ReliabilityOptimization, Distributions, Zygote

# deterministic limit state with random material property p
g = (x, p) -> x[1] + 2x[2] - p       # margin convention: g ≥ 0 is safe
p = LogNormal(0.0, 0.5)

# first-order reliability index and failure probability at the design x
β = reliability_index(g, [2.0, 1.0], p)          # β = distance to g = 0 in u-space
pf = failure_probability(g, [2.0, 1.0], p)       # ≈ Φ(−β)

# the most probable point (in original parameter space)
p0 = getp0(g, [2.0, 1.0], p)

# gradients with respect to the design
dβdx = Zygote.gradient(x -> reliability_index(g, x, p), [2.0, 1.0])[1]

# probabilistic constraint for use in a Nonconvex model:
# c(x) ≤ 0  ⟺  P(g(x, p) ≤ 0) ≥ Φ(β_target)
c = reliability_constraint(g, p, 3.0)            # PMA by default
c = reliability_constraint(g, p, 3.0; method = FORM(RIA()))   # RIA version
```

## Random functions

```julia
# wrap a deterministic function so it returns the linearized output distribution
f = (x, p) -> [p[1] * x[1], p[2] - x[2]]
p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))
rf = RandomFunction(f, p)                        # FORM(RIA()) by default

d = rf([1.0, 2.0])                               # MvNormal(muf, covf)
mean(d), cov(d)

# robust objective: mean + k·std, differentiable w.r.t. x
obj = x -> mean(rf(x))[1] + 2 * sqrt(cov(rf(x))[1, 1])
Zygote.gradient(obj, [1.0, 2.0])
```

Limit states may be scalar- or vector-valued for `RIA`; `PMA` and `SORM`
require scalar-valued limit states expressed with the margin convention
(positive = safe).

## Distribution transformations

Random parameters are handled in standard normal space (`u ~ N(0, I)`).
Gaussian parameters use the exact affine transformation; any univariate
distribution is transformed exactly by its Rosenblatt transformation
`u = Φ⁻¹(F_X(p))`, covering the classic textbook table
(cf. Nowak & Collins, *Reliability of Structures*, Table 5.1):

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

Closed-form differentiable transforms are used for `LogNormal`, `Uniform`,
`Exponential`, `Gumbel`, `Weibull` and `Frechet`; other distributions (e.g.
`Gamma`, `Beta`) use the generic `cdf`/`quantile` form.

```julia
p = Exponential(1.3)                             # any univariate distribution
β = reliability_index(g, x, p)
```

Correlated non-Gaussian parameters are supported via the Nataf
transformation, which solves the equivalent-normal correlation matrix
`ρ₀` so that the back-transformed parameters have the target correlation:

```julia
marginals = [LogNormal(0.0, 0.4), Gumbel(1.0, 0.3), Weibull(2.0, 1.2)]
ρ = [1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0]
tr = nataf_transform(marginals, ρ)

β = reliability_index(g, x, tr)
rf = RandomFunction(f, tr)                       # f(x, p) with p a vector
```

Custom transformations (e.g. a user-implemented Rosenblatt or Nataf
mapping) can be supplied directly:

```julia
tr = StandardNormalTransform(
    u -> exp.(u), p -> log.(p);                 # p = from_u(u), u = to_u(p)
    μ = [exp(0.5)], Σ = [exp(1) * (exp(1) - 1);;],
)
```

## Solvers

The inner reliability problems (MPP search) are solved with a Nonconvex
optimizer; the default is `IpoptAlg`. Any optimizer that supports equality
constraints and reports multipliers via `result.problem.mult_g` can be
substituted, and deterministic multistarts are available:

```julia
reliability_index(g, x, p; method = FORM(RIA(nstarts = 4)))
performance_measure(g, x, p; method = FORM(PMA(β_target = 3.0)))
```

The inner `VecModel` and solver `Workspace` are built once per
`RandomFunction` and reused across evaluations, warm-starting each solve
from the previous MPP.

The default limit-state Jacobians are central finite differences; a custom
Jacobian (e.g. Zygote-based) can be supplied for better performance:

```julia
rf = RandomFunction(f, p; jacobian = (f, x, p) -> Zygote.jacobian(q -> f(x, q), p)[1])
```

## Conventions and limitations

- The reliability index `β` is independent of which side of the limit state
  fails; the PMA performance measure and `reliability_constraint` use the
  margin convention (positive values safe, `g = 0` the failure surface).
- The output-distribution approximation is first-order (linearization at the
  MPP); `SORM` corrects the failure probability for limit-state curvature but
  is not differentiable.
- `RandomFunction` output-distribution gradients require the parameter
  transformation to be Zygote-differentiable (true for the built-in
  closed-form transforms; reliability indices and constraints are
  differentiable for any distribution).
- The inner problem workspace is not thread-safe.
