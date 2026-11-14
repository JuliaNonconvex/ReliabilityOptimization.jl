# ReliabilityOptimization

```@raw html
<p align="center">
  <img src="https://raw.githubusercontent.com/JuliaNonconvex/ReliabilityOptimization.jl/master/docs/src/assets/logo.png" alt="JuliaNonconvex logo" width="180"/>
</p>
```



`ReliabilityOptimization` provides reliability-based design optimization
(RBDO/RBTO) on top of the [Nonconvex](https://github.com/JuliaNonconvex/Nonconvex.jl)
ecosystem. A deterministic function `f(x, p)` of the design variables `x`
and the random parameters `p` is wrapped into a
[`RandomFunction`](@ref), which returns the first-order Gaussian
approximation of the output distribution, linearized at the most probable
point (MPP). The reliability analysis is performed with

- [`FORM`](@ref) first-order reliability methods:
  - [`RIA`](@ref) — the reliability index approach (Hasofer-Lind), and
  - [`PMA`](@ref) — the performance measure approach;
- [`SORM`](@ref) — the second-order (Breitung curvature-corrected) method.

All first-order quantities — the `RandomFunction` output distribution, the
reliability index, the performance measure and the probabilistic
constraints — are differentiable with respect to the design variables.

This documentation is a fragment of the [Nonconvex documentation](https://julianonconvex.github.io/Nonconvex.jl/)
and is also built standalone.

## Quick start

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> # a limit state with a random property p (margin convention: g ≥ 0 is safe)

julia> g = (x, p) -> x[1] + 2x[2] - p[1];

julia> p = MvNormal(zeros(1), ones(1, 1));

julia> β = reliability_index(g, [2.0, 1.0], p); # = 4

julia> pf = failure_probability(g, [2.0, 1.0], p); # = Φ(−4)

julia> p0 = getp0(g, [2.0, 1.0], p); # the MPP

julia> using Zygote

julia> Zygote.gradient(x -> reliability_index(g, x, p), [2.0, 1.0])[1];
```

## Theory

The methods are reviewed in [Theory](theory.md): the limit-state
formulation, standard normal space, the Hasofer-Lind reliability index,
FORM (RIA and PMA) with its linearization and sensitivity analysis, and
SORM's curvature correction, with references.

## Reliability methods

[`FORM`](@ref) wraps a first-order method: [`RIA`](@ref) solves the
Hasofer-Lind problem for the reliability index, [`PMA`](@ref) solves the
inverse performance-measure problem, and [`SORM`](@ref) corrects the
failure probability for limit-state curvature. See
[Reliability methods](methods.md).

## Random functions

[`RandomFunction`](@ref) turns a deterministic function with random
parameters into a callable returning the linearized Gaussian output
distribution, differentiable with respect to the design. See
[Random functions](random_function.md).

## Distribution transformations

All reliability computations happen in standard normal space. Gaussian
parameters use the exact affine transformation, univariate distributions
use their Rosenblatt transformation (the classic textbook table: Normal,
Lognormal, Uniform, Exponential, Gumbel, Weibull, Frechet, Gamma, ...), and
correlated non-Gaussian parameters use the [`nataf_transform`](@ref)
transformation. See
[Distribution transformations](transforms.md).

## Probabilistic constraints

[`reliability_constraint`](@ref) builds constraints of the form
`P(g(x, p) ≥ 0) ≥ Φ(β_target)` (margin convention: positive values safe)
for Nonconvex models, in either the RIA or the PMA formulation. See
[Reliability constraints](constraints.md).

## Conventions and limitations

- The reliability index `β` is independent of which side of the limit
  state fails; the PMA performance measure and
  [`reliability_constraint`](@ref) use the margin convention (positive
  values safe, `g = 0` the failure surface).
- Limit states may be scalar- or vector-valued for [`RIA`](@ref), but must
  be scalar-valued for [`PMA`](@ref) and [`SORM`](@ref).
- The output-distribution approximation is first-order. The default
  limit-state Jacobians are central finite differences; supply a
  `jacobian` function for better performance.
- The inner reliability problems are solved with a Nonconvex optimizer
  (default: `IpoptAlg`); the inner `VecModel` and solver `Workspace` are
  built once per `RandomFunction` and reused with warm starts.
- `SORM` indices are not differentiable, and the inner problem workspace
  is not thread-safe.
