# Changelog

## 0.6.0 (unreleased)

- Support NonconvexCore 2: `NonconvexCore = "2"` is now the compat lower
  bound (NonconvexUtils is no longer used), and
  `ImplicitDifferentiation.ImplicitFunction` is imported explicitly
  (NonconvexCore 2 exports its own `ImplicitFunction`).
- Complete the reliability-analysis functionality, formulated in standard
  normal space:
  - `RIA` rewritten with the Hasofer-Lind problem in u-space, vector-valued
    limit states, deterministic multistarts, and a reusable inner
    `VecModel`/`Workspace` with MPP warm starts.
  - `PMA` (performance measure approach) with the margin convention.
  - `SORM` with Breitung's curvature correction.
- Distribution transformations: the classic textbook Rosenblatt table
  (Normal, Lognormal, Uniform, Exponential, Gumbel, Weibull, Frechet,
  Gamma, ...) with closed-form differentiable special cases, a generic
  `cdf`/`quantile` fallback for any univariate distribution, and
  `nataf_transform` for correlated non-Gaussian parameters.
- API: `reliability_index`, `failure_probability`, `performance_measure`,
  `getp0`, `reliability_constraint` (alias `probabilistic_constraint`),
  and a pluggable `jacobian` argument.
- Test suite reorganized with TestItems and a separate test environment;
  Aqua and JET quality checks; JuliaFormatter checks; codecov PR comments.
- Docs fragment (DocumenterFragments) with an introduction, a theory
  review, per-method pages, the transformation table, and per-PR preview
  deployment.

## 0.4.1

- NonconvexCore 1 support
