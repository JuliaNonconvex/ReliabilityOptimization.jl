# NEWS

## 0.5.3

- Fixed the RandomFunction rrule BoundsError and the scalar-design Jacobian
  pullback; widened the dependency compatibility bounds; added Pkg to the
  test environment; codecov OIDC/v5 uploads.

## 0.5.2

- Re-enabled the TopOpt example testitem now that the Core-2 upstream is
  registered; restored the full README with the docs-first badges; logo
  sizing fixes.

## 0.5.1

- Widened the Zygote compat for the Core-2 ecosystem; added the
  organization logo to the docs and README; fixed the docs deployment
  (destinations persisted to gh-pages).

## 0.5.0

- NonconvexCore 2 is now required (`NonconvexCore = "2"`); NonconvexUtils
  is no longer a dependency.
- Reliability methods are now formulated in standard normal space
  (`u ~ N(0, I)`), which changes numerical results for non-standard-normal
  parameters (previously incorrect). The `RandomFunction` constructor
  signature is unchanged, but the internal fields changed.
- `probabilistic_constraint` is now a function rather than a `const` alias.
- `ChainRulesCore.ProjectTo(::SVector{...})` extension removed (method
  piracy); `StaticArraysCore` is no longer a dependency.
