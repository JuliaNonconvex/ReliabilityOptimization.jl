# NEWS

## 0.5.6

- Docs: package-name section titles and left-aligned logos (no code changes).
## 0.5.5

- `ImplicitDifferentiation` compat tightened to `0.9` (the code requires the ≥ 0.9 API).\n- The TopOpt example gradients are now asserted.
## 0.5.4

- Signed the RIA reliability index: `reliability_constraint` (RIA) and
  `failure_probability` now treat designs whose mean point lies in the
  failure region as unreliable (failure probability near 1, with a warning)
  instead of reporting the requirement satisfied via the unsigned index.

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
