# NEWS

## 0.6.0

- NonconvexCore 2 is now required (`NonconvexCore = "2"`); NonconvexUtils
  is no longer a dependency.
- Reliability methods are now formulated in standard normal space
  (`u ~ N(0, I)`), which changes numerical results for non-standard-normal
  parameters (previously incorrect). The `RandomFunction` constructor
  signature is unchanged, but the internal fields changed.
- `probabilistic_constraint` is now a function rather than a `const` alias.
- `ChainRulesCore.ProjectTo(::SVector{...})` extension removed (method
  piracy); `StaticArraysCore` is no longer a dependency.
