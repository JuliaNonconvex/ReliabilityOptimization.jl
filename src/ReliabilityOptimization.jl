"""
    ReliabilityOptimization

Reliability-based design optimization (RBDO/RBTO) on top of the Nonconvex
ecosystem.

A deterministic function `f(x, p)` of the design variables `x` and random
parameters `p` is wrapped into a [`RandomFunction`](@ref), which returns the
first-order Gaussian approximation of the output distribution linearized at
the most probable point. The reliability analysis is performed with:

- [`FORM`](@ref) first-order reliability methods:
  - [`RIA`](@ref) — the reliability index approach (Hasofer-Lind), and
  - [`PMA`](@ref) — the performance measure approach;
- [`SORM`](@ref) — the second-order (Breitung curvature-corrected) method.

The reliability index, failure probability, and performance measure are
available through [`reliability_index`](@ref), [`failure_probability`](@ref)
and [`performance_measure`](@ref), and probabilistic constraints can be
added to Nonconvex models with [`reliability_constraint`](@ref). All
first-order quantities are differentiable with respect to the design
variables.

**Random parameters** are handled in standard normal space (`u ~ N(0, I)`):
Gaussian parameters use the exact affine transformation, univariate
distributions use their Rosenblatt transformation `u = Φ⁻¹(F_X(p))` (the
classic textbook table: Normal, Lognormal, Uniform, Exponential, Gumbel,
Weibull, Frechet, Gamma, ...), and correlated non-Gaussian parameters use
the [`nataf_transform`](@ref) transformation.

**Conventions**: the reliability index is independent of the failure-side
convention, but the PMA performance measure and [`reliability_constraint`](@ref)
use the margin convention (positive limit-state values safe, `g = 0` the
failure surface). Limit states may be scalar- or vector-valued for RIA, but
must be scalar-valued for PMA and SORM.

**Limitations**: the output-distribution approximation is first-order; the
default Jacobians are finite-difference based (supply a `jacobian` function
for better performance); the inner optimizer must support equality
constraints and expose constraint multipliers via `result.problem.mult_g`
(the default [`IpoptAlg`](@ref) does); `SORM` indices are not
differentiable; the inner problem workspace is not thread-safe.
"""
module ReliabilityOptimization

using ImplicitDifferentiation, Zygote, LinearAlgebra, ChainRulesCore, SparseArrays
using UnPack, NonconvexIpopt, Statistics, Distributions, Reexport
using FiniteDifferences, StaticArraysCore, ADTypes, SpecialFunctions
using NonconvexCore: Model, addvar!, add_eq_constraint!, Workspace
import NonconvexCore: tovecmodel, optimize!
@reexport using LinearAlgebra
@reexport using Statistics

export RandomFunction, FORM, RIA, PMA, SORM, MvNormal, Normal
export StandardNormalTransform, nataf_transform
export reliability_index, failure_probability, performance_measure, getp0
export reliability_constraint, probabilistic_constraint

include("transforms.jl")
include("jacobians.jl")
include("methods.jl")
include("sorm.jl")
include("random_function.jl")
include("constraints.jl")

end
