# Reliability methods

All reliability methods are formulated in the standard normal space
`u ~ N(0, I)`, where the Hasofer-Lind reliability index is the Euclidean
norm of the most probable point. The mapping between the original random
parameters and `u` is handled by the [distribution
transformations](transforms.md).

## First-order reliability method

[`FORM`](@ref) wraps the method that defines the inner reliability
problem:

## Reliability index approach

[`RIA`](@ref) (Hasofer-Lind) minimizes `‖u‖²` subject to the limit state
`g(x, p(u)) = 0`. The reliability index is `β = ‖u*‖` and `p* = p(u*)` is
the most probable point. Limit states may be scalar- or vector-valued.

The inner problem is solved with a Nonconvex optimizer (default
`IpoptAlg`); any optimizer that supports equality constraints and reports
multipliers via `result.problem.mult_g` can be substituted. The inner
`VecModel` and solver `Workspace` are built once and reused across
evaluations, warm-starting each solve from the previous MPP. `nstarts`
adds deterministic multistart points:

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> g = (x, p) -> p[1] - 2.0 - x[1];

julia> p = MvNormal(zeros(1), ones(1, 1));

julia> reliability_index(g, [0.0], p); # β = 2

julia> reliability_index(g, [0.0], p; method = FORM(RIA(nstarts = 3)));
```

## Performance measure approach

[`PMA`](@ref) minimizes the limit state over the `β_target` sphere. The
limit state must be scalar-valued and expressed with the **margin
convention** (positive values safe); the performance measure
`g* = g(x, p(u*))` satisfies `g* ≥ 0` whenever the reliability index is at
least `β_target`.

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> # margin convention: g = 3 + x₁ − p₁ − p₂ ≥ 0 is safe

julia> g = (x, p) -> 3.0 + x[1] - p[1] - p[2];

julia> p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2));

julia> # the minimum of g over ‖u‖ = 2 is attained at u = (√2, √2): g* = 3 + x₁ − 2√2

julia> performance_measure(g, [0.0], p; method = FORM(PMA(β_target = 2.0)));
```

## Second-order reliability method

[`SORM`](@ref) applies Breitung's curvature correction to the MPP of an
inner first-order method, improving the failure-probability estimate for
curved limit states.

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> # linear limit state: zero curvature, SORM == FORM

julia> g = (x, p) -> p[1] - 2.0;

julia> p = MvNormal(zeros(1), ones(1, 1));

julia> reliability_index(g, [0.0], p; method = SORM()); # 2.0

julia> # curved limit state: the Breitung correction refines Φ(−β)

julia> gc = (x, p) -> p[2] - (2.5 + 0.1 * p[1]^2);

julia> pc = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2));

julia> failure_probability(gc, [0.0], pc; method = SORM());
```

SORM indices are evaluation-only: differentiating them is not supported;
use [`FORM`](@ref) methods inside optimization objectives or constraints.
