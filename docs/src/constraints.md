# Reliability constraints

## Reliability index and failure probability

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> g = (x, p) -> p[1] - 2.0 - x[1];

julia> p = MvNormal(zeros(1), ones(1, 1));

julia> reliability_index(g, [0.0], p); # 2.0

julia> failure_probability(g, [0.0], p); # Φ(−2)
```

With `method = SORM(...)` the second-order (Breitung-corrected) values are
returned instead.

## Performance measure

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> g = (x, p) -> 3.0 + x[1] - p[1] - p[2]; # margin convention

julia> p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2));

julia> performance_measure(g, [0.0], p); # PMA by default: g* = 3 − 2√2
```

The performance measure is the minimum of the limit state over the
`β_target` sphere in standard normal space; `g* ≥ 0` exactly when the
reliability index is at least `β_target`. The default method is
`FORM(PMA())` with `β_target = 3.0`; pass `method = FORM(PMA(β_target = ...))`
for a custom target.

## Probabilistic constraints

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> g = (x, p) -> 3.0 + x[1] - p[1] - p[2]; # margin convention

julia> p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2));

julia> # c(x) ≤ 0 ⟺ P(g(x, p) ≤ 0) ≥ Φ(2): PMA by default

julia> c = reliability_constraint(g, p, 2.0);

julia> c([-0.3]); # > 0: unreliable

julia> c([0.3]); # < 0: reliable

julia> # RIA formulation: c(x) = β_target − β(x)

julia> c_ria = reliability_constraint(g, p, 2.0; method = FORM(RIA()));

julia> # probabilistic_constraint is an alias of reliability_constraint

julia> c_alias = probabilistic_constraint(g, p, 2.0);

julia> c_alias([0.0]) ≈ c([0.0]);
```

The returned closure keeps a single reusable inner-optimization workspace
and is differentiable with respect to `x`, so it can be added directly to a
Nonconvex model:

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions, NonconvexIpopt

julia> g = (x, p) -> 3.0 + x[1] - p[1] - p[2];

julia> p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2));

julia> c = reliability_constraint(g, p, 2.0);

julia> m = Model(x -> x[1]^2 + x[2]^2);

julia> addvar!(m, [-2.0, -2.0], [2.0, 2.0]);

julia> add_ineq_constraint!(m, c);

julia> r = optimize(m, IpoptAlg(), [0.5, 0.5]; options = IpoptOptions(print_level = 0));
```
