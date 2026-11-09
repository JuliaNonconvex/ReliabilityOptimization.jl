# Random functions

A [`RandomFunction`](@ref) wraps a deterministic function `f(x, p)` of the
design variables `x` and the random parameters `p` into a callable that
returns the first-order Gaussian approximation of the output distribution,
linearized at the point selected by the reliability method.

Calling `rf(x)` returns `MvNormal(muf, covf)` with

```math
\text{muf}  = f(x, p_0) + \nabla_p f(x, p_0) \, (\mu - p_0),
\qquad
\text{covf} = \nabla_p f(x, p_0) \, \Sigma \, \nabla_p f(x, p_0)^\mathsf{T},
```

where `μ` and `Σ` are the mean and covariance of `p` and `p0` is the point
selected by the method (the MPP for [`RIA`](@ref), the minimum-performance
point for [`PMA`](@ref)).

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> f = (x, p) -> [p[1] * x[1], p[2] - x[2]];

julia> p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2));

julia> rf = RandomFunction(f, p); # FORM(RIA()) by default

julia> d = rf([1.5, 2.0]); # MvNormal(muf, covf)

julia> mean(d); # [0.0, −2.0]

julia> cov(d); # [2.25 0; 0 1]

julia> # robust objective: mean + k·std, differentiable w.r.t. x

julia> using Zygote

julia> obj = x -> mean(rf(x))[1] + 2 * sqrt(cov(rf(x))[1, 1]);

julia> Zygote.gradient(obj, [1.5, 2.0])[1];
```

## The most probable point

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions

julia> g = (x, p) -> p[1] - 2.0 - x[1];

julia> p = MvNormal(zeros(1), ones(1, 1));

julia> getp0(g, [0.0], p); # [2.0]
```

## Custom Jacobians

The default limit-state Jacobian `∂f/∂p` is central finite differences. A
custom Jacobian — for example a Zygote-based one — can be supplied for
better performance; it must be differentiable itself for gradients of the
output distribution:

```jldoctest; output = false, filter = r"[\s\S]*" => ""
julia> using ReliabilityOptimization, Distributions, Zygote

julia> f = (x, p) -> [p[1] * x[1], p[2] - x[2]];

julia> p = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2));

julia> jac = (f, x, p) -> Zygote.jacobian(q -> f(x, q), p)[1];

julia> rf = RandomFunction(f, p; jacobian = jac);
```
