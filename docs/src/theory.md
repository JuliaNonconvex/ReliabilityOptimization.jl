# Theory

This page reviews the theory behind the reliability methods implemented in
this package: the reliability index, the first- and second-order reliability
methods (FORM/SORM), the performance measure approach (PMA), the
distribution transformations, and the sensitivity analysis that makes
everything differentiable.

## 1. Limit states and failure probability

A *limit state function* `g(x, p)` describes the performance of a design `x`
under random parameters `p`. The surface `g = 0` separates the safe domain
`{g ≤ 0}` from the failure domain `{g > 0}` (or the reverse, depending on
convention). The *probability of failure* is the probability mass of the
failure domain,

```math
P_f = \mathrm{P}[g(x, P) > 0] = \int_{g(x, p) > 0} f_P(p) \, \mathrm{d}p ,
```

where `f_P` is the joint density of the parameters. Directly evaluating this
integral is difficult: it is high-dimensional, the failure region is rarely
known explicitly (each evaluation of `g` may be a finite-element solve), and
the integrand is concentrated in a tiny corner of the space. Reliability
methods replace the integral with a geometric problem in a well-chosen
space.

## 2. Standard normal space

All reliability computations in this package happen in the *standard normal
space* `u ~ N(0, I)`, where the density is rotationally symmetric and decays
as `exp(−‖u‖²/2)` — most of the probability mass lives near the origin, and
the distance from the origin has a direct probabilistic meaning. The
mapping from the original parameters is handled by the
[distribution transformations](transforms.md):

- Gaussian parameters: the exact affine map `p = μ + L u` with
  `Σ = L Lᵀ` (Cholesky).
- Any univariate distribution: the Rosenblatt transform
  `p = F_X^{-1}(Φ(u))`, `u = Φ⁻¹(F_X(p))` — see the textbook table in
  [Distribution transformations](transforms.md).
- Correlated non-Gaussian parameters: the [`nataf_transform`](@ref), which
  solves the equivalent-normal correlation `ρ₀` so the back-transformed
  parameters have the target correlation.

## 3. The reliability index

The oldest index, the *Cornell index*, `β = μ_g / σ_g`, is not invariant
under reparametrization. The modern definition, due to Hasofer and Lind,
is geometric: the reliability index is the distance from the origin to the
failure surface in standard normal space,

```math
\beta = \min_{u} \;\; \|u\| \qquad \text{s.t.} \;\; g(x, T(u)) = 0 .
```

The minimizer `u*` is the *most probable point* (MPP) — the failure point
with the largest probability density — and `p* = T(u*)` is the MPP in the
original parameter space. Because the density is rotationally symmetric,
the probability content of the half-space tangent to the failure surface at
the MPP is exactly `Φ(−β)`:

```math
P_f \approx \Phi(-\beta) .
```

This is the *first-order reliability method* (FORM) estimate: exact for
linear limit states, asymptotic in `β` otherwise.

## 4. FORM: RIA and PMA

### Reliability index approach (RIA)

[`RIA`](@ref) solves the Hasofer-Lind problem directly: minimize `‖u‖²`
subject to `g(x, T(u)) = 0`. With the multiplier `λ` the KKT conditions are

```math
2 u + \nabla_u g^\mathsf{T} \lambda = 0, \qquad g = 0,
```

and `β = ‖u*‖`. The problem supports several limit states simultaneously
(`g` vector-valued): the MPP is then the closest point on the intersection
of the limit-state surfaces.

The MPP is also the linearization point of the output distribution: a
first-order Taylor expansion of `f(x, p)` around `p*` gives the Gaussian
approximation used by [`RandomFunction`](@ref),

```math
\begin{aligned}
\text{muf}  &= f(x, p_*) + \nabla_p f(x, p_*) \, (\mu - p_*), \\
\text{covf} &= \nabla_p f(x, p_*) \, \Sigma \, \nabla_p f(x, p_*)^\mathsf{T}.
\end{aligned}
```

### Performance measure approach (PMA)

[`PMA`](@ref) inverts the problem: it minimizes the limit state over the
target sphere,

```math
g_* = \min_{u} \;\; g(x, T(u)) \qquad \text{s.t.} \;\; \|u\| = \beta_t ,
```

with the *margin convention* (positive values safe). For well-behaved limit
states `g_* ≥ 0` if and only if `β ≥ β_t`, so the reliability requirement
`P_f ≤ Φ(−β_t)` becomes the ordinary constraint `g_* ≥ 0`. PMA tends to be
numerically better behaved than RIA: the sphere constraint keeps the
iterate away from the degenerate point `u = 0`, and it avoids the
equality-constraint failures that plague RIA when `β` is large.

### Sensitivities

The design sensitivities of `β` (RIA) or `g_*` (PMA) follow from the
implicit function theorem: the MPP solves a KKT system
`c(x, u(x), λ(x)) = 0`, so

```math
\frac{\mathrm{d} (u, \lambda)}{\mathrm{d} x}
= - \left[ \frac{\partial c}{\partial (u, \lambda)} \right]^{-1}
  \frac{\partial c}{\partial x} .
```

This package differentiates through the inner optimization problem with
[ImplicitDifferentiation.jl](https://github.com/JuliaDiff/ImplicitDifferentiation.jl),
which is why [`reliability_index`](@ref), [`performance_measure`](@ref) and
[`reliability_constraint`](@ref) are differentiable with respect to the
design.

## 5. SORM: curvature correction

FORM replaces the failure surface with its tangent hyperplane. The
*second-order reliability method* ([`SORM`](@ref)) keeps the quadratic term:
in coordinates with `u_n` along the MPP direction, the surface is locally

```math
u_n = \beta + \frac{1}{2} u_t^\mathsf{T} A u_t ,
```

where `A = -(\mathrm{I} - \alpha\alpha^\mathsf{T}) \nabla_u^2 g \, (\mathrm{I} - \alpha\alpha^\mathsf{T}) / \|\nabla_u g\|`
and `α = u*/β`. A Laplace-type (large-`β`) argument gives Breitung's
asymptotic formula

```math
P_f \approx \Phi(-\beta) \prod_{i=1}^{n-1} (1 + \beta \kappa_i)^{-1/2},
\qquad
\kappa_i = \text{eig}(A) ,
```

where the `κ_i` are the principal curvatures of the limit-state surface at
the MPP (positive when the surface curves away from the origin). The
correction is inactive for linear limit states (`κ = 0`) and undefined when
`1 + βκ_i ≤ 0` for some `i`, in which case the package falls back to FORM
with a warning. SORM indices are evaluation-only: differentiating through
the curvature computation is not supported.

## 6. What the approximations ignore

- **First-order output statistics.** The linearized mean and covariance are
  exact only for linear `f` in Gaussian parameters.
- **Local MPP.** The Hasofer-Lind problem is nonconvex in general; the inner
  solver finds a local solution, warm-started from the previous MPP. The
  deterministic multistarts of [`RIA`](@ref)/[`PMA`](@ref) mitigate this.
- **Tail dependence.** FORM/SORM model the rare-event tail through the
  geometry of the failure surface at a single point; they cannot capture
  tail dependence or multimodal failure regions, for which
  importance-sampling or subset-simulation methods are needed.
- **Numerical derivatives.** The default limit-state Jacobians are central
  finite differences; supply a `jacobian` function for exact derivatives.

## References

- Hasofer, A. M. and Lind, N. C. (1974). *Exact and invariant second-moment
  code format*. Journal of the Engineering Mechanics Division, 100(1).
- Rackwitz, R. and Fiessler, B. (1978). *Structural reliability under
  combined random load sequences*. Computers & Structures, 9(5).
- Breitung, K. (1984). *Asymptotic approximations for multinormal
  integrals*. Journal of Engineering Mechanics, 110(3).
- Hohenbichler, M. and Rackwitz, R. (1988). *Improvement of second-order
  reliability estimates by importance sampling*. Journal of Engineering
  Mechanics, 114(12).
- Madsen, H. O., Krenk, S. and Lind, N. C. (1986). *Methods of Structural
  Safety*. Prentice-Hall.
- Nataf, A. (1962). *Détermination des distributions de probabilité dont les
  marges sont données*. Comptes Rendus de l'Académie des Sciences, 225.
- Liu, P.-L. and Der Kiureghian, A. (1986). *Multivariate distribution
  models with prescribed marginals and covariances*. Probabilistic
  Engineering Mechanics, 1(2).
- Tu, J., Choi, K. K. and Park, Y. H. (1999). *A new study on
  reliability-based design optimization*. Journal of Mechanical Design,
  121(4).
- Melchers, R. E. (1999). *Structural Reliability Analysis and Prediction*.
  Wiley.
- Nowak, A. S. and Collins, K. R. (2012). *Reliability of Structures*.
  CRC Press.
