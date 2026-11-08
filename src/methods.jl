# Reliability methods: FORM with RIA and PMA inner problems.

"""
    FORM(method)

First-order reliability method wrapper. `method` selects the inner
reliability problem: [`RIA`](@ref) or [`PMA`](@ref).
"""
struct FORM{M}
    method::M
end

"""
    RIA(optim_alg, optim_options; nstarts = 1)

Reliability index approach (Hasofer-Lind): minimize `‖u‖²` subject to the
limit state `g(x, p(u)) = 0` in standard normal space. The reliability index
is `β = ‖u*‖` and `p* = p(u*)` is the most probable point (MPP). Supports
scalar- and vector-valued limit states.

- `optim_alg`: any Nonconvex optimizer supporting equality constraints
  (default: [`IpoptAlg`](@ref)).
- `optim_options`: options forwarded to the inner solver.
- `nstarts`: number of deterministic inner-solve starts (warm start plus
  `nstarts - 1` axis-aligned points); the best feasible MPP is kept.
"""
struct RIA{A,O}
    optim_alg::A
    optim_options::O
    nstarts::Int
end
RIA(optim_alg::A, optim_options::O; nstarts::Integer = 1) where {A,O} =
    RIA(optim_alg, optim_options, Int(nstarts))
RIA(;
    optim_alg = IpoptAlg(),
    optim_options = IpoptOptions(print_level = 0, max_wall_time = 30.0, tol = 1e-12),
    nstarts::Integer = 1,
) = RIA(optim_alg, optim_options, Int(nstarts))

"""
    PMA(; β_target = 3.0, optim_alg = IpoptAlg(), optim_options = ..., nstarts = 1)

Performance measure approach: minimize the limit state `g(x, p(u))` subject
to `‖u‖² = β_target²` in standard normal space. The limit state must be
scalar-valued and expressed with the **margin convention** (positive values
safe, `g = 0` the failure surface); the performance measure
`g* = g(x, p(u*))` satisfies `g* ≥ 0` whenever the reliability index is at
least `β_target`. See [`performance_measure`](@ref) and
[`reliability_constraint`](@ref).
"""
struct PMA{A,O}
    optim_alg::A
    optim_options::O
    β_target::Float64
    nstarts::Int
end
function PMA(;
    β_target::Real = 3.0,
    optim_alg = IpoptAlg(),
    optim_options = IpoptOptions(print_level = 0, max_wall_time = 30.0, tol = 1e-12),
    nstarts::Integer = 1,
)
    return PMA(optim_alg, optim_options, Float64(β_target), Int(nstarts))
end

"""
    NestedSolver

A reusable inner-optimization workspace: the inner `VecModel` and the solver
`Workspace` are built once and re-used across evaluations of the outer
design variables, warm-starting each solve from the previous solution.
"""
mutable struct NestedSolver{W,M,X}
    workspace::W
    vecmodel::M
    x_ref::X
    start::Vector{Float64}
end

function _solve!(s::NestedSolver, start::AbstractVector)
    ws = s.workspace
    hasproperty(ws, :x0) && (ws.x0 .= start)
    return optimize!(ws)
end

function _ria_solver(f, tr, alg, options, x_ref)
    n = dim(tr)
    obj = u -> dot(u, u)
    constr = u -> f(x_ref[], from_u(tr, u))
    model = Model(obj)
    addvar!(model, fill(-Inf, n), fill(Inf, n))
    add_eq_constraint!(model, constr)
    vmodel, x0, _ = tovecmodel(model, zeros(n))
    ws = Workspace(vmodel, alg, copy(x0); options = options)
    return NestedSolver(ws, vmodel, x_ref, zeros(n))
end

function _pma_solver(f, tr, alg, options, β_target, x_ref)
    n = dim(tr)
    obj = u -> _scalar_limit_state(f(x_ref[], from_u(tr, u)))
    constr = u -> dot(u, u) - β_target^2
    model = Model(obj)
    addvar!(model, fill(-Inf, n), fill(Inf, n))
    add_eq_constraint!(model, constr)
    vmodel, x0, _ = tovecmodel(model, zeros(n))
    ws = Workspace(vmodel, alg, copy(x0); options = options)
    return NestedSolver(ws, vmodel, x_ref, fill(β_target / sqrt(n), n))
end

# Deterministic multistart points: the warm start followed by axis-aligned
# points of growing radius (avoids a Random dependency and is reproducible).
function _starts(warm::AbstractVector, nstarts::Integer; radius::Real)
    starts = [copy(warm)]
    n = length(warm)
    for i in 1:(nstarts - 1)
        u = zeros(n)
        d = (i - 1) % n + 1
        u[d] = radius * (1 + (i - 1) ÷ n)
        push!(starts, u)
    end
    return starts
end

# Select the best candidate from the multistart solves: feasible solutions
# always beat infeasible ones; among feasible solutions minimize the
# reliability index (RIA) or the limit state (PMA); among infeasible ones
# minimize the constraint violation.
function _best_candidate(candidates, key)
    best = first(candidates)
    for c in Iterators.drop(candidates, 1)
        if (c.viol ≤ 1e-6 && best.viol > 1e-6) ||
           (c.viol ≤ 1e-6 == best.viol ≤ 1e-6 && key(c) < key(best)) ||
           (c.viol > 1e-6 && best.viol > 1e-6 && c.viol < best.viol)
            best = c
        end
    end
    return best
end

function get_forward(f, tr, method::FORM{<:RIA}, jac)
    @unpack optim_alg, optim_options, nstarts = method.method
    x_ref = Ref{Vector{Float64}}(zeros(dim(tr)))
    solver = Ref{Union{Nothing,NestedSolver}}(nothing)
    function forward(x)
        x_ref[] = float.(vec(x))
        if solver[] === nothing
            solver[] = _ria_solver(f, tr, optim_alg, optim_options, x_ref)
        end
        s = solver[]
        candidates = map(_starts(s.start, nstarts; radius = 1.0)) do st
            result = _solve!(s, st)
            u = vec(result.minimizer)
            gval = _vec(f(x_ref[], from_u(tr, u)))
            return (
                viol = maximum(abs, gval),
                u = u,
                mult = vec(result.problem.mult_g),
            )
        end
        best = _best_candidate(candidates, c -> dot(c.u, c.u))
        s.start .= best.u
        # ImplicitDifferentiation ≥ 0.9 requires the solver to return
        # (solution, byproduct); there is no meaningful byproduct here.
        return vcat(best.u, best.mult), nothing
    end
    return forward
end

function get_forward(f, tr, method::FORM{<:PMA}, jac)
    @unpack optim_alg, optim_options, β_target, nstarts = method.method
    x_ref = Ref{Vector{Float64}}(zeros(dim(tr)))
    solver = Ref{Union{Nothing,NestedSolver}}(nothing)
    function forward(x)
        x_ref[] = float.(vec(x))
        if solver[] === nothing
            solver[] = _pma_solver(f, tr, optim_alg, optim_options, β_target, x_ref)
        end
        s = solver[]
        candidates = map(_starts(s.start, nstarts; radius = β_target)) do st
            result = _solve!(s, st)
            u = vec(result.minimizer)
            gval = _scalar_limit_state(f(x_ref[], from_u(tr, u)))
            return (
                viol = abs(dot(u, u) - β_target^2),
                u = u,
                mult = vec(result.problem.mult_g),
                g = gval,
            )
        end
        best = _best_candidate(candidates, c -> c.g)
        s.start .= best.u
        return vcat(best.u, best.mult), nothing
    end
    return forward
end

function get_conditions(f, tr::AffineTransform, method::FORM{<:RIA}, jac)
    L = tr.L
    n = dim(tr)
    # ImplicitDifferentiation ≥ 0.9 passes the solver byproduct as the third
    # argument of the conditions function.
    return function kkt_conditions(x, y, z)
        u = y[1:n]
        λ = y[n+1:end]
        p = from_u(tr, u)
        J = jac(f, x, p)
        return vcat(2u + L' * (J' * λ), _vec(f(x, p)))
    end
end

function get_conditions(f, tr::StandardNormalTransform, method::FORM{<:RIA}, jac)
    n = dim(tr)
    return function kkt_conditions(x, y, z)
        u = y[1:n]
        λ = y[n+1:end]
        # The chain rule through a general (non-affine) transformation is
        # handled with finite differences on the Lagrangian in u-space.
        ∇u = FiniteDifferences.grad(
            fdm, q -> dot(λ, _vec(f(x, tr.from_u(q)))), u
        )[1]
        return vcat(2u + ∇u, _vec(f(x, tr.from_u(u))))
    end
end

function get_conditions(f, tr::AffineTransform, method::FORM{<:PMA}, jac)
    L = tr.L
    n = dim(tr)
    β_target = method.method.β_target
    return function kkt_conditions(x, y, z)
        u = y[1:n]
        λ = y[n+1]
        p = from_u(tr, u)
        J = jac(f, x, p)
        ∇u = L' * vec(J)
        return vcat(∇u + 2λ * u, dot(u, u) - β_target^2)
    end
end

function get_conditions(f, tr::StandardNormalTransform, method::FORM{<:PMA}, jac)
    n = dim(tr)
    β_target = method.method.β_target
    return function kkt_conditions(x, y, z)
        u = y[1:n]
        λ = y[n+1]
        ∇u = FiniteDifferences.grad(
            fdm, q -> _scalar_limit_state(f(x, tr.from_u(q))), u
        )[1]
        return vcat(∇u + 2λ * u, dot(u, u) - β_target^2)
    end
end

function get_implicit(f, tr, method, jac)
    forward = get_forward(f, tr, method, jac)
    kkt_conditions = get_conditions(f, tr, method, jac)
    # The conditions embed a finite-difference Jacobian (see `_jacobian`),
    # whose derivative with respect to the parameters cannot be obtained by
    # reverse-mode AD. Differentiate the conditions with finite differences
    # as well, so the curvature term λᵀ d²g/dp² of the KKT Jacobian is
    # included instead of being silently dropped.
    backends = (; x = AutoFiniteDifferences(fdm), y = AutoFiniteDifferences(fdm))
    return ImplicitFunction(forward, kkt_conditions; backends = backends)
end
