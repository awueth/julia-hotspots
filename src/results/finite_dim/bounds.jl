# Phase 3: the finite-dimensional certificate values (the tuning loop).
#
# Computes exactly the rows of the `d = 10^18` table at the end of
# writeup/certificate.typ — no full pointwise `‖φ₁ − φ*‖∞` bound (there is no
# barrier at finite d). Read this file top-to-bottom to trace every summary
# value. The first ultracontractivity constant is the only expensive interval
# computation and is wrapped by caches.jl. Buffered wing masses are enclosed by
# cheap adaptive interval quadrature once before the tuning loop. Everything
# else is floating-point `Optim` and closed-form algebra, recomputed each run.
#
# finite_dim reuses the shared log_concave potential; it also reads the log_concave
# results (infinite-dimensional eigenvalue bounds + potential constants) and scales
# them to finite d via the comparison theorem (writeup/barrel.typ).

using Optim
using TOML
using IntervalArithmetic: interval, sup

include(joinpath(@__DIR__, "..", "..", "potentials", "lse_potential.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "solver.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "eigenfunction_hot_spot.jl"))
include(joinpath(@__DIR__, "..", "..", "ultracontractivity", "constants.jl"))

using .LSEPotentials:
    density_cell_bounds, interval_model, load_lse_potential, potential_functions
using .EigenfunctionHotSpot: sampled_hot_spot_difference
using .UltracontractivityConstants: first_constant
using .ValidatedQuadrature: integrate_box_adaptive

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const POTENTIAL_CHK = joinpath(
    PROJECT_ROOT, "checkpoints", "log_concave_extension", "high-resolution",
    "lse_global_potential.chk",
)
const FIT_CHK = joinpath(PROJECT_ROOT, "checkpoints", "first_eigenfunction_finite_dim.chk")
# Infinite-dimensional inputs from the log_concave pipeline.
const LOG_CONCAVE_DIR = joinpath(PROJECT_ROOT, "writeup", "results", "log_concave_extension", "high-resolution")
const POTENTIAL_TOML = joinpath(LOG_CONCAVE_DIR, "0-make-potential.toml")
const SUMMARY_TOML = joinpath(LOG_CONCAVE_DIR, "summary.toml")

int_tuple(v) = Tuple(Int.(v))

# ---------------------------------------------------------------------------
# Eigenvalue comparison theorem (writeup/barrel.typ, @thm:eigenvalue-convergence).
# (1-ε)²β λ_k(μ) ≤ λ_k(d) ≤ (1+ε)²/β λ_k(μ). Closed form, floating point.
# ---------------------------------------------------------------------------
function comparison_constants(sup_pot_val::Float64, sup_pot_grad::Float64, d::Float64)
    d > sup_pot_val ||
        throw(ArgumentError("dimension d must exceed sup_pot_val; got d=$d, sup_pot_val=$sup_pot_val"))
    epsilon = sqrt(sup_pot_val^2 + 0.25 * d * sup_pot_grad^2) / (d - sup_pot_val)
    log_beta = log1p(-sup_pot_val / d) - sup_pot_val^2 / (d - sup_pot_val)
    beta = log_beta < log(floatmin(Float64)) ? 0.0 : exp(log_beta)
    lower_multiplier = (1.0 - epsilon)^2 * beta
    upper_multiplier = beta == 0.0 ? Inf : (1.0 + epsilon)^2 / beta
    return (; epsilon, beta, lower_multiplier, upper_multiplier)
end

# ---------------------------------------------------------------------------
# Finite-dimensional ultracontractivity constants (writeup/ultracontractivity.typ).
# These are the log-space reimplementations of the module's finite_dim_barrier /
# finite_dim_correction_factor, which overflow at d = 1e18. They wrap the interval
# `first_constant` (via the cached `first_const`) inside float `Optim` searches.
# ---------------------------------------------------------------------------
function logaddexp(a::Float64, b::Float64)
    a == -Inf && return b
    b == -Inf && return a
    m = max(a, b)
    return m + log1p(exp(min(a, b) - m))
end

"""Logarithm of the finite-dimensional radial barrier, evaluated stably."""
function log_radial_barrier(
    sup_pot_val::Float64, d::Float64, alpha::Float64, beta::Float64, gamma::Float64, T::Float64,
)
    0 < alpha < beta < 2 || return Inf
    gamma > 0 || return Inf
    d >= (4gamma / (beta - alpha))^2 || return Inf

    sqrt_d = sqrt(d)
    log_K0 = alpha * (2gamma / (beta - alpha))^2
    log_A = log(beta * d) - gamma * sqrt_d +
            beta * (sup_pot_val / 2 - sup_pot_val^2 / (4d))
    log_A_term = log(2 * (1 + inv(d)) * T + 0.25) + log_A
    log_heat_tail = beta * d / 4 - gamma * sqrt_d -
                    ((d + 1) / 2) * log1p(4beta * T)
    return logaddexp(logaddexp(log_K0, log_A_term), log_heat_tail)
end

"""Optimize the two free radial-barrier parameters `beta` and `gamma`."""
function optimized_radial_barrier(sup_pot_val::Float64, d::Float64, alpha::Float64, T::Float64)
    0 < alpha < 2 || throw(ArgumentError("radial exponent must lie in (0, 2)"))
    T > 0 || throw(ArgumentError("barrier time must be positive"))

    # Logistic coordinates enforce alpha < beta < 2 and gamma < (beta-alpha)√d/4.
    function objective(x)
        p = inv(1 + exp(-x[1]))
        beta = alpha + (2 - alpha) * p
        q = inv(1 + exp(-x[2]))
        gamma = (beta - alpha) * sqrt(d) * q / 4
        return log_radial_barrier(sup_pot_val, d, alpha, beta, gamma, T)
    end

    optimum = optimize(
        objective, [0.0, -18.0], NelderMead(),
        Optim.Options(iterations=10_000, g_tol=1e-10),
    )
    x = Optim.minimizer(optimum)
    p = inv(1 + exp(-x[1]))
    beta = alpha + (2 - alpha) * p
    q = inv(1 + exp(-x[2]))
    gamma = (beta - alpha) * sqrt(d) * q / 4
    return (log_bound=Optim.minimum(optimum), beta=beta, gamma=gamma)
end

function finite_dim_first_correction(sup_pot_val::Float64, d::Float64, t::Float64)
    density_lower = (d - sup_pot_val) / d * exp(-sup_pot_val^2 / (d - sup_pot_val))
    eta = inv(t * d) * (1 + sup_pot_val^2 / 2)
    0 < density_lower && eta < 1 || error("finite-dimensional C1 correction is invalid")
    return inv(sqrt(density_lower * (1 - eta)))
end

"""Direct (crude) finite-dimensional `L² -> L∞` bound at total time `t`."""
function optimized_direct_split(sup_pot_val::Float64, d::Float64, t::Float64)
    t > 0.5 || throw(ArgumentError("the direct barrier bound requires total time t > 1/2"))
    lower_t1 = 0.5 + 1e-8
    upper_t1 = t - 1e-8
    lower_t1 < upper_t1 || throw(ArgumentError("no admissible time split for t=$t"))

    function split_objective(t1)
        alpha = inv(t1)
        radial = optimized_radial_barrier(sup_pot_val, d, alpha, t - t1)
        return alpha * ((2pi)^2 + 1) + radial.log_bound
    end

    split_optimum = optimize(split_objective, lower_t1, upper_t1, Brent(); abs_tol=1e-9, rel_tol=1e-9)
    t1 = Optim.minimizer(split_optimum)
    t2 = t - t1
    alpha = inv(t1)
    radial = optimized_radial_barrier(sup_pot_val, d, alpha, t2)
    return (t1=t1, t2=t2, alpha=alpha, radial=radial)
end

function direct_finite_dim_constant(
    sup_pot_val::Float64, d::Float64, t::Float64, cells_core, cells_wing,
)
    split = optimized_direct_split(sup_pot_val, d, t)
    C1_infinity = sup(first_const(POTENTIAL_CHK; t=split.t1, cells_core, cells_wing))
    C1 = C1_infinity * finite_dim_first_correction(sup_pot_val, d, split.t1)
    log_bound = log(C1) + split.alpha * ((2pi)^2 + 1) + split.radial.log_bound
    return (bound=exp(log_bound), log_bound=log_bound, t1=split.t1, t2=split.t2, C1=C1, radial=split.radial)
end

function core_buffer_terms(
    alpha::Float64, ell::Float64, radius::Float64, log_N::Float64,
    delta::Float64, infinite_mass_upper::Float64, finite_mass_upper::Float64,
)
    cutoff = ell + delta
    log_core = 2alpha * cutoff^2
    log_wing = finite_mass_upper == 0.0 ? -Inf :
        2alpha * radius^2 + log_N + log(finite_mass_upper)
    return (
        delta=delta,
        cutoff=cutoff,
        infinite_mass_upper=infinite_mass_upper,
        finite_mass_upper=finite_mass_upper,
        log_bracket=logaddexp(log_core, log_wing),
        log_core=log_core,
        log_wing=log_wing,
    )
end

"""Core-reduced finite-dimensional constant recomputed at total time `t`."""
function core_finite_dim_constant(
    sup_pot_val::Float64, d::Float64, t::Float64,
    ell::Float64, radius::Float64, buffer_delta::Float64,
    infinite_mass_upper::Float64, finite_mass_upper::Float64, cells_core, cells_wing,
)
    t > 2 || return nothing
    lower_t1 = 1.0 + 1e-7
    upper_t1 = t - 1.0 - 1e-7
    lower_t1 < upper_t1 || return nothing

    # Optimize the nested split using the dominant exponential factors; the
    # validated C1 factors are inserted at the selected split.
    function outer_objective(t1)
        t2 = t - t1
        inner = optimized_direct_split(sup_pot_val, d, t2 / 2)
        inner_log_without_C1 = inner.alpha * ((2pi)^2 + 1) + inner.radial.log_bound
        log_N_approx = 2inner_log_without_C1
        alpha = inv(t1)
        radial = optimized_radial_barrier(sup_pot_val, d, 2alpha, t2)
        buffer = core_buffer_terms(
            alpha, ell, radius, log_N_approx, buffer_delta,
            infinite_mass_upper, finite_mass_upper,
        )
        return alpha + 0.5buffer.log_bracket + 0.5radial.log_bound
    end

    outer_optimum = optimize(outer_objective, lower_t1, upper_t1, Brent(); abs_tol=1e-7, rel_tol=1e-7)
    t1 = Optim.minimizer(outer_optimum)
    t2 = t - t1
    alpha = inv(t1)

    outer_C1_infinity = sup(first_const(POTENTIAL_CHK; t=t1, cells_core, cells_wing))
    outer_C1 = outer_C1_infinity * finite_dim_first_correction(sup_pot_val, d, t1)

    inner = direct_finite_dim_constant(sup_pot_val, d, t2 / 2, cells_core, cells_wing)
    log_N = 2inner.log_bound
    radial = optimized_radial_barrier(sup_pot_val, d, 2alpha, t2)
    buffer = core_buffer_terms(
        alpha, ell, radius, log_N, buffer_delta,
        infinite_mass_upper, finite_mass_upper,
    )
    log_C2 = alpha + 0.5buffer.log_bracket + 0.5radial.log_bound
    log_bound = log(outer_C1) + log_C2
    return (
        bound=exp(log_bound), log_bound=log_bound, t1=t1, t2=t2,
        C1=outer_C1, C2=exp(log_C2), inner=inner, radial=radial, buffer=buffer,
    )
end

# ---------------------------------------------------------------------------
# Expensive interval quadratures (cache boundaries). Keyed on artifact PATHS plus
# cheap scalars only — never the deserialized objects. caches.jl rebinds these to
# on-disk caches.
# ---------------------------------------------------------------------------

# Sampled diagnostics: normalized boundary-residual sup ‖∂ₙφ*‖∞ (finite-d normal
# scaling applied in the solver), the floating-point sampled L∞ ‖φ*‖∞ at r=0, and
# the interior−boundary gap H(φ*) = sup_Ω° φ* − sup_∂Ω φ* (same sampling as the
# d=∞ hot-spot diagnostic, so both certificate tables report a comparable value).
function candidate_samples_impl(fit_chk, potential_chk; residual_grid, linf_grid, hot_spot_grid)
    fit = load_fitted_eigenfunction(fit_chk)
    potential = load_lse_potential(potential_chk)
    V, gradV = potential_functions(potential)
    geometry = make_geometry(fit.d, fit.diam_x, fit.diam_y, V, gradV, GridSampler(residual_grid...))
    residuals, _, _ = boundary_residual(geometry, fit.coefficients, fit.λ, fit.n_modes, residual_grid)
    residual_inf = maximum(abs, residuals)

    xs, ys, _ = GridSampler(linf_grid...)(fit.diam_x, fit.diam_y)
    sampled_linf = maximum(abs(u(fit, x, y, 0.0)) for x in xs, y in ys)
    hot_spot_effect = sampled_hot_spot_difference(fit, hot_spot_grid).effect
    return (; residual_inf, sampled_linf, hot_spot_effect)
end

# First ultracontractivity constant C₁ (interval); sup() gives the rigorous bound.
first_const_impl(potential_chk; t, cells_core, cells_wing) =
    first_constant(load_lse_potential(potential_chk); t, cells_core, cells_wing).C1
# Infinite-dimensional mass of the buffered wing
#   {|x₁| >= cutoff} × [-Ly, Ly]
# divided by Z. Symmetry is included in the numerator and therefore cancels
# correctly against the full normalization stored in the potential.
function buffered_wing_mass(potential_chk; cutoff)
    potential = load_lse_potential(potential_chk)
    potential.normalization === nothing && error("missing potential normalization certificate")
    dom = potential.domain
    dom.Lx <= cutoff < dom.x_max ||
        throw(ArgumentError("buffered wing cutoff must lie in [Lx, x_max)"))

    model = interval_model(potential.model)
    integrand(x, y) = density_cell_bounds(model, x, y)
    quadrant = integrate_box_adaptive(
        integrand,
        [interval(cutoff, dom.x_max), interval(0.0, dom.Ly)];
        init=(4, 8),
        atol=1e-14,
        maxcells=1_000,
    )
    full_mass = interval(dom.quadrant_factor) * quadrant
    normalization = interval(potential.normalization.lo, potential.normalization.hi)
    return full_mass / normalization
end

# ---------------------------------------------------------------------------
# Orchestration. Returns the summary sub-tables, fully formatted.
# ---------------------------------------------------------------------------
function compute_bounds(config)
    potential = load_lse_potential(POTENTIAL_CHK)
    fit = load_fitted_eigenfunction(FIT_CHK)
    lambda_star = Float64(fit.λ)
    d = Float64(fit.d)
    isinf(d) && error("finite_dim requires a finite dimension d")

    potential_result = TOML.parsefile(POTENTIAL_TOML)["result"]
    constants = potential_result["potential_constants"]
    sup_pot_val = Float64(constants["sup_pot_val"])
    sup_pot_grad = Float64(constants["sup_pot_grad"])

    # --- 1. Candidate diagnostics (residual + sampled L∞) -------------------
    ev = config["evaluate"]
    samples = candidate_samples(
        FIT_CHK, POTENTIAL_CHK;
        residual_grid=int_tuple(ev["residual_grid"]),
        linf_grid=int_tuple(ev["linf_grid"]),
        hot_spot_grid=int_tuple(ev["hot_spot_grid"]),
    )

    # --- 2. Eigenvalue bounds via the comparison theorem --------------------
    comparison = comparison_constants(sup_pot_val, sup_pot_grad, d)
    infinite = TOML.parsefile(SUMMARY_TOML)["eigenvalue_bounds"]
    lambda1_lower = comparison.lower_multiplier * Float64(infinite["lambda1_lower"])
    lambda1_upper = comparison.upper_multiplier * Float64(infinite["lambda1_upper"])
    lambda2_lower = comparison.lower_multiplier * Float64(infinite["lambda2_lower"])
    lambda_error = max(lambda_star - lambda1_lower, lambda1_upper - lambda_star)

    # --- 3. Finite-dimensional ultracontractivity constants -----------------
    uc = config["ultracontractivity"]
    decay_split = Float64(uc["decay_split"])
    cells_core = int_tuple(uc["first_constant_cells_core"])
    cells_wing = int_tuple(uc["first_constant_cells_wing"])
    times = Float64.(uc["times"])
    ell = Float64(potential.domain.Lx)
    radius = Float64(potential.domain.x_max)
    buffer_delta = Float64(uc["buffer_delta"])
    0 < buffer_delta < radius - ell ||
        throw(ArgumentError("buffer_delta must lie in (0, x_max - Lx)"))

    density_ratio_upper = d / (d - sup_pot_val) * exp(sup_pot_val^2 / (d - sup_pot_val))
    infinite_mass_upper = sup(buffered_wing_mass(
        POTENTIAL_CHK; cutoff=ell + buffer_delta,
    ))
    finite_mass_upper = density_ratio_upper * infinite_mass_upper

    direct_constants = [direct_finite_dim_constant(sup_pot_val, d, t, cells_core, cells_wing) for t in times]
    core_constants = [
        core_finite_dim_constant(
            sup_pot_val, d, t, ell, radius, buffer_delta,
            infinite_mass_upper, finite_mass_upper, cells_core, cells_wing,
        )
        for t in times
    ]
    ultracontractivity_constants = [
        core === nothing ? direct.bound : min(direct.bound, core.bound)
        for (direct, core) in zip(direct_constants, core_constants)
    ]

    # --- 4. Alpha-split pointwise head / tail terms -------------------------
    tail_rate = lambda1_upper - (1 - decay_split) * lambda2_lower
    tail_rate < 0 || error("pointwise tail does not decay; exponent is $tail_rate")
    candidate_s1_values = times ./ decay_split
    tail_grid = [candidate_s1_values; Inf]
    tail_interval_integrals = [
        isinf(tail_grid[i + 1]) ?
            -exp(tail_rate * tail_grid[i]) / tail_rate :
            (exp(tail_rate * tail_grid[i + 1]) - exp(tail_rate * tail_grid[i])) / tail_rate
        for i in eachindex(candidate_s1_values)
    ]

    # Minimize head_factor + tail_integral over s1; on interval i the left-endpoint
    # C_(αs_i) is fixed, so the only interior stationary point is explicit.
    function tail_bound_inside_interval(i::Int, s::Float64)
        later = i == lastindex(candidate_s1_values) ? 0.0 :
            sum(ultracontractivity_constants[(i + 1):end] .* tail_interval_integrals[(i + 1):end])
        interval_piece = isinf(tail_grid[i + 1]) ?
            -ultracontractivity_constants[i] * exp(tail_rate * s) / tail_rate :
            ultracontractivity_constants[i] *
            (exp(tail_rate * tail_grid[i + 1]) - exp(tail_rate * s)) / tail_rate
        return interval_piece + later
    end

    selection_candidates = Tuple{Float64,Int}[]
    for i in eachindex(candidate_s1_values)
        left, right = tail_grid[i], tail_grid[i + 1]
        push!(selection_candidates, (left, i))
        stationary = log(ultracontractivity_constants[i] * lambda1_lower / lambda1_upper) /
                     (lambda1_upper - tail_rate)
        left < stationary < right && push!(selection_candidates, (stationary, i))
    end
    selection_values = [
        expm1(s * lambda1_upper) / lambda1_lower + tail_bound_inside_interval(i, s)
        for (s, i) in selection_candidates
    ]
    selected_s1, selected_interval = selection_candidates[argmin(selection_values)]
    head_term = expm1(selected_s1 * lambda1_upper) / lambda1_lower
    tail_term = tail_bound_inside_interval(selected_interval, selected_s1)

    return Dict(
        "mps_candidate" => Dict(
            "lambda" => lambda_star,
            "n_modes" => collect(fit.n_modes),
            "dimension" => d,
        ),
        "residual" => Dict(
            "normal_derivative_inf" => samples.residual_inf,
        ),
        "eigenfunction" => Dict(
            "sampled_linf" => samples.sampled_linf,
            "hot_spot_effect" => samples.hot_spot_effect,
        ),
        "eigenvalue_bounds" => Dict(
            "lambda1_lower" => lambda1_lower,
            "lambda1_upper" => lambda1_upper,
            "lambda2_lower" => lambda2_lower,
            "lambda_error" => lambda_error,
            "epsilon" => comparison.epsilon,
            "beta" => comparison.beta,
            "lower_multiplier" => comparison.lower_multiplier,
            "upper_multiplier" => comparison.upper_multiplier,
        ),
        "pointwise" => Dict(
            "s1" => selected_s1,
            "alpha" => decay_split,
            "tail_rate" => tail_rate,
            "head_term" => head_term,
            "tail_term" => tail_term,
        ),
    )
end
