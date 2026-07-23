#= Step 3
This file computes the pointwise distance between the approximate eigenfunction
and the true eigenfunction, as well as the hot-spot effect.  
The expensive interval quadratures are isolated into the five `*_impl` functions below; 
caches.jl wraps them under the bare names (`fem_eigenvalues`, `fourier_core_integrals`,
`first_const`, `second_const`, `total_const`) that the orchestration calls, 
so tuning reuses them from disk.
=#

using IntervalArithmetic
using TaylorModels: inf, sup

include(joinpath(@__DIR__, "..", "..", "potentials", "lse_potential.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "solver.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "fourier_energy_integration.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "eigenvalue_fem.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "eigenfunction_linf_norm.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "eigenfunction_hot_spot.jl"))
include(joinpath(@__DIR__, "..", "..", "ultracontractivity", "constants.jl"))

using .LSEPotentials
using .EigenvalueFEM: compute_fem_eigenvalues
using .EigenfunctionLinfNorm: interval_linf_norm
using .EigenfunctionHotSpot: sampled_hot_spot_difference
using .UltracontractivityConstants: first_constant, second_constant, total_constant

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
# The shared high-resolution artifacts (also consumed by finite_dim). Override
# LCE_ARTIFACT_DIR only for testing against the low-resolution / debug builds.
const ARTIFACT_DIR = get(
    ENV, "LCE_ARTIFACT_DIR",
    joinpath(PROJECT_ROOT, "checkpoints", "log_concave_extension", "high-resolution"),
)
const POTENTIAL_CHK = joinpath(ARTIFACT_DIR, "lse_global_potential.chk")
const FIT_CHK = joinpath(ARTIFACT_DIR, "first_eigenfunction.chk")

int_tuple(v) = Tuple(Int.(v))

function wing_correction_factor(lambda_upper, t, C, wing_mass; multiplicity::Integer=1, include_constant_mode::Bool=false)
    multiplicity > 0 || throw(ArgumentError("multiplicity must be positive"))
    eigenfunction_term =
        interval(multiplicity) * exp(interval(2.0) * lambda_upper * t) * C^2
    constant_term = include_constant_mode ? interval(1.0) : interval(0.0)
    return interval(1.0) - (constant_term + eigenfunction_term) * wing_mass
end

function _exp_integral(a, b, rate)
    rate < 0.0 || error("tail exponent must be negative, got $rate")
    isinf(b) && return -exp(rate * a) / rate
    return (exp(rate * b) - exp(rate * a)) / rate
end

# ---------------------------------------------------------------------------
# Expensive interval quadratures (cache boundaries). Keyed on artifact PATHS
# plus cheap scalars only — never the deserialized objects, whose hash is not
# content-stable across Julia sessions. caches.jl rebinds these `*_impl`
# functions to on-disk caches.
# ---------------------------------------------------------------------------

# FEM Neumann eigenvalues on the symmetric core (floating point).
function fem_eigenvalues_impl(potential_chk; partition, nev)
    potential = load_lse_potential(potential_chk)
    values, _, oscillation = compute_fem_eigenvalues(potential; partition, nev)
    return (values=collect(values), oscillation=oscillation.value)
end

# Core Fourier energy integrals: numerator = ∫|∇u|² dμ, denominator = ∫u² dμ
# over the first quadrant. Serves both the L² norm (denominator) and the
# Rayleigh quotient (both). ~20 min at final resolution; the closure is buried
# here so it never enters the cache key.
function fourier_core_integrals_impl(potential_chk, fit_chk; cells)
    potential = load_lse_potential(potential_chk)
    fit_interval = load_fitted_eigenfunction(fit_chk; intervalized=true)
    energy = prepare_fourier_energy(fit_interval)
    density_model = interval_model(potential.model)
    weight_bounds(x, y) = density_cell_bounds(density_model, x, y)
    core_box = [interval(0.0, potential.domain.Lx), interval(0.0, potential.domain.Ly)]
    return integrate_weighted_fourier_energy(energy, weight_bounds, core_box; cells=cells)
end

# Sampled/searched candidate diagnostics: unnormalized boundary residual sup,
# interval L∞ norm, and the sampled interior−boundary hot-spot gap. Independent
# of every tuned hyperparameter.
function candidate_samples_impl(potential_chk, fit_chk; residual_grid, hot_spot_grid, linf_atol)
    potential = load_lse_potential(potential_chk)
    fit = load_fitted_eigenfunction(fit_chk)
    fit_interval = load_fitted_eigenfunction(fit_chk; intervalized=true)
    V, gradV = potential_functions(potential)
    geometry = make_geometry(fit.d, fit.diam_x, fit.diam_y, V, gradV, GridSampler(residual_grid...))
    residuals, _, _ = boundary_residual(
        geometry, fit.coefficients, fit.λ, fit.n_modes, GridSampler(residual_grid...),
    )
    residual_value = maximum(abs, residuals)
    linf = interval_linf_norm(fit_interval, linf_atol).linf
    hot_spot_effect = sampled_hot_spot_difference(fit, hot_spot_grid).effect
    return (; residual_value, linf, hot_spot_effect)
end

# Ultracontractivity constants. C1 (first) and C2 (second) are cached separately
# so lower_bounds (which uses them at distinct t/cell parameters) and pointwise
# (which uses total_constant = C1*C2 per time) share enclosures.
first_const_impl(potential_chk; t, cells_core, cells_wing) =
    first_constant(load_lse_potential(potential_chk); t, cells_core, cells_wing).C1
second_const_impl(potential_chk; t1, t2, delta, ny_wing) =
    second_constant(load_lse_potential(potential_chk); t1, t2, delta, ny_wing).C2
total_const_impl(potential_chk, t1, t2; cells_core, cells_wing) =
    total_constant(load_lse_potential(potential_chk), t1, t2; cells_core, cells_wing)

# Compute the four summary sub-tables.
function compute_bounds(config)
    potential = load_lse_potential(POTENTIAL_CHK)
    fit = load_fitted_eigenfunction(FIT_CHK)
    fit_interval = load_fitted_eigenfunction(FIT_CHK; intervalized=true)

    Lx = potential.domain.Lx
    Ly = potential.domain.Ly
    measure_mass = interval(potential.normalization.lo, potential.normalization.hi)
    wing_mass = interval(potential.wing_mass.lo, potential.wing_mass.hi)
    quadrant_factor = interval(potential.domain.quadrant_factor)
    quadrant_wing_mass = measure_mass * wing_mass / quadrant_factor

    # 1. FEM eigenvalues
    fem = config["fem"]
    partition = int_tuple(fem["partition"])
    nev = Int(fem["nev"])
    h_K = hypot(2Lx / partition[1], 2Ly / partition[2])
    fem_result = fem_eigenvalues(POTENTIAL_CHK; partition, nev)
    eigenvalues = fem_result.values
    oscillation = fem_result.oscillation

    # 2. Candidate diagnostics (no L² normalization)
    # The hot-spot criterion H(φ*) > 2E(φ*) is scale-invariant (remark after
    # @thm:pointwise-limit), and since μ is a probability measure the pointwise
    # bound uses ‖·‖_L²(μ) ≤ ‖·‖_∞, so only L∞ diagnostics enter. 
    ev = config["evaluate"]
    samples = candidate_samples(
        POTENTIAL_CHK, FIT_CHK;
        residual_grid=int_tuple(ev["residual_grid"]),
        hot_spot_grid=int_tuple(ev["hot_spot_grid"]),
        linf_atol=Float64(ev["linf_atol"]),
    )
    residual_linf = samples.residual_value
    phi_linf = sup(samples.linf)
    hot_spot_effect = samples.hot_spot_effect

    # 3. Rayleigh quotient (eigenvalue upper bound)
    core_cells = int_tuple(config["rayleigh"]["core_cells"])
    linf_bounds = fourier_linf_bounds(fit_interval)
    ray_core = fourier_core_integrals(POTENTIAL_CHK, FIT_CHK; cells=core_cells)
    wing_numerator = interval(Float64, 0.0, sup(linf_bounds.gradient^2 * quadrant_wing_mass))
    ray_wing_denominator = interval(Float64, 0.0, sup(linf_bounds.u^2 * quadrant_wing_mass))
    numerator = ray_core.numerator + wing_numerator
    denominator = ray_core.denominator + ray_wing_denominator
    rayleigh = numerator / denominator
    lambda1_upper = sup(rayleigh)

    # 5. Crouzeix–Raviart lower bounds + wing corrections
    lb = config["lower_bounds"]
    positive_tolerance = Float64(lb["fem_positive_tolerance"])
    positive_eigenvalues = [v for v in eigenvalues if v > positive_tolerance]
    length(positive_eigenvalues) >= 2 ||
        error("FEM result must contain at least two positive eigenvalues")
    lambda1_h, lambda2_h = positive_eigenvalues[1:2]
    cr_constant = Float64(lb["cr_interpolation_constant"])
    C_h = exp(oscillation) * cr_constant * h_K
    lambda1_core = lambda1_h / (1 + lambda1_h * C_h^2)
    lambda2_core = lambda2_h / (1 + lambda2_h * C_h^2)

    cells_core = int_tuple(lb["first_constant_cells_core"])
    cells_wing = int_tuple(lb["first_constant_cells_wing"])
    ny_wing = Int(lb["second_constant_ny_wing"])
    ultra_C(params) = begin
        t1 = Float64(params["t1"]); t2 = Float64(params["t2"]); delta = Float64(params["delta"])
        C1 = first_const(POTENTIAL_CHK; t=t1, cells_core, cells_wing)
        C2 = second_const(POTENTIAL_CHK; t1, t2, delta, ny_wing)
        return (C=C1 * C2, t=interval(t1) + interval(t2))
    end
    lambda1_parts = ultra_C(lb["lambda1"])
    lambda2_parts = ultra_C(lb["lambda2"])
    lambda1_factor = wing_correction_factor(
        interval(lambda1_upper), lambda1_parts.t, lambda1_parts.C, wing_mass; multiplicity=1,
    )
    lambda2_factor = wing_correction_factor(
        interval(Float64(lb["lambda2_upper"])),
        lambda2_parts.t,
        lambda2_parts.C,
        wing_mass;
        multiplicity=2,
        include_constant_mode=true,
    )
    lambda1_lower = inf(interval(lambda1_core) * lambda1_factor)
    lambda2_lower = inf(interval(lambda2_core) * lambda2_factor)

    # 6. Pointwise head/tail combination
    pw = config["pointwise"]
    alpha = Float64(pw["alpha"])
    ultracontractivity_t2 = Float64(pw["ultracontractivity_t2"])
    pw_cells_core = int_tuple(pw["ultracontractivity_cells_core"])
    pw_cells_wing = int_tuple(pw["ultracontractivity_cells_wing"])
    lambda_star = Float64(fit.λ)
    lambda_error = max(lambda_star - lambda1_lower, lambda1_upper - lambda_star)
    time_grid = Float64.(pw["times"])
    length(time_grid) >= 2 && isinf(last(time_grid)) ||
        error("pointwise time grid must end at +Inf")
    issorted(time_grid) || error("pointwise time grid must be sorted")
    ultracon_times = time_grid[1:(end - 1)]
    tail_rate = lambda1_upper - lambda2_lower * (1.0 - alpha)
    tail_decays = tail_rate < 0.0
    ultracon_splits = [(alpha * s - ultracontractivity_t2, ultracontractivity_t2) for s in ultracon_times]
    all(first(split) > 0 for split in ultracon_splits) || error("all ultracontractivity t1 values must be positive")
    constants_hi = [
        sup(total_const(POTENTIAL_CHK, t1, t2; cells_core=pw_cells_core, cells_wing=pw_cells_wing))
        for (t1, t2) in ultracon_splits
    ]
    if tail_decays
        tail_interval_integrals = [
            _exp_integral(time_grid[i], time_grid[i + 1], tail_rate)
            for i in 1:(length(time_grid) - 1)
        ]
        tail_integrals = [
            sum(constants_hi[i:end] .* tail_interval_integrals[i:end])
            for i in eachindex(ultracon_times)
        ]
    else
        tail_integrals = fill(Inf, length(ultracon_times))
    end
    head_integrals = [(exp(s1 * lambda1_upper) - 1.0) / lambda1_lower for s1 in ultracon_times]
    # head/tail are the bare integral factors (the two integral rows of the
    # certificate table). The single coefficient
    # ‖(L-λ₁)φ*‖ ≤ ‖(L-λ*)φ*‖_∞ + |λ*-λ₁| ‖φ*‖_∞ multiplies their sum; it bounds
    # both the L∞ (head) and, since μ is a probability measure, the L²(μ) (tail)
    # contribution, so a single factor covers both legs (@thm:pointwise-limit).
    coefficient = residual_linf + lambda_error * phi_linf
    integral_factors = head_integrals .+ tail_integrals
    pointwise_bounds = coefficient .* integral_factors
    selected = argmin(pointwise_bounds)

    return Dict(
        "fem" => Dict(
            "partition" => collect(partition),
            "h_K" => h_K,
            "max_oscillation" => oscillation,
            "raw_eigenvalues" => eigenvalues,
        ),
        "mps_candidate" => Dict(
            "lambda" => lambda_star,
            "n_modes" => collect(fit.n_modes),
            "residual_inf" => residual_linf,
            "linf" => phi_linf,
            "hot_spot_effect" => hot_spot_effect,
        ),
        "eigenvalue_bounds" => Dict(
            "lambda1_lower" => lambda1_lower,
            "lambda1_upper" => lambda1_upper,
            "lambda2_lower" => lambda2_lower,
            "core_lambda1_cr_lower" => lambda1_core,
            "core_lambda2_cr_lower" => lambda2_core,
            "lambda1_wing_correction_factor" => Dict("lo" => inf(x), "hi" => sup(x)),
            "lambda2_wing_correction_factor" => Dict("lo" => inf(x), "hi" => sup(x)),
        ),
        "pointwise_limit" => Dict(
            "s1" => ultracon_times[selected],
            "alpha" => alpha,
            "tail_rate" => tail_rate,
            "tail_decays" => tail_decays,
            "lambda_error" => lambda_error,
            "residual_bound" => residual_linf,
            "phi_linf_bound" => phi_linf,
            "coefficient" => coefficient,
            "head_term" => head_integrals[selected],
            "tail_term" => tail_integrals[selected],
            "pointwise_linf_bound" => pointwise_bounds[selected],
        ),
    )
end
