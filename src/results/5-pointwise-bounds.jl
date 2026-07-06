using TOML
using IntervalArithmetic: inf, sup

include(joinpath(@__DIR__, "..", "ultracontractivity", "constants.jl"))

using .UltracontractivityConstants: total_constant
using .LSEPotentials: load_lse_potential

project_root = normpath(joinpath(@__DIR__, "..", ".."))
result_dir = joinpath(project_root, "results")
result_path = joinpath(result_dir, "5-pointwise-bounds.toml")

fem_result_path = joinpath(result_dir, "1-fem-eigenvalues.toml")
fit_result_path = joinpath(result_dir, "2a-fit-mps-candidate.toml")
residual_result_path = joinpath(result_dir, "2b-evaluate-mps-candidate.toml")
upper_result_path = joinpath(result_dir, "3-eigenvalues-upper-bounds.toml")
lower_result_path = joinpath(result_dir, "4-eigenvalues-lower-bounds.toml")

project_relative(path) = relpath(normpath(path), project_root)
interval_table(lo, hi) = Dict("lo" => lo, "hi" => hi)
interval_table(x) = interval_table(inf(x), sup(x))

function exp_integral(a, b, rate)
    rate < 0.0 || error("Tail exponent must be negative, got $rate")
    if isinf(b)
        return -exp(rate * a) / rate
    end
    return (exp(rate * b) - exp(rate * a)) / rate
end

fem_result = TOML.parsefile(fem_result_path)
fit_result = TOML.parsefile(fit_result_path)
residual_result = TOML.parsefile(residual_result_path)
upper_result = TOML.parsefile(upper_result_path)
lower_result = TOML.parsefile(lower_result_path)

potential_checkpoint = joinpath(project_root, fit_result["inputs"]["potential_checkpoint"])
fit_checkpoint = joinpath(project_root, fit_result["outputs"]["fit_checkpoint"])
potential = load_lse_potential(potential_checkpoint)

lambda_star = fit_result["parameters"]["lambda"]
lambda1_lower = lower_result["result"]["full_eigenvalue_lower_bound"]
lambda1_upper = upper_result["result"]["lambda1_upper"]
lambda2_lower = fem_result["result"]["eigenvalues"][3]
lambda_error = max(lambda_star - lambda1_lower, lambda1_upper - lambda_star)

l2_norm = upper_result["result"]["l2_norm"]
l2_norm_lower = l2_norm["lo"]
l2_norm_upper = l2_norm["hi"]
l2_scale_upper = inv(sqrt(l2_norm_lower))
l2_scale_lower = inv(sqrt(l2_norm_upper))

residual_linf = residual_result["result"]["high_resolution_residual_inf"]
normalized_residual_linf = residual_linf * l2_scale_upper
# Since mu is a probability measure, ||r||_L2(mu) <= ||r||_oo.
normalized_residual_l2_bound = normalized_residual_linf

phi_linf = residual_result["result"]["sampled_linf"]
phi_linf_location = residual_result["result"]["sampled_linf_location"]
phi_linf_value = residual_result["result"]["signed_value_at_sampled_linf"]
normalized_phi_linf = phi_linf * l2_scale_upper

alpha = 0.56
ultracontractivity_t2 = 0.001
ultracontractivity_cells_core = (16, 16)
ultracontractivity_cells_wing = (8, 8)
times = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
         2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0,
         10.0, 20.0, 30.0, 40.0, 50.0, 100.0, Inf]

tail_rate = lambda1_upper - lambda2_lower * (1.0 - alpha)
tail_rate < 0.0 || error(
    "Pointwise tail does not decay: lambda1_upper - lambda2_lower * (1 - alpha) = $tail_rate"
)

ultracon_times = times[1:end-1]
ultracon_splits = [
    (alpha * s - ultracontractivity_t2, ultracontractivity_t2)
    for s in ultracon_times
]

ultracon_constants = [
    total_constant(
        potential,
        t1,
        t2;
        cells_core=ultracontractivity_cells_core,
        cells_wing=ultracontractivity_cells_wing,
    )
    for (t1, t2) in ultracon_splits
]
ultracon_constant_lo = inf.(ultracon_constants)
ultracon_constant_hi = sup.(ultracon_constants)

tail_interval_integrals = [
    exp_integral(times[i], times[i + 1], tail_rate)
    for i in 1:(length(times) - 1)
]
tail_integrals = [
    sum(ultracon_constant_hi[i:end] .* tail_interval_integrals[i:end])
    for i in eachindex(ultracon_times)
]

head_integrals = [
    (exp(s1 * lambda1_upper) - 1.0) / lambda1_lower
    for s1 in ultracon_times
]

head_coefficient = normalized_residual_linf + lambda_error * normalized_phi_linf
tail_coefficient = normalized_residual_l2_bound + lambda_error
head_terms = head_coefficient .* head_integrals
tail_terms = tail_coefficient .* tail_integrals
pointwise_bounds = head_terms .+ tail_terms
selected_index = firstindex(ultracon_times)

mkpath(result_dir)
open(result_path, "w") do io
    TOML.print(io, Dict(
        "step" => "5-pointwise-bounds",
        "inputs" => Dict(
            "fem_eigenvalues" => project_relative(fem_result_path),
            "fit_result" => project_relative(fit_result_path),
            "residual_result" => project_relative(residual_result_path),
            "upper_bound_result" => project_relative(upper_result_path),
            "lower_bound_result" => project_relative(lower_result_path),
            "potential_checkpoint" => project_relative(potential_checkpoint),
            "fit_checkpoint" => project_relative(fit_checkpoint),
        ),
        "parameters" => Dict(
            "alpha" => alpha,
            "candidate_s1_values" => ultracon_times,
            "ultracontractivity_t2" => ultracontractivity_t2,
            "ultracontractivity_cells_core" => collect(ultracontractivity_cells_core),
            "ultracontractivity_cells_wing" => collect(ultracontractivity_cells_wing),
            "phi_linf_sampler" => residual_result["parameters"]["residual_sampler"],
        ),
        "eigenvalues" => Dict(
            "lambda_star" => lambda_star,
            "lambda1_lower" => lambda1_lower,
            "lambda1_upper" => lambda1_upper,
            "lambda2_lower" => lambda2_lower,
            "lambda_error" => lambda_error,
            "tail_rate" => tail_rate,
        ),
        "normalization" => Dict(
            "l2_norm" => interval_table(l2_norm_lower, l2_norm_upper),
            "l2_scale" => interval_table(l2_scale_lower, l2_scale_upper),
        ),
        "residual" => Dict(
            "linf" => residual_linf,
            "normalized_linf_bound" => normalized_residual_linf,
            "normalized_l2_bound" => normalized_residual_l2_bound,
        ),
        "phi_star" => Dict(
            "sampled_linf" => phi_linf,
            "normalized_sampled_linf_bound" => normalized_phi_linf,
            "sampled_linf_location" => Dict(
                "x" => phi_linf_location["x"],
                "y" => phi_linf_location["y"],
            ),
            "signed_value_at_sampled_linf" => phi_linf_value,
        ),
        "ultracontractivity" => Dict(
            "times" => ultracon_times,
            "splits_t1" => first.(ultracon_splits),
            "splits_t2" => last.(ultracon_splits),
            "constant_lo" => ultracon_constant_lo,
            "constant_hi" => ultracon_constant_hi,
            "tail_interval_integrals" => tail_interval_integrals,
            "tail_integrals_from_s1" => tail_integrals,
        ),
        "bound_terms" => Dict(
            "head_coefficient" => head_coefficient,
            "tail_coefficient" => tail_coefficient,
            "head_integrals" => head_integrals,
            "head_terms" => head_terms,
            "tail_terms" => tail_terms,
            "candidate_bounds" => pointwise_bounds,
        ),
        "result" => Dict(
            "s1" => ultracon_times[selected_index],
            "head_term" => head_terms[selected_index],
            "tail_term" => tail_terms[selected_index],
            "pointwise_linf_bound" => pointwise_bounds[selected_index],
        ),
    ); sorted=true)
end

println("Pointwise bound saved to ", result_path)
println("s1 = ", ultracon_times[selected_index])
println("Pointwise L∞ bound <= ", pointwise_bounds[selected_index])
