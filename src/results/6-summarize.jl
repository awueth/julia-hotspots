using TOML

project_root = normpath(joinpath(@__DIR__, "..", ".."))
result_dir = joinpath(project_root, "results")
summary_path = joinpath(result_dir, "summary.toml")

result_file(name) = joinpath(result_dir, name)
project_relative(path) = relpath(normpath(path), project_root)

fem_path = result_file("1-fem-eigenvalues.toml")
fit_path = result_file("2a-fit-mps-candidate.toml")
evaluation_path = result_file("2b-evaluate-mps-candidate.toml")
upper_path = result_file("3-eigenvalues-upper-bounds.toml")
lower_path = result_file("4-eigenvalues-lower-bounds.toml")
pointwise_path = result_file("5-pointwise-bounds.toml")

fem = TOML.parsefile(fem_path)
fit = TOML.parsefile(fit_path)
evaluation = TOML.parsefile(evaluation_path)
upper = TOML.parsefile(upper_path)
lower = TOML.parsefile(lower_path)
pointwise = TOML.parsefile(pointwise_path)

summary = Dict(
    "inputs" => Dict(
        "fem_eigenvalues" => project_relative(fem_path),
        "fit_mps_candidate" => project_relative(fit_path),
        "evaluate_mps_candidate" => project_relative(evaluation_path),
        "eigenvalue_upper_bound" => project_relative(upper_path),
        "eigenvalue_lower_bound" => project_relative(lower_path),
        "pointwise_bound" => project_relative(pointwise_path),
        "potential_checkpoint" => fit["inputs"]["potential_checkpoint"],
        "fit_checkpoint" => fit["outputs"]["fit_checkpoint"],
    ),
    "mps_candidate" => Dict(
        "lambda" => fit["parameters"]["lambda"],
        "n_modes" => fit["parameters"]["n_modes"],
        "fit_residual_inf" => fit["diagnostics"]["boundary_residual_inf"],
        "high_resolution_residual_inf" => evaluation["result"]["high_resolution_residual_inf"],
        "sampled_linf" => evaluation["result"]["sampled_linf"],
        "l2_norm" => upper["result"]["l2_norm"],
    ),
    "fem" => Dict(
        "partition" => fem["parameters"]["partition"],
        "h_K" => fem["result"]["h_K"],
        "max_oscillation" => fem["result"]["max_oscillation"],
    ),
    "eigenvalue_bounds" => Dict(
        "lambda1_lower" => lower["result"]["full_eigenvalue_lower_bound"],
        "lambda1_upper" => upper["result"]["lambda1_upper"],
        "lambda2_lower" => pointwise["eigenvalues"]["lambda2_lower"],
        "core_lambda1_lower" => lower["result"]["core_eigenvalue_lower_bound"],
        "core_correction_factor" => lower["result"]["core_correction_factor"],
    ),
    "ultracontractivity_lower_bound_step" => Dict(
        "C1" => lower["ultracontractivity"]["C1"],
        "C2" => lower["ultracontractivity"]["C2"],
        "C" => lower["ultracontractivity"]["C"],
        "C2_wing_steepness" => lower["ultracontractivity"]["C2_wing_steepness"],
    ),
    "pointwise_limit" => Dict(
        "s1" => pointwise["result"]["s1"],
        "alpha" => pointwise["parameters"]["alpha"],
        "tail_rate" => pointwise["eigenvalues"]["tail_rate"],
        "lambda_error" => pointwise["eigenvalues"]["lambda_error"],
        "normalized_residual_bound" => pointwise["residual"]["normalized_linf_bound"],
        "normalized_phi_linf_bound" => pointwise["phi_star"]["normalized_sampled_linf_bound"],
        "head_term" => pointwise["result"]["head_term"],
        "tail_term" => pointwise["result"]["tail_term"],
        "pointwise_linf_bound" => pointwise["result"]["pointwise_linf_bound"],
    ),
)

mkpath(result_dir)
open(summary_path, "w") do io
    TOML.print(io, summary; sorted=true)
end

println("Summary saved to ", summary_path)
