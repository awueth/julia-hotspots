include(joinpath(@__DIR__, "..", "ultracontractivity", "constants.jl"))

using TOML
using .UltracontractivityConstants: first_constant, second_constant
using .LSEPotentials: load_lse_potential
using IntervalArithmetic

project_root = normpath(joinpath(@__DIR__, "..", ".."))
checkpoint_path = joinpath(project_root, "checkpoints", "lse_global_potential.chk")
fem_result_path = joinpath(project_root, "results", "1-fem-eigenvalues.toml")
upper_bound_result_path = joinpath(project_root, "results", "3-eigenvalues-upper-bounds.toml")
result_dir = joinpath(project_root, "results")
result_path = joinpath(result_dir, "4-eigenvalues-lower-bounds.toml")

project_relative(path) = isabspath(path) ? relpath(normpath(path), project_root) : path
interval_table(x) = Dict("lo" => inf(x), "hi" => sup(x))

potential = load_lse_potential(checkpoint_path)
wing_mass = interval(potential.wing_mass.lo, potential.wing_mass.hi)
fem_result = TOML.parsefile(fem_result_path)
upper_bound_result = TOML.parsefile(upper_bound_result_path)

# Compute the core correction factor s.t. λ₁ ≥ λ₁^core * factor
begin
    eigenvalue_upper_bound = interval(upper_bound_result["result"]["lambda1_upper"])
    t1 = 0.9
    t2 = 0.1
    t = interval(t1) + interval(t2)
    δ = 0.01
    second_constant_ny_wing = 256

    first_constant_result = first_constant(potential; t=t1, cells_core=(32, 32), cells_wing=(8, 8))
    second_constant_result = second_constant(potential; t1=t1, t2=t2, delta=δ, ny_wing=second_constant_ny_wing)
    C1 = first_constant_result.C1
    C2 = second_constant_result.C2
    C = C1 * C2

    core_correction_factor = interval(1.0) - exp(interval(2.0) * eigenvalue_upper_bound * t) * C^2 * wing_mass
end


# λ₁^core is lower bounded by λ₁^fem / (1 + λ₁^fem * Cₕ^2), 
# where Cₕ = max_K exp(osc_K(V)) 0.1893 h_K
# where h_K = diam K for K a simplex of the FEM mesh, and osc_K(V) = sup_K V - inf_K V.
begin
    positive_tol = 1e-8
    eigenvalue_fem = first(λ for λ in fem_result["result"]["eigenvalues"] if λ > positive_tol)
    h_K = fem_result["result"]["h_K"]
    osc_K = fem_result["result"]["max_oscillation"]
    C_h = exp(osc_K) * 0.1893 * h_K
    λ₁_core_lower_bound = eigenvalue_fem / (1.0 + eigenvalue_fem * C_h^2)
end

eigenvalue_lower_bound = inf(λ₁_core_lower_bound * core_correction_factor)
println("Lower bound on the first nonzero Neumann eigenvalue λ₁ of the weighted operator L = -Δ + ∇V·∇: ", eigenvalue_lower_bound)

mkpath(result_dir)
open(result_path, "w") do io
    TOML.print(io, Dict(
        "step" => "4-eigenvalues-lower-bounds",
        "inputs" => Dict(
            "potential_checkpoint" => project_relative(checkpoint_path),
            "fem_result" => project_relative(fem_result_path),
            "upper_bound_result" => project_relative(upper_bound_result_path),
        ),
        "parameters" => Dict(
            "eigenvalue_upper_bound" => interval_table(eigenvalue_upper_bound),
            "t1" => t1,
            "t2" => t2,
            "delta" => δ,
            "first_constant_cells_core" => [32, 32],
            "first_constant_cells_wing" => [8, 8],
            "second_constant_ny_wing" => second_constant_ny_wing,
            "fem_positive_tol" => positive_tol,
        ),
        "ultracontractivity" => Dict(
            "C1" => interval_table(C1),
            "C2" => interval_table(C2),
            "C2_main" => interval_table(second_constant_result.main),
            "C2_error" => interval_table(second_constant_result.error),
            "C2_wing_steepness" => second_constant_result.Lambda,
            "C" => interval_table(C),
        ),
        "fem" => Dict(
            "eigenvalue" => eigenvalue_fem,
            "osc" => osc_K,
            "C_h" => C_h,
            "h_K" => h_K,
        ),
        "result" => Dict(
            "core_correction_factor" => interval_table(core_correction_factor),
            "core_eigenvalue_lower_bound" => λ₁_core_lower_bound,
            "full_eigenvalue_lower_bound" => eigenvalue_lower_bound,
        ),
    ); sorted=true)
end
println("Result certificate saved to ", result_path)
