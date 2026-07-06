using Revise
using TOML

includet("../solver/solver.jl")
includet("../solver/eigenfunction_io.jl")
includet("../solver/eigenfunction_visualization.jl")
includet("../potentials/lse_potential.jl")

using .LSEPotentials

project_root = normpath(joinpath(@__DIR__, "..", ".."))
fit_checkpoint = joinpath(project_root, "checkpoints", "first_eigenfunction.chk")
result_dir = joinpath(project_root, "results")
result_path = joinpath(result_dir, "2b-evaluate-mps-candidate.toml")

project_relative(path) = isabspath(path) ? relpath(normpath(path), project_root) : path

fit = load_fitted_eigenfunction(fit_checkpoint)
λ = fit.λ
coefficients = fit.coefficients

potential_checkpoint = fit.metadata["potential_checkpoint"]
potential = load_lse_potential(potential_checkpoint)
V, gradV = potential_functions(potential)

d = fit.d
diam_x = fit.diam_x
diam_y = fit.diam_y
sampler = GridSampler(1024, 128)
n_modes = fit.n_modes

geometry = make_geometry(d, diam_x, diam_y, V, gradV, sampler)

plot_u_edge_profile(geometry, coefficients, n_modes, λ)

residual_grid, xs, ys = boundary_residual(geometry, coefficients, λ, n_modes, sampler)
abs_residual_grid = abs.(residual_grid)
max_index = argmax(abs_residual_grid)
residual_inf = abs_residual_grid[max_index]
max_x = xs[max_index[1]]
max_y = ys[max_index[2]]
max_residual = residual_grid[max_index]
println("Infinity norm of fine residual: ", residual_inf)

u_grid = [u(fit, x, y, 0.0) for x in xs, y in ys]
abs_u_grid = abs.(u_grid)
u_max_index = argmax(abs_u_grid)
u_linf = abs_u_grid[u_max_index]
u_max_x = xs[u_max_index[1]]
u_max_y = ys[u_max_index[2]]
u_max_value = u_grid[u_max_index]
println("Sampled infinity norm of fitted eigenfunction: ", u_linf)

residual_plot = Plots.heatmap(
    xs,
    ys,
    abs_residual_grid';
    title="Boundary Residual",
    xlabel="x",
    ylabel="y",
    right_margin=5Plots.mm,
)
display(residual_plot)

mkpath(result_dir)
open(result_path, "w") do io
    TOML.print(io, Dict(
        "step" => "2b-evaluate-mps-candidate",
        "inputs" => Dict(
            "fit_checkpoint" => project_relative(fit_checkpoint),
            "potential_checkpoint" => project_relative(potential_checkpoint),
        ),
        "parameters" => Dict(
            "d" => d,
            "diam_x" => diam_x,
            "diam_y" => diam_y,
            "lambda" => λ,
            "n_modes" => collect(n_modes),
            "residual_sampler" => [1024, 128],
        ),
        "result" => Dict(
            "high_resolution_residual_inf" => residual_inf,
            "max_abs_residual_location" => Dict(
                "x" => max_x,
                "y" => max_y,
            ),
            "signed_residual_at_max" => max_residual,
            "sampled_linf" => u_linf,
            "sampled_linf_location" => Dict(
                "x" => u_max_x,
                "y" => u_max_y,
            ),
            "signed_value_at_sampled_linf" => u_max_value,
        ),
    ); sorted=true)
end
println("Result certificate saved to ", result_path)
