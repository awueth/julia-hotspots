using Revise
using TOML

includet("../solver/solver.jl")
includet("../solver/eigenfunction_io.jl")
includet("../solver/eigenfunction_visualization.jl")
includet("../potentials/lse_potential.jl")

using .LSEPotentials

project_root = normpath(joinpath(@__DIR__, "..", ".."))
potential_checkpoint = joinpath(project_root, "checkpoints", "lse_global_potential.chk")
fit_checkpoint = joinpath(project_root, "checkpoints", "first_eigenfunction.chk")
result_dir = joinpath(project_root, "results")
result_path = joinpath(result_dir, "2a-fit-mps-candidate.toml")

potential = load_lse_potential(potential_checkpoint)
V, gradV = potential_functions(potential)

d = Inf
diam_x = 4.0 * pi
diam_y = 2.0
sampler = GridSampler(512, 32)
n_modes = (32, 8)

geometry = make_geometry(d, diam_x, diam_y, V, gradV, sampler)

# solver = QRSolver(geometry, n_modes, λ, FibonacciSampler(512))
solver = DenseSolver()

λ = 3.80155720841065
# λ, _ = optimize_eigenvalue(geometry, n_modes, (1.0, 2.0), solver)
coefficients, residual = solve(geometry, n_modes, λ, solver)

residual_inf = maximum(abs.(residual))
println("Infinity norm of boundary residual: ", residual_inf)

fit = FittedEigenfunction(
    coefficients,
    λ,
    n_modes,
    d,
    diam_x,
    diam_y;
    metadata=Dict(
        "potential_checkpoint" => potential_checkpoint,
        "n_boundary_points" => n_modes,
    )
)

save_fitted_eigenfunction(fit_checkpoint, fit)

mkpath(result_dir)
open(result_path, "w") do io
    TOML.print(io, Dict(
        "step" => "2a-fit-mps-candidate",
        "inputs" => Dict(
            "potential_checkpoint" => relpath(potential_checkpoint, project_root),
        ),
        "outputs" => Dict(
            "fit_checkpoint" => relpath(fit_checkpoint, project_root),
        ),
        "parameters" => Dict(
            "d" => d,
            "diam_x" => diam_x,
            "diam_y" => diam_y,
            "lambda" => λ,
            "n_modes" => collect(n_modes),
            "sampler" => [512, 32],
            "solver" => string(typeof(solver)),
        ),
        "diagnostics" => Dict(
            "boundary_residual_inf" => residual_inf,
        ),
    ); sorted=true)
end
println("Result certificate saved to ", result_path)
