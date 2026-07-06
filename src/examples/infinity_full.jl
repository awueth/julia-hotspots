using Revise

includet("../solver/solver.jl")
includet("../solver/eigenfunction_visualization.jl")
includet("../potentials/lse_potential.jl")

using .LSEPotentials


const DEFAULT_WING_LENGTH = 1.5 * pi
const DEFAULT_SMOOTH_MAX_STRENGTH = 1e-4
# const DEFAULT_LSE_WING_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_WING_SCALE = 1e6

potential = load_lse_potential(joinpath(@__DIR__, "..", "..", "checkpoints", "lse_global_potential.chk"))

d = Inf
diam_x = 2.0 * (0.5 * pi + DEFAULT_WING_LENGTH)
diam_y = 2.0
# sampler = FibonacciSampler(256 * 64)
sampler = GridSampler(800, 64)
n_modes = (400, 32)
λ = 3.80155720841065

V, gradV = potential_functions(potential)

geometry = make_geometry(d, diam_x, diam_y, V, gradV, sampler)

# solver = QRSolver(geometry, n_modes, λ, FibonacciSampler(512))
solver = DenseSolver()

# λ, _ = optimize_eigenvalue(geometry, n_modes, (3.8, 3.9), solver)
coefficients, residual = solve(geometry, n_modes, λ, solver)

# plot_u_boundary(geometry, coefficients, n_modes, λ)
plot_u_edge_profile(geometry, coefficients, n_modes, λ)

println("Infinity norm of boundary residual: ", maximum(abs.(residual)))

residual_fine, xs, ys = boundary_residual(geometry, coefficients, λ, n_modes, GridSampler(1024, 128))
println("Infinity norm of fine residual: ", maximum(abs.(residual_fine)))
residual_plot = Plots.heatmap(
    xs,
    ys,
    abs.(residual_fine)';
    title="Boundary Residual",
    xlabel="x",
    ylabel="y",
    right_margin=5Plots.mm,
)
Plots.scatter!(
    residual_plot,
    geometry.points.x,
    geometry.points.y;
    label="Collocation points",
    markercolor=:white,
    markeralpha=0.35,
    markersize=1.0,
    markerstrokewidth=0,
)
display(residual_plot)