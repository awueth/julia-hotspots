using Revise

includet("../solver/solver.jl")
includet("../solver/eigenfunction_visualization.jl")
includet("../potentials/potential_interface.jl")

using .PotentialInterface


const DEFAULT_EPSILON = 10.0
const DEFAULT_WING_LENGTH = 1.5 * pi
const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_SMOOTH_MAX_STRENGTH = 1.0
# const DEFAULT_LSE_WING_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_WING_SCALE = 1e6

ε = DEFAULT_EPSILON
core = load_lse_core_potential(checkpoint_path=DEFAULT_LSE_CORE_CHECKPOINT_PATH; Ly=1.0)
domain = potential_domain(core)
#wing = HandmadeWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE)
wing = NonConvexWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE, anchor=core_value(core, domain.Lx, domain.Ly))
pot = SmoothMaxPotential(core, wing; smooth_max_strength=DEFAULT_SMOOTH_MAX_STRENGTH)
# wing = load_lse_wing_potential(
#     checkpoint_path=DEFAULT_LSE_WING_CHECKPOINT_PATH;
#     Lx=domain.Lx,
#     Ly=domain.Ly,
#     scale=DEFAULT_WING_SCALE,
# )
# pot = join_lse_potentials(core, wing)
domain = potential_domain(pot)
d = Inf
diam_x = 2.0 * (domain.Lx + DEFAULT_WING_LENGTH)
diam_y = 2.0 * domain.Ly
n_boundary_points = (1280, 64)
n_modes = (640, 32)
λ = 3.9297514935298103

V, gradV = potential_functions(pot; scale=ε)

geometry = make_geometry(d, diam_x, diam_y, V, gradV, n_boundary_points)

λ, _ = optimize_eigenvalue(geometry, n_modes, (3.85, 4.0))
coefficients, residual = solve_iterative(geometry, n_modes, λ)

#plot_u_boundary(geometry, coefficients, n_modes, λ)
plot_u_edge_profile(geometry, coefficients, n_modes, λ)

println("Infinity norm of boundary residual: ", maximum(abs.(residual)))

residual_fine, xs, ys = boundary_residual(geometry, coefficients, λ, n_modes, (128, 128))
println("Infinity norm of fine residual: ", maximum(abs.(residual_fine)))
heatmap(xs, ys, abs.(residual_fine)'; title="Boundary Residual", xlabel="x", ylabel="y", right_margin=5Plots.mm)


# res_grid = reshape(residual, length(geometry.points.x), length(geometry.points.y))

# W = (res_grid ./ maximum(abs.(res_grid))) .+ 0.1

# coefficients_weighted, residual_weighted = solve_iterative(geometry, n_modes, λ; weights=W)

# plot_u_edge_profile(geometry, coefficients_weighted, n_modes, λ)

# residual_fine, xs, ys = boundary_residual(geometry, coefficients, λ, n_modes, (512, 64))
# println("Infinity norm of fine residual: ", maximum(abs.(residual_fine)))
# heatmap(xs, ys, abs.(residual_fine)'; title="Boundary Residual", xlabel="x", ylabel="y")

# optimize_eigenvalue_iterative(geometry, n_modes, (1.02, 1.1); weights=W)

