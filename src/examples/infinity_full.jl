using Revise

includet("../solver/solver.jl")
includet("../solver/eigenfunction_visualization.jl")
includet("../potentials/potential_interface.jl")

using .PotentialInterface


const DEFAULT_EPSILON = 10.0
const DEFAULT_WING_LENGTH = 1.5 * pi
const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_SMOOTH_MAX_STRENGTH = 1e-4
# const DEFAULT_LSE_WING_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_WING_SCALE = 1e6

ε = DEFAULT_EPSILON
core = load_lse_core_potential(checkpoint_path=DEFAULT_LSE_CORE_CHECKPOINT_PATH; Ly=1.0)
domain = potential_domain(core)
wing = HandmadeWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE)
#wing = NonConvexWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE, anchor=core_value(core, domain.Lx, domain.Ly))
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
# sampler = FibonacciSampler(256 * 64)
sampler = GridSampler(512, 128)
n_modes = (256, 64)
λ = 3.87081387296969

V, gradV = potential_functions(pot; scale=ε)

geometry = make_geometry(d, diam_x, diam_y, V, gradV, sampler)

# solver = QRSolver(geometry, n_modes, λ, FibonacciSampler(512))
solver = DenseSolver()

# λ, _ = optimize_eigenvalue(geometry, n_modes, (3.85, 4.0), solver)
coefficients, residual = solve(geometry, n_modes, λ, solver)

#plot_u_boundary(geometry, coefficients, n_modes, λ)
plot_u_edge_profile(geometry, coefficients, n_modes, λ)

println("Infinity norm of boundary residual: ", maximum(abs.(residual)))

residual_fine, xs, ys = boundary_residual(geometry, coefficients, λ, n_modes, (1024, 128))
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
