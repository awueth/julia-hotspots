using Revise

includet("../solver.jl")
includet("../eigenfunction_visualization.jl")
includet("../potential_interface.jl")

using .PotentialInterface


const DEFAULT_EPSILON = 0.1
const DEFAULT_WING_LENGTH = 5.0
const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_SMOOTH_MAX_STRENGTH = 10.0
const DEFAULT_WING_SCALE = 5e6

ε = DEFAULT_EPSILON
core = load_lse_core_potential(checkpoint_path=DEFAULT_LSE_CORE_CHECKPOINT_PATH)
domain = potential_domain(core)
wing = HandmadeWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE)
pot = SmoothMaxPotential(core, wing; smooth_max_strength=DEFAULT_SMOOTH_MAX_STRENGTH)
domain = potential_domain(pot)
d = Inf
diam_x = 2.0 * (domain.Lx + DEFAULT_WING_LENGTH)
diam_y = 2.0 * domain.Ly
n_boundary_points = (448, 64)
n_modes = (320, 32)
λ = 1.062061862161251 #1.0652958611899328

V, gradV = potential_functions(pot; scale=ε)

geometry = make_geometry(d, diam_x, diam_y, V, gradV, n_boundary_points)

coefficients, residual = solve_iterative(geometry, n_modes, λ)

#plot_u(geometry, coefficients, n_modes, λ)
plot_u_boundary(geometry, coefficients, n_modes, λ)
plot_u_edge_profile(geometry, coefficients, n_modes, λ; r=:boundary)
plot_u_edge_profile(geometry, coefficients, n_modes, λ; r=:interior)

# optimize_eigenvalue_iterative(geometry, n_modes, (1.02, 1.1))

println("Infinity norm of boundary residual: ", maximum(abs.(residual)))

# On euler with: 256 x modes and 512^2 boundary points, we obtain:

# command: julia --project=. -t auto src/examples/infinity_full.jl

# Running optimization...
# Optimization finished. Penalized objective: 5745.31659074317
# Sup norm of V₀: 356.4163407864109
# Unpenalized normalized min Hessian eigenvalue: -16.119670928660884
# Loss: 0.2830567629741757
# 0.005259586590494071
