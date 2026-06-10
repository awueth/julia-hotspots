using Revise

includet("../solver/solver.jl")
includet("../potentials/potential_interface.jl")
includet("../solver/eigenfunction_io.jl")
includet("../solver/eigenfunction_visualization.jl")
includet("../solver/barrier_solving.jl")

using .PotentialInterface

fit = load_fitted_eigenfunction(joinpath(@__DIR__, "..", "..", "checkpoints", "fitted_eigenfunction.chk"))

core = load_lse_core_potential(checkpoint_path=joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk"); Ly=1.0)
wing = HandmadeWingPotential(core.Lx; scale=1e6)
pot = SmoothMaxPotential(core, wing; smooth_max_strength=1e-4)
V, gradV = potential_functions(pot; scale=10.0)

sampler = GridSampler(128, 128)

geometry = make_geometry(fit.d, fit.diam_x, fit.diam_y, V, gradV, sampler)

neumann_vals, xs, ys = boundary_residual(geometry, fit.coefficients, fit.λ, fit.n_modes, sampler)

barrier_modes = (16, 16)

barrier_coefficients = solve_barrier(geometry, barrier_modes, 0.0, abs.(vec(neumann_vals)))

plot_u_boundary(geometry, barrier_coefficients, barrier_modes, 0.0)

barrier_neumann_vals, xs, ys = boundary_residual(geometry, barrier_coefficients, 0.0, barrier_modes, sampler)

failure_points = findall(abs.(neumann_vals) - barrier_neumann_vals .> 0.0)
failure_rate = length(failure_points) / length(neumann_vals)
