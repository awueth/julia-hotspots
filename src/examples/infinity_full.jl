using Revise

includet("../solver.jl")
includet("../potential_generator.jl")

using .PotentialGenerator


ε = 0.1
pot = generate_potential()
d = Inf
diam_x = 4.0 * 2.0 * pot.data.Lx
diam_y = 2.0 * pot.data.Ly
n_boundary_points = 256^2
n_modes = (128, 32)
λ = 2.1375

function V(x, y)
    return ε * PotentialGenerator.V_extended(pot, x, y)
end

function gradV(x, y)
    return ε * PotentialGenerator.∇V_extended(pot, x, y)
end

geometry = make_geometry(d, diam_x, diam_y, V, gradV, n_boundary_points)

coefficients = solve_dense(geometry, n_modes, λ)

#plot_u(geometry, coefficients, n_modes, λ)
plot_u_boundary(geometry, coefficients, n_modes, λ)
plot_u_edge_profile(geometry, coefficients, n_modes, λ; r=:boundary)
plot_u_edge_profile(geometry, coefficients, n_modes, λ; r=:interior)

#find_eigenvalue_dense(geometry, (128, 16), 2.0, 2.2)
#optimize_eigenvalue(geometry, (128, 32), (2.0, 2.2))
"""
For d = 1e9
Optimization Successful: true
Optimal λ: 2.132804191873282
Minimum Loss: 0.3071759118878363
Iterations: 10
(2.132804191873282, 0.3071759118878363)
"""

compute_infinity_norm(geometry, coefficients, n_modes, λ)