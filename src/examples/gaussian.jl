using Revise
includet("../solver.jl")
includet("../eigenfunction_visualization.jl")
includet("../barrier_solving.jl")

function V(x, y)
    return 0.5 * (x^2 + y^2)
end

function gradV(x, y)
    return (x, y)
end

d = Inf
diam_x = 20.0
diam_y = 10.0
n_boundary_points = 2 * 64^2
n_modes = (64, 64)
λ = 0.9265939085423157 #0.7929095457323481

geometry = make_geometry(d, diam_x, diam_y, V, gradV, n_boundary_points)

coefficients = solve_dense(geometry, n_modes, λ)
residual = compute_residual(geometry, coefficients, n_modes, λ)

#plot_u(geometry, coefficients, n_modes, λ)
plot_u_boundary(geometry, coefficients, n_modes, λ)

barrier_coefficients = solve_barrier(geometry, n_modes, 0.1, residual.+0.0001)
# barrier_coefficients = 1.1 * solve_dense(geometry, n_modes, λ+0.01)
#plot_u(geometry, barrier_coefficients, n_modes, λ)
plot_u_boundary(geometry, barrier_coefficients, n_modes, λ)

barrier_residual = compute_residual(geometry, barrier_coefficients, n_modes, λ)

xs = range(0, diam_x/2, length=100)
ys = range(0, diam_y/2, length=100)

u_diff(x, y) = u(geometry, barrier_coefficients, λ, n_modes, x, y, 1.0) - u(geometry, coefficients, λ, n_modes, x, y, 1.0)
Plots.surface(xs, ys, u_diff)

optimize_eigenvalue(geometry, (64, 64), (0.6, 1.4))

# for 64 x 64 modes:
# Optimization Successful: true
# Optimal λ: 0.9265939085423157
# Minimum Loss: 0.2730072590057393
# Iterations: 16
# (0.9265939085423157, 0.2730072590057393)
