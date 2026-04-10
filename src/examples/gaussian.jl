include("../solver.jl")

function V(x, y)
    return 0.5 * (x^2 + y^2)
end

function gradV(x, y)
    return (x, y)
end

d = Inf
diam_x = 20.0
diam_y = 20.0
n_boundary_points = 1024
n_modes = 16
λ = 1.0

geometry = make_geometry(d, diam_x, diam_y, V, gradV, n_boundary_points)

coefficients = solve(geometry, n_modes, λ)

#plot_u(geometry, coefficients, λ)