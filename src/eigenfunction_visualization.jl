if !isdefined(@__MODULE__, :Geometry)
    include("solver.jl")
end

import Plots

function plot_u_boundary(
    geometry::Geometry,
    coefficients::AbstractVector{Float64},
    n_modes::Tuple{Int,Int},
    λ::Float64
)
    res = 128
    xs = LinRange(-geometry.diam_x / 2, geometry.diam_x / 2, res)
    ys = LinRange(-geometry.diam_y / 2, geometry.diam_y / 2, res)

    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    u_boundary = [
        u(geometry, coefficients, λx, λy, λr, x, y, 1.0 - geometry.V(x, y) / geometry.d)
        for y in ys, x in xs
    ]

    plt = Plots.surface(
        xs,
        ys,
        u_boundary;
        xlabel="x",
        ylabel="y",
        zlabel="u(x, y, r_boundary)",
        title="Solution u on the boundary surface",
        color=:coolwarm
    )

    display(plt)
end

function plot_u_edge_profile(
    geometry::Geometry,
    coefficients::AbstractVector{Float64},
    n_modes::Tuple{Int,Int},
    λ::Float64
)
    res = 200
    x_boundary = geometry.diam_x / 2
    xs = LinRange(-geometry.diam_x / 2, geometry.diam_x / 2, res)
    ys = LinRange(-geometry.diam_y / 2, geometry.diam_y / 2, res)
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)

    x0 = xs[argmax([
        u(geometry, coefficients, λx, λy, λr, x, 0.0, 0.0)
        for x in xs
    ])]

    r_boundary(x, y) = 1.0 - geometry.V(x, y) / geometry.d
    profile(x, r_at) = [
        u(geometry, coefficients, λx, λy, λr, x, y, r_at(x, y))
        for y in ys
    ]

    boundary_outer = profile(x_boundary, r_boundary)
    interior_outer = profile(x0, r_boundary)
    boundary_inner = profile(x_boundary, (_, _) -> 0.0)
    interior_inner = profile(x0, (_, _) -> 0.0)
    interface_outer = profile(0.5*pi, r_boundary)
    inner_center_difference =
        u(geometry, coefficients, λx, λy, λr, x0, 0.0, 0.0) -
        u(geometry, coefficients, λx, λy, λr, x_boundary, 0.0, 0.0)

    outer_plot = Plots.plot(
        ys,
        interior_outer;
        color=:red,
        linewidth=2,
        label="Interior Profile",
        xlabel="y",
        ylabel="u(x, y, r_boundary)",
        title="Profiles on the boundary surface"
    )
    Plots.plot!(outer_plot, ys, boundary_outer; color=:blue, linewidth=2, label="Boundary Profile")
    Plots.plot!(outer_plot, ys, interface_outer; color=:green, linewidth=2, label="Interface Profile")

    inner_plot = Plots.plot(
        ys,
        interior_inner;
        color=:red,
        linewidth=2,
        label="Interior Profile",
        xlabel="y",
        ylabel="u(x, y, 0)",
        title="Profiles at r = 0"
    )
    Plots.plot!(inner_plot, ys, boundary_inner; color=:blue, linewidth=2, label="Boundary Profile")
    Plots.annotate!(
        inner_plot,
        -0.5,
        u(geometry, coefficients, λx, λy, λr, x0, 0.0, 0.0),
        Plots.text("Δ(y=0) = $(round(inner_center_difference, sigdigits=6))")
    )

    plt = Plots.plot(
        outer_plot,
        inner_plot;
        layout=(2, 1),
        size=(900, 700),
        plot_title="Profiles at x = $(round(x_boundary, digits=2)) and x0 = $(round(x0, digits=2))"
    )

    display(plt)
end
