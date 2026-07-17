if !isdefined(@__MODULE__, :Geometry)
    include("solver.jl")
end

import Plots

function r_boundary(geometry::Geometry{T, T}, x::T, y::T) where T<:AbstractFloat
    return one(T) - geometry.V(x, y) / geometry.d
end

function r_boundary(geometry::Geometry{Nothing, T}, x::T, y::T) where T<:AbstractFloat
    return one(T)
end

function plot_u_boundary(
    geometry::Geometry{<:Any, T},
    coefficients::AbstractVector{T},
    n_modes::Tuple{Int,Int},
    λ::T
) where {T <: AbstractFloat}
    res = 128
    xs = LinRange(-geometry.diam_x / T(2), geometry.diam_x / T(2), res)
    ys = LinRange(-geometry.diam_y / T(2), geometry.diam_y / T(2), res)

    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    u_boundary = [
        u(geometry, coefficients, λx, λy, λr, x, y, r_boundary(geometry, x, y))
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
    geometry::Geometry{<:Any, T},
    coefficients::AbstractVector{T},
    n_modes::Tuple{Int,Int},
    λ::T
) where {T <: AbstractFloat}
    res = 200
    x_boundary = geometry.diam_x / T(2)
    xs = LinRange(-geometry.diam_x / T(2), geometry.diam_x / T(2), res)
    ys = LinRange(-geometry.diam_y / T(2), geometry.diam_y / T(2), res)
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)

    x0 = xs[argmax([
        u(geometry, coefficients, λx, λy, λr, x, zero(T), zero(T))
        for x in xs
    ])]

    profile(x, r_at) = [
        u(geometry, coefficients, λx, λy, λr, x, y, r_at(x, y))
        for y in ys
    ]

    boundary_outer = profile(x_boundary, (x, y) -> r_boundary(geometry, x, y))
    interior_outer = profile(x0, (x, y) -> r_boundary(geometry, x, y))
    boundary_inner = profile(x_boundary, (_, _) -> zero(T))
    interior_inner = profile(x0, (_, _) -> zero(T))
    interface_outer = profile(T(0.5*pi), (x, y) -> r_boundary(geometry, x, y))

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

    plt = Plots.plot(
        outer_plot,
        inner_plot;
        layout=(2, 1),
        size=(900, 700),
        plot_title="Profiles at x = $(round(x_boundary, digits=2)) and x0 = $(round(x0, digits=2))"
    )

    display(plt)
end
