if !isdefined(@__MODULE__, :Geometry)
    include("solver.jl")
end

import GLMakie as Makie

function plot_u(geometry::Geometry, coefficients::AbstractVector{Float64}, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    res = 64
    xs = LinRange(-geometry.diam_x / 2, geometry.diam_x / 2, res)
    ys = LinRange(-geometry.diam_y / 2, geometry.diam_y / 2, res)
    rs = LinRange(0.95, 1.0, res)

    r_boundary = [1.0 - geometry.V(x, y) / geometry.d for x in xs, y in ys]
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)

    vals = Array{Float64}(undef, length(xs), length(ys), length(rs))
    for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys), (ir, r) in enumerate(rs)
        vals[ix, iy, ir] = r <= r_boundary[ix, iy] ? u(geometry, coefficients, λx, λy, λr, x, y, r) : NaN
    end

    valid_vals = filter(!isnan, vals)
    vmin, vmax = extrema(valid_vals)
    vdiff = vmax == vmin ? 1.0 : vmax - vmin
    dummy_val = vmin - vdiff
    clean_vals = replace(vals, NaN => dummy_val)

    base_cmap = Makie.to_colormap(:coolwarm)
    transparent_half = fill(Makie.RGBAf(0.0, 0.0, 0.0, 0.0), length(base_cmap))
    custom_cmap = vcat(transparent_half, base_cmap)

    fig = Makie.Figure()
    ax = Makie.Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="r")
    Makie.volume!(
        ax,
        (xs[1], xs[end]),
        (ys[1], ys[end]),
        (rs[1], rs[end]),
        clean_vals;
        algorithm=:absorption,
        colormap=custom_cmap,
        colorrange=(dummy_val, vmax)
    )
    Makie.Colorbar(fig[1, 2], colormap=:coolwarm, limits=(vmin, vmax))

    display(fig)
end

function plot_u_boundary(geometry::Geometry, coefficients::AbstractVector{Float64}, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    res = 128
    xs = LinRange(-geometry.diam_x / 2, geometry.diam_x / 2, res)
    ys = LinRange(-geometry.diam_y / 2, geometry.diam_y / 2, res)
    r_boundary = [1.0 - geometry.V(x, y) / geometry.d for x in xs, y in ys]

    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    u_boundary = [
        u(geometry, coefficients, λx, λy, λr, x, y, r_boundary[ix, iy])
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
    ]

    vmin, vmax = extrema(u_boundary)

    fig = Makie.Figure()
    ax = Makie.Axis3(
        fig[1, 1],
        xlabel="x",
        ylabel="y",
        zlabel="u(x, y, r_boundary)",
        title="Solution u on the boundary surface"
    )
    plt = Makie.surface!(
        ax,
        xs,
        ys,
        u_boundary;
        color=u_boundary,
        colormap=:coolwarm,
        colorrange=(vmin, vmax)
    )
    Makie.Colorbar(fig[1, 2], plt, label="u value")

    display(fig)
end

function plot_u_edge_profile(
    geometry::Geometry,
    coefficients::AbstractVector{Float64},
    n_modes::Union{Int,Tuple{Int,Int}},
    λ::Float64;
    r=:boundary
)
    res = 200
    x_boundary = geometry.diam_x / 2
    x_interior = geometry.diam_x / 2 - 2.9
    ys = LinRange(-geometry.diam_y / 2, geometry.diam_y / 2, res)

    if r == :boundary
        rs_boundary = [1.0 - geometry.V(x_boundary, y) / geometry.d for y in ys]
        rs_interior = [1.0 - geometry.V(x_interior, y) / geometry.d for y in ys]
    else
        rs_boundary = fill(0.0, length(ys))
        rs_interior = fill(0.0, length(ys))
    end

    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    u_vals = [
        u(geometry, coefficients, λx, λy, λr, x_boundary, y, rs_boundary[iy])
        for (iy, y) in enumerate(ys)
    ]
    u_vals_interior = [
        u(geometry, coefficients, λx, λy, λr, x_interior, y, rs_interior[iy])
        for (iy, y) in enumerate(ys)
    ]

    fig = Makie.Figure()
    ax = Makie.Axis(
        fig[1, 1],
        xlabel="y",
        ylabel="u(x_boundary, y, r_boundary)",
        title="Profile at x = $(round(x_boundary, digits=2))"
    )
    Makie.lines!(ax, ys, u_vals_interior, color=:red, linewidth=2, label="Interior Profile")
    Makie.lines!(ax, ys, u_vals, color=:blue, linewidth=2, label="Boundary Profile")
    Makie.axislegend(ax)

    display(fig)
end
