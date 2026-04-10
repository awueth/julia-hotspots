include("functions/basis.jl")
using KrylovKit
using LinearAlgebra
using GLMakie
using Optim

struct Geometry{F1,F2}
    d::Float64
    diam_x::Float64
    diam_y::Float64
    V::F1
    gradV::F2
    points::@NamedTuple{x::Vector{Float64}, y::Vector{Float64}, r::Vector{Float64}}
    normals::@NamedTuple{x::Vector{Float64}, y::Vector{Float64}, r::Vector{Float64}}
end

function make_geometry(
    d::Float64, 
    diam_x::Float64, 
    diam_y::Float64, 
    V::F1, 
    gradV::F2,
    n_points ::Int
) where {F1,F2}
    n_side = round(Int, sqrt(n_points))
    x_grid = range(0, 0.5 * diam_x, length=n_side)
    y_grid = range(0, 0.5 * diam_y, length=n_side)
    grid = vec(collect(Iterators.product(x_grid, y_grid)))
    xs = [p[1] for p in grid]
    ys = [p[2] for p in grid]
    rs = 1.0 .- V.(xs, ys) ./ d
    points = (x=xs, y=ys, r=rs)

    grads = gradV.(xs, ys)
    nx = [g[1] for g in grads]
    ny = [g[2] for g in grads]
    nr = 4.0
    inv_len = 1 ./ sqrt.(nx .^ 2 + ny .^ 2 .+ nr .^ 2)
    normals = (x=nx .* inv_len, y=ny .* inv_len, r=nr .* inv_len)

    return Geometry(d, diam_x, diam_y, V, gradV, points, normals)
end

function normal_deriv_at_mode(geometry::Geometry, λx::Float64, λy::Float64, λr::Float64)
    x, y, r = geometry.points
    nx, ny, nr = geometry.normals
    d = geometry.d

    axial_vals, (axial_grads_x, axial_grads_y) = axial_basis(λx, λy, x, y)
    radial_vals, radial_grads = ϕ(d, λr, r)

    return (nx .* axial_grads_x .* radial_vals) .+
           (ny .* axial_grads_y .* radial_vals) .+
           (nr .* axial_vals .* radial_grads)
end

mode_counts(n_modes::Int) = (n_modes, n_modes)
mode_counts(n_modes::Tuple{Int,Int}) = n_modes

function get_eigenvalues(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    n_modes_x, n_modes_y = mode_counts(n_modes)
    modes_x = 1:2:(2 * n_modes_x - 1)
    modes_y = 0:2:(2 * n_modes_y - 2)
    λx_ = (modes_x .* (π / geometry.diam_x)) .^ 2
    λy_ = (modes_y .* (π / geometry.diam_y)) .^ 2

    grid = vec(collect(Iterators.product(λx_, λy_)))
    λx = [g[1] for g in grid]
    λy = [g[2] for g in grid]
    λr = λ .- (λx .+ λy)

    return λx, λy, λr
end

function get_matrix(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    total_modes = length(λx)

    M = zeros(Float64, length(geometry.points.x), total_modes)
    for j in 1:total_modes
        M[:, j] = normal_deriv_at_mode(geometry, λx[j], λy[j], λr[j])
    end
    return M
end

function normalize_mode_columns(A::AbstractMatrix{Float64})
    scales = vec(norm.(eachcol(A)))
    scales = max.(scales, eps(Float64))

    A_scaled = copy(A)
    for j in axes(A_scaled, 2)
        A_scaled[:, j] ./= scales[j]
    end

    return A_scaled, scales
end

function u(
    geometry::Geometry, 
    coefficients::AbstractVector{Float64}, 
    λx::AbstractVector{Float64}, 
    λy::AbstractVector{Float64}, 
    λr::AbstractVector{Float64}, 
    x::Float64, 
    y::Float64, 
    r::Float64
)
    val = 0.0
    for i in eachindex(coefficients)
        av, _ = axial_basis(λx[i], λy[i], x, y)
        rv, _ = ϕ(geometry.d, λr[i], r)
        val += coefficients[i] * av * rv
    end

    return val
end

function plot_u(geometry::Geometry, coefficients::AbstractVector{Float64}, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    res = 64
    xs = LinRange(-geometry.diam_x / 2, geometry.diam_x / 2, res)
    ys = LinRange(-geometry.diam_y / 2, geometry.diam_y / 2, res)
    rs = LinRange(0.95, 1.0, res)

    r_boundary = [1.0 - geometry.V(x, y) / geometry.d for x in xs, y in ys]

    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)

    vals = Array{Float64}(undef, length(xs), length(ys), length(rs))
    for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys), (ir, r) in enumerate(rs)
        if r <= r_boundary[ix, iy]
            vals[ix, iy, ir] = u(geometry, coefficients, λx, λy, λr, x, y, r)
        else
            vals[ix, iy, ir] = NaN
        end
    end

    fig = Figure()
    ax = Axis3(fig[1, 1],
        xlabel="x",
        ylabel="y",
        zlabel="r",
        #aspect = (1, 1, 5)
    )

    # --- THE FIX: Custom Split Colormap ---
    valid_vals = filter(!isnan, vals)
    vmin, vmax = extrema(valid_vals)
    vdiff = vmax == vmin ? 1.0 : (vmax - vmin)

    # 1. Map NaNs to a specific dummy value exactdiam_y one full range below vmin
    dummy_val = vmin - vdiff
    clean_vals = replace(vals, NaN => dummy_val)

    # 2. Build a colormap: bottom half is transparent, top half is coolwarm
    base_cmap = Makie.to_colormap(:coolwarm)
    transparent_half = fill(RGBAf(0.0, 0.0, 0.0, 0.0), length(base_cmap))
    custom_cmap = vcat(transparent_half, base_cmap)

    # 3. Set the colorrange to exactdiam_y span from the dummy value to vmax.
    # This mathematicaldiam_y maps `dummy_val` to the transparent half, 
    # and valid values perfectdiam_y into the coolwarm half.
    plt = volume!(ax, (xs[1], xs[end]), (ys[1], ys[end]), (rs[1], rs[end]), clean_vals;
        algorithm=:absorption,
        colormap=custom_cmap,
        colorrange=(dummy_val, vmax)
    )

    # u_boundary = [u(geometry, coefficients, λ, x, y, one(T) - geometry.V(x, y) / geometry.d) for x in xs, y in ys]

    # Keep the surface plot normal, as it respects nan_color properdiam_y
    # GLMakie.surface!(ax, collect(xs), collect(ys), r_boundary;
    #     color = u_boundary,
    #     colormap = :coolwarm,
    #     colorrange = (vmin, vmax),
    # )

    # Decouple the colorbar from the volume object so it ondiam_y shows the valid coolwarm limits
    Colorbar(fig[1, 2], colormap=:coolwarm, limits=(vmin, vmax))

    display(fig)
end

function plot_u_boundary(geometry::Geometry, coefficients::AbstractVector{Float64}, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    res = 128
    xs = LinRange(-geometry.diam_x / 2, geometry.diam_x / 2, res)
    ys = LinRange(-geometry.diam_y / 2, geometry.diam_y / 2, res)

    # 1. Calculate the r-coordinate of the boundary at each (x, y)
    r_boundary = [1.0 - geometry.V(x, y) / geometry.d for x in xs, y in ys]

    # 2. Get eigenvalues
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)

    # 3. Evaluate u at the boundary
    u_boundary = [u(geometry, coefficients, λx, λy, λr, x, y, r_boundary[ix, iy]) 
                  for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)]

    # 4. Visualization: Map Z to u_boundary
    fig = Figure()
    ax = Axis3(fig[1, 1],
        xlabel="x",
        ylabel="y",
        zlabel="u(x, y, r_boundary)",
        title="Solution u on the boundary surface"
    )

    vmin, vmax = extrema(u_boundary)

    # Note: We now use u_boundary as the height (3rd argument)
    plt = GLMakie.surface!(ax, xs, ys, u_boundary;
        color = u_boundary,
        colormap = :coolwarm,
        colorrange = (vmin, vmax),
    )

    Colorbar(fig[1, 2], plt, label="u value")

    display(fig)
end

function plot_u_edge_profile(geometry::Geometry, coefficients::AbstractVector{Float64}, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64; r=:boundary)
    res = 200
    # 1. Fix x at the right boundary
    x_boundary = geometry.diam_x / 2
    x_interior = geometry.diam_x / 2 - 2.0
    
    # 2. Vary y across the full range
    ys = LinRange(-geometry.diam_y / 2, geometry.diam_y / 2, res)
    
    # 3. Calculate r_boundary at every y for the fixed x
    if r == :boundary
        rs_boundary = [1.0 - geometry.V(x_boundary, y) / geometry.d for y in ys]
        rs_interior = [1.0 - geometry.V(x_interior, y) / geometry.d for y in ys]
    else
        rs_boundary = fill(0.00, length(ys))
        rs_interior = fill(0.00, length(ys))
    end

    # 4. Get basis eigenvalues
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)

    # 5. Evaluate u along that line
    u_vals = [u(geometry, coefficients, λx, λy, λr, x_boundary, y, rs_boundary[iy]) 
              for (iy, y) in enumerate(ys)]
    u_vals_interior = [u(geometry, coefficients, λx, λy, λr, x_interior, y, rs_interior[iy]) 
                       for (iy, y) in enumerate(ys)]

    # 6. Plotting
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "y",
        ylabel = "u(x_boundary, y, r_boundary)",
        title = "Profile at x = $(round(x_boundary, digits=2))"
    )

    lines!(ax, ys, u_vals_interior, color = :red, linewidth = 2, label="Interior Profile")
    lines!(ax, ys, u_vals, color = :blue, linewidth = 2, label="Boundary Profile")
    axislegend(ax)
    #GLMakie.ylims!(ax, extrema(u_vals) .+ (-0.001, 0.001))
    
    # Optional: Highlight boundary behavior
    # hlines!(ax, [0], color = :black, linestyle = :dash) 

    display(fig)
end


function find_eigenvalue(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, lower::Float64, upper::Float64)
    n_modes_x, n_modes_y = mode_counts(n_modes)
    c₀ = zeros(Float64, n_modes_x * n_modes_y)
    c₀[1] = 1.0

    λ_vals = collect(range(lower, upper, length=100))
    losses = Vector{Float64}(undef, length(λ_vals))

    for (i, λ) in enumerate(λ_vals)
        A = get_matrix(geometry, n_modes, λ)
        M = A'
        vals, _, _, info = svdsolve(M, c₀, 1, :SR)
        losses[i] = vals[1]
    end

    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel="λ",
        ylabel="Singular value (loss)",
        title="Loss vs Eigenvalue"
    )
    lines!(ax, λ_vals, losses)
    display(fig)
end

function find_eigenvalue_dense(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, lower::Float64, upper::Float64)
    λ_vals = collect(range(lower, upper, length=100))
    losses = Vector{Float64}(undef, length(λ_vals))

    for (i, λ) in enumerate(λ_vals)
        A = get_matrix(geometry, n_modes, λ)
        F = svd(A)
        losses[i] = F.S[end] 
    end

    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel="λ",
        ylabel="Singular value (loss)",
        title="Loss vs Eigenvalue"
    )
    lines!(ax, λ_vals, losses)
    display(fig)
end

function optimize_eigenvalue(geometry::Geometry, n_modes::Tuple{Int,Int}, bounds::Tuple{Float64, Float64})
    lower, upper = bounds

    function objective(λ)
        A = get_matrix(geometry, n_modes, λ)
        return svdvals(A)[end]  
    end

    result = optimize(objective, lower, upper, Brent())
    best_λ = Optim.minimizer(result)
    best_loss = Optim.minimum(result)

    println("Optimization Successful: ", Optim.converged(result))
    println("Optimal λ: ", best_λ)
    println("Minimum Loss: ", best_loss)
    println("Iterations: ", Optim.iterations(result))
    
    return best_λ, best_loss
end


function solve(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    A = get_matrix(geometry, n_modes, λ)
    A_scaled, _ = normalize_mode_columns(A)
    F_scaled = svd(A_scaled)
    M = A'
    c₀ = F_scaled.V[:, end]

    vals, lvecs, rvecs, info = svdsolve(M, c₀, 1, :SR)
    best_coefs = lvecs[1] ./ norm(lvecs[1])

    println("Loss: ", vals[1])
    println("Left most singular vector: ", best_coefs)
    println("Info: ", info)

    return best_coefs
end

function solve_dense(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    A = get_matrix(geometry, n_modes, λ)

    # Calculate the exact dense SVD (extremely robust, solves in ~0.2s)
    F = svd(A)
    
    # LinearAlgebra.svd sorts singular values in descending order (largest to smallest)
    # Since you want the equivalent of :SR (smallest real), we take the last index
    loss = F.S[end]
    
    best_coefs = F.V[:, end] ./ norm(F.V[:, end])

    println("Loss: ", loss)
    # println("Left most singular vector: ", best_coefs)

    return best_coefs
end

function compute_infinity_norm(geometry::Geometry, coefficients, n_modes, λ)
    A = get_matrix(geometry, n_modes, λ)
    ∂ₙu = A * coefficients

    return maximum(abs.(∂ₙu))
end
