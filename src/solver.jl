include("functions/basis.jl")

using KrylovKit
using LinearAlgebra
using Optim
using Random

struct Geometry{F1,F2}
    d::Float64
    diam_x::Float64
    diam_y::Float64
    V::F1
    gradV::F2
    points::@NamedTuple{x::Vector{Float64}, y::Vector{Float64}, r::Matrix{Float64}}
    normals::@NamedTuple{x::Matrix{Float64}, y::Matrix{Float64}, r::Matrix{Float64}}
end

function make_geometry(
    d::Float64, 
    diam_x::Float64, 
    diam_y::Float64, 
    V::F1, 
    gradV::F2,
    n_points::Tuple{Int,Int}
) where {F1,F2}
    n_x, n_y = n_points
    
    xs = collect(range(0, 0.5 * diam_x, length=n_x))
    push!(xs, 0.5 * pi)
    ys = collect(range(0, 0.5 * diam_y, length=n_y))

    rs = [1.0 - V(x, y) / d for x in xs, y in ys]
    points = (x=xs, y=ys, r=rs)

    grads = [gradV(x, y) for x in xs, y in ys]
    nx = [g[1] for g in grads]
    ny = [g[2] for g in grads]
    nr = fill(4.0, size(nx)) 
    
    inv_len = 1 ./ sqrt.(nx .^ 2 .+ ny .^ 2 .+ nr .^ 2)
    normals = (x=nx .* inv_len, y=ny .* inv_len, r=nr .* inv_len)

    return Geometry(d, diam_x, diam_y, V, gradV, points, normals)
end

mode_counts(n_modes::Int) = (n_modes, n_modes)
mode_counts(n_modes::Tuple{Int,Int}) = n_modes

function get_eigenvalues(diam_x::Float64, diam_y::Float64, n_modes::Tuple{Int,Int}, λ::Float64)
    n_modes_x, n_modes_y = mode_counts(n_modes)
    modes_x = 1:2:(2 * n_modes_x - 1)
    modes_y = 0:2:(2 * n_modes_y - 2)
    λx_modes = (modes_x .* (π / diam_x)) .^ 2
    λy_modes = (modes_y .* (π / diam_y)) .^ 2

    grid = vec(collect(Iterators.product(λx_modes, λy_modes)))
    λx = [g[1] for g in grid]
    λy = [g[2] for g in grid]
    λr = λ .- (λx .+ λy)

    return λx, λy, λr
end

function get_eigenvalues(geometry::Geometry, n_modes::Tuple{Int,Int}, λ::Float64)
    return get_eigenvalues(geometry.diam_x, geometry.diam_y, n_modes, λ)
end

function get_matrix(
    xs::Vector{Float64},
    ys::Vector{Float64},
    rs::Matrix{Float64},
    nxs::Matrix{Float64},
    nys::Matrix{Float64},
    nrs::Matrix{Float64},
    λx::Vector{Float64},
    λy::Vector{Float64},
    λr::Vector{Float64},
    diam_x::Float64,
    diam_y::Float64,
    d::Float64;
    weights::Union{Nothing, Matrix{Float64}} = Nothing() # New optional argument
)
    n_x, n_y = length(xs), length(ys)
    total_modes = length(λx)
    M_tensor = zeros(Float64, n_x, n_y, total_modes)

    Threads.@threads for j in eachindex(λx)
        lx, ly, lr = λx[j], λy[j], λr[j]
        @inbounds for iy in eachindex(ys), ix in eachindex(xs)
            av, (agx, agy) = axial_basis(lx, ly, diam_x, diam_y, xs[ix], ys[iy])
            rv, rgrad = ϕ(d, lr, rs[ix, iy])
            
            val = (nxs[ix, iy] * agx * rv) + 
                  (nys[ix, iy] * agy * rv) + 
                  (nrs[ix, iy] * av * rgrad)
            
            # Apply weight if provided
            M_tensor[ix, iy, j] = isnothing(weights) ? val : val * weights[ix, iy]
        end
    end

    return reshape(M_tensor, n_x * n_y, total_modes)
end

function get_matrix(geometry::Geometry, n_modes::Tuple{Int,Int}, λ::Float64; weights::Union{Nothing, Matrix{Float64}} = Nothing())
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    diam_x, diam_y = geometry.diam_x, geometry.diam_y
    total_modes = length(λx)
    
    xs, ys, rs = geometry.points
    nxs, nys, nrs = geometry.normals
    d = geometry.d

    return get_matrix(xs, ys, rs, nxs, nys, nrs, λx, λy, λr, diam_x, diam_y, d; weights=weights)
end

function u(
    d::Float64,
    coefficients::AbstractVector{Float64},
    λx::AbstractVector{Float64},
    λy::AbstractVector{Float64},
    λr::AbstractVector{Float64},
    diam_x::Float64,
    diam_y::Float64,
    x::Float64,
    y::Float64,
    r::Float64
)
    val = 0.0
    for i in eachindex(coefficients)
        av, _ = axial_basis(λx[i], λy[i], diam_x, diam_y, x, y)
        rv, _ = ϕ(d, λr[i], r)
        val += coefficients[i] * av * rv
    end
    return val
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
    diam_x, diam_y = geometry.diam_x, geometry.diam_y
    return u(geometry.d, coefficients, λx, λy, λr, diam_x, diam_y, x, y, r)
end

function u(
    d::Float64,
    diam_x::Float64,
    diam_y::Float64,
    coefficients::AbstractVector{Float64},
    λ::Float64,
    n_modes::Tuple{Int,Int},
    x::Float64,
    y::Float64,
    r::Float64
)
    λx, λy, λr = get_eigenvalues(diam_x, diam_y, n_modes, λ)
    return u(d, coefficients, λx, λy, λr, diam_x, diam_y, x, y, r)
end

function u(
    geometry::Geometry,
    coefficients::AbstractVector{Float64},
    λ::Float64,
    n_modes::Tuple{Int,Int},
    x::Float64,
    y::Float64,
    r::Float64
)
    return u(geometry.d, geometry.diam_x, geometry.diam_y, coefficients, λ, n_modes, x, y, r)
end

function optimize_eigenvalue(geometry::Geometry, n_modes::Tuple{Int,Int}, bounds::Tuple{Float64,Float64}; weights=nothing)
    lower, upper = bounds

    objective(λ) = svdvals(get_matrix(geometry, n_modes, λ; weights=weights))[end]

    result = optimize(objective, lower, upper, Brent())
    best_λ = Optim.minimizer(result)
    best_loss = Optim.minimum(result)

    println("Optimization Successful: ", Optim.converged(result))
    println("Optimal λ: ", best_λ)
    println("Minimum Loss: ", best_loss)
    println("Iterations: ", Optim.iterations(result))

    return best_λ, best_loss
end

function submatrix_initial_guess(A::AbstractMatrix{Float64}, n_modes::Tuple{Int,Int})
    n_modes_x, n_modes_y = mode_counts(n_modes)
    seed_modes_x = min(16, n_modes_x)
    seed_modes_y = min(32, n_modes_y)
    seed_cols = [
        ix + (iy - 1) * n_modes_x
        for iy in 1:seed_modes_y
        for ix in 1:seed_modes_x
    ]

    F_seed = svd(@view A[:, seed_cols]; full=false)
    c₀ = zeros(Float64, size(A, 2))
    c₀[seed_cols] .= F_seed.V[:, end]

    return c₀ ./ norm(c₀)
end

function shifted_cholesky(A::Symmetric{Float64,<:AbstractMatrix{Float64}})
    scale = max(opnorm(A, Inf), one(Float64))
    shift = eps(Float64) * scale

    for _ in 1:8
        try
            return cholesky(A + shift * I), shift
        catch err
            err isa PosDefException || rethrow()
            shift *= 100.0
        end
    end

    return cholesky(A + shift * I), shift
end

function iterative_normal_solution(A::AbstractMatrix{Float64}, n_modes::Tuple{Int,Int})
    c₀ = submatrix_initial_guess(A, n_modes)

    normal_matrix = Symmetric(A' * A)
    factor, _ = shifted_cholesky(normal_matrix)
    inverse_normal_matvec(c) = factor \ c

    _, vecs, info = eigsolve(
        inverse_normal_matvec,
        c₀,
        1,
        :LM;
        issymmetric=true,
        krylovdim=min(length(c₀), 80),
        maxiter=500,
        tol=1e-10,
        eager=true
    )
    best_coefs = vecs[1] ./ norm(vecs[1])
    best_coefs .*= iszero(best_coefs[1]) ? one(best_coefs[1]) : sign(best_coefs[1])

    normal_loss = dot(best_coefs, normal_matrix * best_coefs)
    loss = sqrt(max(normal_loss, zero(normal_loss)))

    return best_coefs, loss, info
end

function solve_iterative(geometry::Geometry, n_modes::Tuple{Int,Int}, λ::Float64; weights=nothing)
    A = get_matrix(geometry, n_modes, λ; weights=weights)
    best_coefs, loss, info = iterative_normal_solution(A, n_modes)

    println("Loss: ", loss)
    println("Info: ", info)

    residual = A * best_coefs

    return best_coefs, residual
end

function optimize_eigenvalue_iterative(geometry::Geometry, n_modes::Tuple{Int,Int}, bounds::Tuple{Float64,Float64}; weights=nothing)
    lower, upper = bounds

    function objective(λ)
        A = get_matrix(geometry, n_modes, λ; weights=weights)
        _, loss, _ = iterative_normal_solution(A, n_modes)
        return loss
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

function solve_dense(geometry::Geometry, n_modes::Tuple{Int,Int}, λ::Float64; weights=nothing)
    A = get_matrix(geometry, n_modes, λ; weights=weights)
    F = svd!(A; full=false)
    loss = F.S[end]

    v_norm = norm(F.V[:, end])
    best_coefs = F.V[:, end] ./ v_norm
    c_sign = sign(best_coefs[1])
    best_coefs .*= c_sign

    println("Loss: ", loss)
    
    residual = F.U[:, end] .* (loss * c_sign / v_norm)

    return best_coefs, residual
end

function boundary_residual(
    geometry::Geometry,
    coefficients::AbstractVector{Float64},
    λx::AbstractVector{Float64},
    λy::AbstractVector{Float64},
    λr::AbstractVector{Float64},
    x::Float64,
    y::Float64
)
    r = 1.0 - geometry.V(x, y) / geometry.d
    ∂xV, ∂yV = geometry.gradV(x, y)
    nr = 4.0
    inv_len = inv(sqrt(∂xV^2 + ∂yV^2 + nr^2))
    nx, ny, nr_scaled = ∂xV * inv_len, ∂yV * inv_len, nr * inv_len
    diam_x, diam_y = geometry.diam_x, geometry.diam_y

    residual = 0.0
    @inbounds @simd for i in eachindex(coefficients)
        av, (agx, agy) = axial_basis(λx[i], λy[i], diam_x, diam_y, x, y)
        rv, rgrad = ϕ(geometry.d, λr[i], r)
        residual += coefficients[i] * (nx * agx * rv + ny * agy * rv + nr_scaled * av * rgrad)
    end

    return residual
end

function boundary_residual(
    geometry::Geometry,
    coefficients::AbstractVector{Float64},
    λ::Float64,
    n_modes::Tuple{Int,Int},
    grid_size::Tuple{Int,Int}
)
    nx, ny = grid_size
    dx = (0.5 * geometry.diam_x) / nx
    dy = (0.5 * geometry.diam_y) / ny

    xs = [ (i-1)*dx + rand()*dx for i in 1:nx ]
    ys = [ (j-1)*dy + rand()*dy for j in 1:ny ]

    residuals = zeros(Float64, length(xs), length(ys))

    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)

    Threads.@threads for j in eachindex(ys)
        for i in eachindex(xs)
            residuals[i, j] = boundary_residual(geometry, coefficients, λx, λy, λr, xs[i], ys[j])
        end
    end

    return residuals, xs, ys
end