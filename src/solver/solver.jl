include("../functions/basis.jl")

using KrylovKit
using AppleAccelerate
using LinearAlgebra
using Optim
using Random

const InfiniteCartesianPoints{T} = NamedTuple{
    (:x, :y, :r),
    <:Tuple{AbstractVector{T}, AbstractVector{T}, Nothing}
}

const FiniteCartesianPoints{T} = NamedTuple{
    (:x, :y, :r),
    <:Tuple{AbstractVector{T}, AbstractVector{T}, <:AbstractMatrix{T}}
}

const CartesianNormals{T} = NamedTuple{
    (:x, :y, :r),
    <:Tuple{<:AbstractMatrix{T}, <:AbstractMatrix{T}, <:AbstractMatrix{T}}
}

struct Geometry{DT, T <: AbstractFloat, F1, F2, P, N}
    d::DT
    diam_x::T
    diam_y::T
    V::F1
    gradV::F2
    points::P
    normals::N
end

function fibonacci_lattice_points(diam_x::T, diam_y::T, n_samples::Int) where {T <: AbstractFloat}
    half_x = T(0.5) * diam_x
    half_y = T(0.5) * diam_y
    golden_ratio = (one(T) + sqrt(T(5.0))) / T(2.0)

    xs = Vector{T}(undef, n_samples)
    ys = Vector{T}(undef, n_samples)
    for i in 1:n_samples
        k = i - 1
        xs[i] = ((k + T(0.5)) / n_samples) * half_x
        ys[i] = mod(k / golden_ratio, one(T)) * half_y
    end

    return xs, ys
end

# If n_points is a tuple, sample from a uniform grid. 
function make_geometry(
    d::T, 
    diam_x::T, 
    diam_y::T, 
    V::F1, 
    gradV::F2,
    n_points::Tuple{Int,Int}
) where {T <: AbstractFloat, F1, F2}
    n_x, n_y = n_points
    
    xs = collect(range(zero(T), T(0.5) * diam_x, length=n_x))
    push!(xs, T(0.5) * π)
    ys = collect(range(zero(T), T(0.5) * diam_y, length=n_y))

    rs = isinf(d) ? nothing : [one(T) - V(x, y) / d for x in xs, y in ys]
    points = (x=xs, y=ys, r=rs)

    grads = [gradV(x, y) for x in xs, y in ys]
    nx = [T(g[1]) for g in grads]
    ny = [T(g[2]) for g in grads]
    nr = fill(T(4.0), size(nx)) 
    
    inv_len = one(T) ./ sqrt.(nx .^ 2 .+ ny .^ 2 .+ nr .^ 2)
    normals = (x=nx .* inv_len, y=ny .* inv_len, r=nr .* inv_len)

    return Geometry(isinf(d) ? nothing : d, diam_x, diam_y, V, gradV, points, normals)
end

# If n_points is an integer, use Fibonacci sampling
function make_geometry(
    d::T, 
    diam_x::T, 
    diam_y::T, 
    V::F1, 
    gradV::F2,
    n_points::Int
) where {T <: AbstractFloat, F1, F2}
    xs, ys = fibonacci_lattice_points(diam_x, diam_y, n_points)

    rs = isinf(d) ? nothing : [one(T) - V(xs[i], ys[i]) / d for i in eachindex(xs)]
    points = (x=xs, y=ys, r=rs)

    grads = [gradV(xs[i], ys[i]) for i in eachindex(xs)]
    nx = [T(g[1]) for g in grads]
    ny = [T(g[2]) for g in grads]
    nr = fill(T(4.0), size(nx))

    inv_len = one(T) ./ sqrt.(nx .^ 2 .+ ny .^ 2 .+ nr .^ 2)
    normals = (x=nx .* inv_len, y=ny .* inv_len, r=nr .* inv_len)

    return Geometry(isinf(d) ? nothing : d, diam_x, diam_y, V, gradV, points, normals)
end

function get_eigenvalues(diam_x::T, diam_y::T, n_modes::Tuple{Int,Int}, λ::T) where {T <: AbstractFloat}
    n_modes_x, n_modes_y = n_modes
    modes_x = T.(1:2:(2 * n_modes_x - 1))
    modes_y = T.(0:2:(2 * n_modes_y - 2))
    λx_modes = (modes_x .* (T(π) / diam_x)) .^ 2
    λy_modes = (modes_y .* (T(π) / diam_y)) .^ 2

    grid = vec(collect(Iterators.product(λx_modes, λy_modes)))
    λx = [g[1] for g in grid]
    λy = [g[2] for g in grid]
    λr = λ .- (λx .+ λy)

    return λx, λy, λr
end

function get_eigenvalues(geometry::Geometry{<:Any, T}, n_modes::Tuple{Int,Int}, λ::T) where {T <: AbstractFloat}
    return get_eigenvalues(geometry.diam_x, geometry.diam_y, n_modes, λ)
end

function _weights_vector(weights, n_rows::Int)
    weights_vec = isnothing(weights) ? nothing : vec(weights)
    if !isnothing(weights_vec) && length(weights_vec) != n_rows
        throw(DimensionMismatch("weights must contain $n_rows entries, got $(length(weights_vec))"))
    end
    return weights_vec
end

function get_matrix(
    geometry::Geometry{Nothing, T}, 
    n_modes::Tuple{Int, Int},
    λ::T;
    weights::Union{Nothing, AbstractVector{T}, AbstractMatrix{T}} = nothing
) where {T <: AbstractFloat}
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    diam_x, diam_y = geometry.diam_x, geometry.diam_y
    
    xs, ys, rs = geometry.points
    nxs, nys, nrs = geometry.normals
    d = geometry.d

    n_samples = length(xs)

    total_modes = length(λx)
    weights_vec = _weights_vector(weights, n_samples)

    M = zeros(T, n_samples, total_modes)

    Threads.@threads for j in eachindex(λx)
        lx, ly, lr = λx[j], λy[j], λr[j]
        rv, rgrad = (one(T), T(-0.25) * lr)
        @inbounds for i in eachindex(xs)
            av, (agx, agy) = axial_basis(lx, ly, diam_x, diam_y, xs[i], ys[i])
            
            val = (nxs[i] * agx * rv) +
                  (nys[i] * agy * rv) +
                  (nrs[i] * av * rgrad)
            
            M[i, j] = isnothing(weights_vec) ? val : val * weights_vec[i]
        end
    end

    return M
end

function get_matrix(
    geometry::Geometry{T, T, F1, F2, P, N}, 
    n_modes::Tuple{Int, Int},
    λ::T;
    weights::Union{Nothing, AbstractVector{T}, AbstractMatrix{T}} = nothing
) where {T <: AbstractFloat, F1, F2, P <: FiniteCartesianPoints{T}, N <: CartesianNormals{T}}
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    diam_x, diam_y = geometry.diam_x, geometry.diam_y
    
    xs, ys, rs = geometry.points
    nxs, nys, nrs = geometry.normals
    d = geometry.d
    
    n_x, n_y = length(xs), length(ys)
    total_modes = length(λx)
    weights_vec = _weights_vector(weights, n_x * n_y)
    M_tensor = zeros(T, n_x, n_y, total_modes)

    Threads.@threads for j in eachindex(λx)
        lx, ly, lr = λx[j], λy[j], λr[j]
        @inbounds for iy in eachindex(ys), ix in eachindex(xs)
            av, (agx, agy) = axial_basis(lx, ly, diam_x, diam_y, xs[ix], ys[iy])
            rv, rgrad = ϕ(d, lr, rs[ix, iy])
            
            val = (nxs[ix, iy] * agx * rv) + 
                  (nys[ix, iy] * agy * rv) + 
                  (nrs[ix, iy] * av * rgrad)
            
            # Apply weight if provided
            row = ix + (iy - 1) * n_x
            M_tensor[ix, iy, j] = isnothing(weights_vec) ? val : val * weights_vec[row]
        end
    end

    return reshape(M_tensor, n_x * n_y, total_modes)
end

function get_matrix(
    geometry::Geometry{T, T}, 
    n_modes::Tuple{Int, Int},
    λ::T;
    weights::Union{Nothing, AbstractVector{T}, AbstractMatrix{T}} = nothing
) where {T <: AbstractFloat} # Non-Cartesian collocation points
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    diam_x, diam_y = geometry.diam_x, geometry.diam_y
    
    xs, ys, rs = geometry.points
    nxs, nys, nrs = geometry.normals
    d = geometry.d

    n_samples = length(xs)

    total_modes = length(λx)
    weights_vec = _weights_vector(weights, n_samples)

    M = zeros(T, n_samples, total_modes)

    Threads.@threads for j in eachindex(λx)
        lx, ly, lr = λx[j], λy[j], λr[j]
        @inbounds for i in eachindex(xs)
            av, (agx, agy) = axial_basis(lx, ly, diam_x, diam_y, xs[i], ys[i])
            rv, rgrad = ϕ(d, lr, rs[i])
            
            val = (nxs[i] * agx * rv) +
                  (nys[i] * agy * rv) +
                  (nrs[i] * av * rgrad)
            
            M[i, j] = isnothing(weights_vec) ? val : val * weights_vec[i]
        end
    end

    return M
end

function get_matrix(
    geometry::Geometry{Nothing, T, F1, F2, P, N}, 
    n_modes::Tuple{Int, Int},
    λ::T;
    weights::Union{Nothing, AbstractVector{T}, AbstractMatrix{T}} = nothing
) where {T <: AbstractFloat, F1, F2, P <: InfiniteCartesianPoints{T}, N <: CartesianNormals{T}}
    xs, ys = geometry.points.x, geometry.points.y
    nxs, nys, nrs = geometry.normals.x, geometry.normals.y, geometry.normals.r
    n_x, n_y = length(xs), length(ys)
    diam_x, diam_y = geometry.diam_x, geometry.diam_y
    mx, my = n_modes
    total_modes = mx * my

    weights_vec = _weights_vector(weights, n_x * n_y)
    tables = axial_basis_tables(xs, ys, n_modes, diam_x, diam_y)

     M = Matrix{T}(undef, n_x * n_y, total_modes)

    Threads.@threads for col in 1:total_modes
        p = ((col - 1) % mx) + 1
        q = ((col - 1) ÷ mx) + 1
        rgrad = T(-0.25) * (λ - tables.λx_modes[p] - tables.λy_modes[q])

        @inbounds for iy in eachindex(ys), ix in eachindex(xs)
            row = ix + (iy - 1) * n_x

            av, (agx, agy) = axial_basis(tables, ix, iy, p, q)

            val = nxs[ix, iy] * agx +
                nys[ix, iy] * agy +
                nrs[ix, iy] * av * rgrad

            M[row, col] = isnothing(weights_vec) ? val : val * weights_vec[row]
        end
    end

    return M
end

function u(
    d::Nothing,
    coefficients::AbstractVector{T},
    λx::AbstractVector{T},
    λy::AbstractVector{T},
    λr::AbstractVector{T},
    diam_x::T,
    diam_y::T,
    x::T,
    y::T,
    r::T
) where {T <: AbstractFloat}
    val = zero(T)
    for i in eachindex(coefficients)
        av, _ = axial_basis(λx[i], λy[i], diam_x, diam_y, x, y)
        rv, _ = ϕ(T(Inf), λr[i], r)
        val += coefficients[i] * av * rv
    end
    return val
end

function u(
    d::T,
    coefficients::AbstractVector{T},
    λx::AbstractVector{T},
    λy::AbstractVector{T},
    λr::AbstractVector{T},
    diam_x::T,
    diam_y::T,
    x::T,
    y::T,
    r::T
) where {T <: AbstractFloat}
    val = zero(T)
    for i in eachindex(coefficients)
        av, _ = axial_basis(λx[i], λy[i], diam_x, diam_y, x, y)
        rv, _ = ϕ(d, λr[i], r)
        val += coefficients[i] * av * rv
    end
    return val
end

function u(
    geometry::Geometry{<:Any, T},
    coefficients::AbstractVector{T},
    λx::AbstractVector{T},
    λy::AbstractVector{T},
    λr::AbstractVector{T},
    x::T,
    y::T,
    r::T
) where {T <: AbstractFloat}
    diam_x, diam_y = geometry.diam_x, geometry.diam_y
    return u(geometry.d, coefficients, λx, λy, λr, diam_x, diam_y, x, y, r)
end

function u(
    d::T,
    diam_x::T,
    diam_y::T,
    coefficients::AbstractVector{T},
    λ::T,
    n_modes::Tuple{Int,Int},
    x::T,
    y::T,
    r::T
) where {T <: AbstractFloat}
    λx, λy, λr = get_eigenvalues(diam_x, diam_y, n_modes, λ)
    return u(d, coefficients, λx, λy, λr, diam_x, diam_y, x, y, r)
end

function u(
    geometry::Geometry{<:Any, T},
    coefficients::AbstractVector{T},
    λ::T,
    n_modes::Tuple{Int,Int},
    x::T,
    y::T,
    r::T
) where {T <: AbstractFloat}
    return u(geometry.d, geometry.diam_x, geometry.diam_y, coefficients, λ, n_modes, x, y, r)
end

function optimize_eigenvalue(geometry::Geometry{<:Any, T}, n_modes::Tuple{Int,Int}, bounds::Tuple{T,T}; weights=nothing, solver=:iterative) where {T <: AbstractFloat}
    lower, upper = bounds

    if solver == :iterative
        objective = λ -> begin
            A = get_matrix(geometry, n_modes, λ; weights=weights)
            _, loss, _ = iterative_normal_solution(A, n_modes)
            return loss
        end
    elseif solver == :dense
        objective = λ -> svdvals(get_matrix(geometry, n_modes, λ; weights=weights))[end]
    else
        throw(ArgumentError("unknown solver: $solver"))
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

function submatrix_initial_guess(A::AbstractMatrix{T}, n_modes::Tuple{Int,Int}) where {T}
    n_modes_x, n_modes_y = n_modes
    seed_modes_x = min(16, n_modes_x)
    seed_modes_y = min(32, n_modes_y)
    seed_cols = [
        ix + (iy - 1) * n_modes_x
        for iy in 1:seed_modes_y
        for ix in 1:seed_modes_x
    ]

    F_seed = svd(@view A[:, seed_cols]; full=false)
    c₀ = zeros(T, size(A, 2))
    c₀[seed_cols] .= F_seed.V[:, end]

    return c₀ ./ norm(c₀)
end

function shifted_cholesky(A::Symmetric{T,<:AbstractMatrix}) where {T}
    scale = max(opnorm(A, Inf), one(T))
    shift = eps(T) * scale

    for _ in 1:8
        try
            return cholesky(A + shift * I), shift
        catch err
            err isa PosDefException || rethrow()
            shift *= T(100.0)
        end
    end

    return cholesky(A + shift * I), shift
end

function iterative_normal_solution(A::AbstractMatrix{T}, n_modes::Tuple{Int,Int}) where {T}
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
        tol=T(1e-10),
        eager=true
    )
    best_coefs = vecs[1] ./ norm(vecs[1])
    best_coefs .*= iszero(best_coefs[1]) ? one(T) : sign(best_coefs[1])

    normal_loss = dot(best_coefs, normal_matrix * best_coefs)
    loss = sqrt(max(normal_loss, zero(T)))

    return best_coefs, loss, info
end

function solve_iterative(geometry::Geometry{<:Any, T}, n_modes::Tuple{Int,Int}, λ::T; weights=nothing) where {T <: AbstractFloat}
    A = get_matrix(geometry, n_modes, λ; weights=weights)
    best_coefs, loss, info = iterative_normal_solution(A, n_modes)

    println("Loss: ", loss)
    println("Info: ", info)

    residual = A * best_coefs

    return best_coefs, residual
end

function solve_dense(geometry::Geometry{<:Any, T}, n_modes::Tuple{Int,Int}, λ::T; weights=nothing) where {T <: AbstractFloat}
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
    geometry::Geometry{T, T},
    coefficients::AbstractVector{T},
    λx::AbstractVector{T},
    λy::AbstractVector{T},
    λr::AbstractVector{T},
    x::T,
    y::T
) where {T <: AbstractFloat}
    r = one(T) - geometry.V(x, y) / geometry.d
    ∂xV, ∂yV = geometry.gradV(x, y)
    nr = T(4.0)
    inv_len = inv(sqrt(T(∂xV^2 + ∂yV^2 + nr^2)))
    nx, ny, nr_scaled = T(∂xV) * inv_len, T(∂yV) * inv_len, nr * inv_len
    diam_x, diam_y = geometry.diam_x, geometry.diam_y

    residual = zero(T)
    @inbounds @simd for i in eachindex(coefficients)
        av, (agx, agy) = axial_basis(λx[i], λy[i], diam_x, diam_y, x, y)
        rv, rgrad = ϕ(geometry.d, λr[i], r)
        residual += coefficients[i] * (nx * agx * rv + ny * agy * rv + nr_scaled * av * rgrad)
    end

    return residual
end

function boundary_residual(
    geometry::Geometry{T, T},
    coefficients::AbstractVector{T},
    λ::T,
    n_modes::Tuple{Int,Int},
    grid_size::Tuple{Int,Int}
) where {T <: AbstractFloat}
    nx, ny = grid_size
    dx = (T(0.5) * geometry.diam_x) / nx
    dy = (T(0.5) * geometry.diam_y) / ny

    xs = [ (i-1)*dx + T(rand())*dx for i in 1:nx ]
    ys = [ (j-1)*dy + T(rand())*dy for j in 1:ny ]

    residuals = zeros(T, length(xs), length(ys))

    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)

    Threads.@threads for j in eachindex(ys)
        for i in eachindex(xs)
            residuals[i, j] = boundary_residual(geometry, coefficients, λx, λy, λr, xs[i], ys[j])
        end
    end

    return residuals, xs, ys
end

function boundary_residual(
    geometry::Geometry{Nothing, T},
    coefficients::AbstractVector{T},
    λ::T,
    n_modes::Tuple{Int,Int},
    grid_size::Tuple{Int,Int}
) where {T <: AbstractFloat}
    nx_grid, ny_grid = grid_size
    mx, my = n_modes
    dx = (T(0.5) * geometry.diam_x) / nx_grid
    dy = (T(0.5) * geometry.diam_y) / ny_grid

    # 1. Create Cartesian grid coordinates
    xs = [ (i-1)*dx + T(rand())*dx for i in 1:nx_grid ]
    ys = [ (j-1)*dy + T(rand())*dy for j in 1:ny_grid ]

    # 2. Reshape modal parameters into (mx, my) matrices
    λx_vec, λy_vec, λr_vec = get_eigenvalues(geometry, n_modes, λ)
    C = reshape(coefficients, mx, my)
    Lx = reshape(λx_vec, mx, my)
    Ly = reshape(λy_vec, mx, my)
    Lr = reshape(λr_vec, mx, my)

    # 3. Precompute k-vectors and normalization factors
    Kx_vec = sqrt.(Lx[:, 1]) # kx only depends on the x-mode index
    Ky_vec = sqrt.(Ly[1, :]) # ky only depends on the y-mode index
    
    norm_x_sq = T(0.5) * geometry.diam_x
    # norm_y_sq depends on whether λy is 0
    norm_y_sq = [ (ly == zero(T)) ? geometry.diam_y : (geometry.diam_y / T(2.0)) for ly in Ly[1, :] ]
    InvNorm = one(T) ./ sqrt.(norm_x_sq .* norm_y_sq') # (1, my) broadcasting

    # 4. Construct Weight Matrices (combining coefficients with basis constants)
    # W1 for nx terms (∂x), W2 for ny terms (∂y), W3 for nr terms (value * rgrad)
    W1 = C .* InvNorm .* Kx_vec
    W2 = C .* InvNorm .* (-Ky_vec')
    W3 = C .* InvNorm .* (T(-0.25) .* Lr)

    # 5. Evaluate 1D basis functions on the grid
    Sx_grid = sin.(xs .* Kx_vec')  # (nx_grid, mx)
    Cx_grid = cos.(xs .* Kx_vec')  # (nx_grid, mx)
    Sy_grid = sin.(ys .* Ky_vec')  # (ny_grid, my)
    Cy_grid = cos.(ys .* Ky_vec')  # (ny_grid, my)

    # 6. High-performance Matrix Multiplications (GEMM)
    # T1[i, j] = Σ_jx,jy W1[jx,jy] * cos(kx*xi) * cos(ky*yj)
    T1 = Cx_grid * (W1 * Cy_grid') 
    # T2[i, j] = Σ_jx,jy W2[jx,jy] * sin(kx*xi) * sin(ky*yj)
    T2 = Sx_grid * (W2 * Sy_grid')
    # T3[i, j] = Σ_jx,jy W3[jx,jy] * sin(kx*xi) * cos(ky*yj)
    T3 = Sx_grid * (W3 * Cy_grid')

    # 7. Final assembly with point-dependent normals
    residuals = zeros(T, nx_grid, ny_grid)
    gradV = geometry.gradV
    nr_base = T(4.0)
    
    Threads.@threads for j in 1:ny_grid
        for i in 1:nx_grid
            gx, gy = gradV(xs[i], ys[j])
            inv_len = inv(sqrt(T(gx^2 + gy^2 + nr_base^2)))
            nx = T(gx) * inv_len
            ny = T(gy) * inv_len
            nr = nr_base * inv_len
            
            residuals[i, j] = nx * T1[i, j] + ny * T2[i, j] + nr * T3[i, j]
        end
    end

    return residuals, xs, ys
end
