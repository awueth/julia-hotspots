include("../functions/basis.jl")

using KrylovKit
using AppleAccelerate
using LinearAlgebra
using Optim
using Random

"""
Solver for the pde -Δu = λu with Neumann boundary conditions on the domain
    
{(x, y, w) : x ∈ [-diam_x/2, diam_x/2], y ∈ [-diam_y/2, diam_y/2], |w| ≤ (√d - V(x, y)/√d)/2}

After a change of variables, we obtain the effective problem -Δ_tranformed u = λu on the domain

{(x, y, r) : x ∈ [0, diam_x/2], y ∈ [0, diam_y/2], r ≤ 1 - V(x, y)/d}

with Dirichlet boundary conditions at x=0 and Neumann boundary conditions everywhere else. 

We use the method of particular solutions to solve this problem. We express u as a linear combination of basis functions:

u(x, y, r) = ∑ cₙ axial_basisₙ(x, y) * radial_basisₙ(r).

The radial_basis functions are such that radial_basisₙ(1) = 1. Therefore, at d=∞, the the domain reduces to

{(x, y) : x ∈ [0, diam_x/2], y ∈ [0, diam_y/2]}

and the ansatz to u(x, y) = ∑ cₙ axial_basisₙ(x, y).

In d=∞, the Neumann boundary condition at the face r=1-V/d converges to -Δu + ∇V ⋅ ∇u = λu. The measure converges to 1/Z exp(-V(x, y)) dx dy. 

"""

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

struct FibonacciSampler
    n::Int
end

function (sampler::FibonacciSampler)(diam_x::T, diam_y::T) where {T <: AbstractFloat}
    n_samples = sampler.n
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

    return xs, ys, false
end

struct GridSampler
    nx::Int
    ny::Int
end

function (sampler::GridSampler)(diam_x::T, diam_y::T) where {T <: AbstractFloat}
    n_x, n_y = sampler.nx, sampler.ny
    xs = collect(range(zero(T), T(0.5) * diam_x, length=n_x))
    #push!(xs, T(0.5) * pi)
    ys = collect(range(zero(T), T(0.5) * diam_y, length=n_y))

    return xs, ys, true
end
 
function make_geometry(
    d::T, 
    diam_x::T, 
    diam_y::T, 
    V::F1, 
    gradV::F2,
    sampler
) where {T <: AbstractFloat, F1, F2}    
    xs, ys, is_grid = sampler(diam_x, diam_y)

    if is_grid
        rs = isinf(d) ? nothing : [one(T) - V(x, y) / d for x in xs, y in ys]
        grads = [gradV(x, y) for x in xs, y in ys]
    else
        rs = isinf(d) ? nothing : [one(T) - V(xs[i], ys[i]) / d for i in eachindex(xs)]
        grads = [gradV(xs[i], ys[i]) for i in eachindex(xs)]
    end

    points = (x=xs, y=ys, r=rs)
    nx = [T(g[1]) for g in grads]
    ny = [T(g[2]) for g in grads]
    nr = fill(T(4.0), size(nx)) 
    
    # inv_len is the wrong normalization factor for the normals if our objective is to compute the physical normal derivative.
    # However, it does minimize the residual of the Neumann boundary condition.
    # The true physical normal derivative would underflow in the wings. 
    # The current normalization lets the eigenfunction converge to what we expect, 
    # however the scaling is somewhat arbitrary. 
    # It would be better to introduce the scaling explicitly at some point in the solver.
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

function get_interior_matrix(
    xs::AbstractVector{T},
    ys::AbstractVector{T},
    λx::AbstractVector{T},
    λy::AbstractVector{T},
    diam_x::T,
    diam_y::T,
    V 
) where {T <: AbstractFloat}

    M = Matrix{T}(undef, length(xs),  length(λx))

    Threads.@threads for j in eachindex(λx)
        lx, ly = λx[j], λy[j]
        @inbounds for i in eachindex(xs)
            av, _ = axial_basis(lx, ly, diam_x, diam_y, xs[i], ys[i])
            
            M[i, j] = av * exp(-0.5 * V(xs[i], ys[i]))
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

abstract type AbstractSolver end
struct DenseSolver <: AbstractSolver end
struct IterativeSolver <: AbstractSolver end
struct QRSolver <: AbstractSolver
    interior_matrix
end

function QRSolver(geometry::Geometry{Nothing, T}, n_modes::Tuple{Int,Int}, λ::T, sampler) where {T <: AbstractFloat}
    xs, ys, _ = sampler(geometry.diam_x, geometry.diam_y)

    λx, λy, _ = get_eigenvalues(geometry, n_modes, λ)
    interior_matrix = get_interior_matrix(xs, ys, λx, λy, geometry.diam_x, geometry.diam_y, geometry.V)

    return QRSolver(interior_matrix)
end

function solve_coefficients(A, n_modes, solver::DenseSolver; weights=nothing)
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

function solve_coefficients(A, n_modes, solver::IterativeSolver; weights=nothing)
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
    best_coefs .*= sign(best_coefs[1])

    normal_loss = dot(best_coefs, normal_matrix * best_coefs)
    loss = sqrt(normal_loss)

    return best_coefs, loss # , info
end

function solve_coefficients(A_boundary, n_modes, solver::QRSolver; weights=nothing)

    A_interior = solver.interior_matrix

    A = vcat(A_boundary, A_interior)
    F = qr(A)
    Q_boundary = F.Q[1:size(A_boundary, 1), 1:size(A_boundary, 2)]
    S = svd!(Q_boundary; full=false)
    loss = S.S[end]
    v = S.V[:, end]
    best_coefs = F.R \ v
    best_coefs .*= sign(best_coefs[1])
    best_coefs ./= norm(best_coefs) # Temporary

    return best_coefs, loss
end

function solve(
    geometry::Geometry{<:Any, T},
    n_modes::Tuple{Int,Int},
    λ::T, 
    solver::AbstractSolver;
    weights=nothing
) where {T <: AbstractFloat}
    A = get_matrix(geometry, n_modes, λ; weights=weights)
    return solve_coefficients(A, n_modes, solver; weights=weights)
end

function solver_loss(A, n_modes, solver::IterativeSolver)
    _, loss = solve_coefficients(A, n_modes, solver)
    return loss
end

function solver_loss(A, n_modes, solver::DenseSolver)
    return svdvals(A)[end]
end

function optimize_eigenvalue(
    geometry::Geometry{<:Any, T}, 
    n_modes::Tuple{Int,Int}, 
    bounds::Tuple{T,T},
    solver::AbstractSolver; 
    weights=nothing, 
) where {T <: AbstractFloat}

    lower, upper = bounds

    objective = λ -> begin
        A = get_matrix(geometry, n_modes, λ; weights=weights)
        return solver_loss(A, n_modes, solver)
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
    # Unit outward normal on the finite-d barrel: ∇G = (∇V/(2√d), w/|w|) has
    # length sqrt(4d + |∇V|²)/(2√d), so the assembled numerator ∇V·∇ₓφ + 4∂ᵣφ
    # becomes the physical normal derivative ∂ₙφ once divided by sqrt(4d + |∇V|²)
    # (see writeup/barrel.typ).
    inv_len = inv(sqrt(T(T(4.0) * geometry.d + ∂xV^2 + ∂yV^2)))
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
    sampler
) where {T <: AbstractFloat}
    xs, ys, is_grid = sampler(geometry.diam_x, geometry.diam_y)

    @assert is_grid "Sampler must produce a grid for this function"
    mx, my = n_modes

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

    # 4. Construct Weight Matrices (combining coefficients with basis constants)
    # W1 for nx terms (∂x), W2 for ny terms (∂y), W3 for nr terms (value * rgrad)
    W1 = C .* Kx_vec
    W2 = C .* (-Ky_vec')
    W3 = C .* (T(-0.25) .* Lr)

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
    residuals = zeros(T, length(xs), length(ys))
    gradV = geometry.gradV
    nr_base = T(4.0)
    
    Threads.@threads for j in eachindex(ys)
        for i in eachindex(xs)
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
