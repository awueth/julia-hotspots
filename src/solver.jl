include("functions/basis.jl")

using KrylovKit
using LinearAlgebra
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
    n_points::Tuple{Int,Int}
) where {F1,F2}
    n_x, n_y = n_points
    x_grid = range(0, 0.5 * diam_x, length=n_x)
    y_grid = range(0, 0.5 * diam_y, length=n_y)
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

mode_counts(n_modes::Int) = (n_modes, n_modes)
mode_counts(n_modes::Tuple{Int,Int}) = n_modes

function get_eigenvalues(diam_x::Float64, diam_y::Float64, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
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

function get_eigenvalues(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    return get_eigenvalues(geometry.diam_x, geometry.diam_y, n_modes, λ)
end

function get_matrix(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    λx, λy, λr = get_eigenvalues(geometry, n_modes, λ)
    total_modes = length(λx)
    n_points = length(geometry.points.x)

    M = zeros(Float64, n_points, total_modes)

    xs, ys, rs = geometry.points
    nxs, nys, nrs = geometry.normals
    d = geometry.d

    Threads.@threads for j in 1:total_modes
        lx, ly, lr = λx[j], λy[j], λr[j]

        @inbounds for i in 1:n_points
            av, (agx, agy) = axial_basis(lx, ly, xs[i], ys[i])
            rv, rgrad = ϕ(d, lr, rs[i])
            
            M[i, j] = (nxs[i] * agx * rv) + 
                      (nys[i] * agy * rv) + 
                      (nrs[i] * av * rgrad)
        end
    end

    return M
end

function u(
    d::Float64,
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
    return u(geometry.d, coefficients, λx, λy, λr, x, y, r)
end

function u(
    d::Float64,
    diam_x::Float64,
    diam_y::Float64,
    coefficients::AbstractVector{Float64},
    λ::Float64,
    n_modes::Union{Int,Tuple{Int,Int}},
    x::Float64,
    y::Float64,
    r::Float64
)
    λx, λy, λr = get_eigenvalues(diam_x, diam_y, n_modes, λ)
    return u(d, coefficients, λx, λy, λr, x, y, r)
end

function u(
    geometry::Geometry,
    coefficients::AbstractVector{Float64},
    λ::Float64,
    n_modes::Union{Int,Tuple{Int,Int}},
    x::Float64,
    y::Float64,
    r::Float64
)
    return u(geometry.d, geometry.diam_x, geometry.diam_y, coefficients, λ, n_modes, x, y, r)
end

function optimize_eigenvalue(geometry::Geometry, n_modes::Tuple{Int,Int}, bounds::Tuple{Float64,Float64})
    lower, upper = bounds

    objective(λ) = svdvals(get_matrix(geometry, n_modes, λ))[end]

    result = optimize(objective, lower, upper, Brent())
    best_λ = Optim.minimizer(result)
    best_loss = Optim.minimum(result)

    println("Optimization Successful: ", Optim.converged(result))
    println("Optimal λ: ", best_λ)
    println("Minimum Loss: ", best_loss)
    println("Iterations: ", Optim.iterations(result))

    return best_λ, best_loss
end

function submatrix_initial_guess(A::AbstractMatrix{Float64}, n_modes::Union{Int,Tuple{Int,Int}})
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

function iterative_normal_solution(A::AbstractMatrix{Float64}, n_modes::Union{Int,Tuple{Int,Int}})
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

function solve_iterative(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    A = get_matrix(geometry, n_modes, λ)
    best_coefs, loss, info = iterative_normal_solution(A, n_modes)

    println("Loss: ", loss)
    println("Info: ", info)

    residual = A * best_coefs

    return best_coefs, residual
end

function optimize_eigenvalue_iterative(geometry::Geometry, n_modes::Tuple{Int,Int}, bounds::Tuple{Float64,Float64})
    lower, upper = bounds

    function objective(λ)
        A = get_matrix(geometry, n_modes, λ)
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

function solve_dense(geometry::Geometry, n_modes::Union{Int,Tuple{Int,Int}}, λ::Float64)
    A = get_matrix(geometry, n_modes, λ)
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
