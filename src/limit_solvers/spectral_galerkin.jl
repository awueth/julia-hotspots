module SpectralGalerkin

using FastGaussQuadrature
using KrylovKit
using LinearAlgebra

export SpectralGalerkinProblem, solve_galerkin, reconstruct_field, reconstruct_full_field
export evolve_coefficients, evaluate_solution
export TensorProductBasis, MixedSineBasis1D, HalfCosineBasis1D
export RectangularDomain

# ─────────────────────────────────────────────
# 1. Domain
# ─────────────────────────────────────────────

struct RectangularDomain
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64

    function RectangularDomain(x_min, x_max, y_min, y_max)
        @assert x_max > x_min && y_max > y_min "Domain dimensions must be positive"
        new(Float64(x_min), Float64(x_max), Float64(y_min), Float64(y_max))
    end
end

# ─────────────────────────────────────────────
# 2. Abstract Basis Types
# ─────────────────────────────────────────────

abstract type AbstractBasis1D end

n_modes(b::AbstractBasis1D) = b.n_modes

# ─────────────────────────────────────────────
# Mixed Sine Basis (X-direction, 1st quadrant)
# ─────────────────────────────────────────────

"""
    MixedSineBasis1D(n_modes, width)

Basis on [0, width] with Dirichlet at x=0 and Neumann at x=width:
    φ_m(x) = sin((2m-1) π x / (2 width)),  m = 1, 2, ..., n_modes
"""
struct MixedSineBasis1D <: AbstractBasis1D
    n_modes::Int
    width::Float64
end

mode_indices(b::MixedSineBasis1D) = 1:b.n_modes

@inline function evaluate(b::MixedSineBasis1D, m::Int, x::Float64)
    k = (2 * m - 1) * π / (2 * b.width)
    return sin(k * x)
end

@inline function evaluate_deriv(b::MixedSineBasis1D, m::Int, x::Float64)
    k = (2 * m - 1) * π / (2 * b.width)
    return k * cos(k * x)
end

function analytical_mass(b::MixedSineBasis1D)
    M = zeros(b.n_modes, b.n_modes)
    for i in 1:b.n_modes
        M[i, i] = b.width / 2.0
    end
    return M
end

function analytical_stiffness(b::MixedSineBasis1D)
    A = zeros(b.n_modes, b.n_modes)
    for (i, m) in enumerate(mode_indices(b))
        k = (2 * m - 1) * π / (2 * b.width)
        A[i, i] = k^2 * (b.width / 2.0)
    end
    return A
end

# ─────────────────────────────────────────────
# Half Cosine Basis (Y-direction, 1st quadrant)
# ─────────────────────────────────────────────

"""
    HalfCosineBasis1D(n_modes, width)

Cosine basis on [0, width] with Neumann BCs at both ends:
    φ_m(y) = cos(m π y / width),  m = 0, 1, ..., n_modes-1
"""
struct HalfCosineBasis1D <: AbstractBasis1D
    n_modes::Int
    width::Float64
end

mode_indices(b::HalfCosineBasis1D) = 0:(b.n_modes - 1)

@inline function evaluate(b::HalfCosineBasis1D, m::Int, y::Float64)
    k = m * π / b.width
    return cos(k * y)
end

@inline function evaluate_deriv(b::HalfCosineBasis1D, m::Int, y::Float64)
    k = m * π / b.width
    return -k * sin(k * y)
end

function analytical_mass(b::HalfCosineBasis1D)
    M = zeros(b.n_modes, b.n_modes)
    for (i, m) in enumerate(mode_indices(b))
        M[i, i] = m == 0 ? b.width : b.width / 2.0
    end
    return M
end

function analytical_stiffness(b::HalfCosineBasis1D)
    A = zeros(b.n_modes, b.n_modes)
    for (i, m) in enumerate(mode_indices(b))
        k = m * π / b.width
        A[i, i] = k^2 * (m == 0 ? b.width : b.width / 2.0)
    end
    return A
end

# ─────────────────────────────────────────────
# 7. Tensor Product Basis
# ─────────────────────────────────────────────

struct TensorProductBasis{Bx<:AbstractBasis1D,By<:AbstractBasis1D}
    basis_x::Bx
    basis_y::By
end

n_dof(b::TensorProductBasis) = n_modes(b.basis_x) * n_modes(b.basis_y)

# ─────────────────────────────────────────────
# 8. Matrix Assembly
# ─────────────────────────────────────────────

"""
    assemble_matrices(basis, domain, grad_V, Nquad)

Assemble mass (M), stiffness (K), and advection (B) matrices.

This solver only supports `TensorProductBasis(MixedSineBasis1D, HalfCosineBasis1D)`.
For this basis pair the mass and stiffness matrices are diagonal, so they are
stored as diagonal vectors. Advection is assembled numerically.
"""
function assemble_matrices(
    basis::TensorProductBasis{MixedSineBasis1D,HalfCosineBasis1D},
    domain::RectangularDomain,
    grad_V::Function,
    Nquad::Int,
)
    bx, by = basis.basis_x, basis.basis_y

    # Mass matrix
    Mx = analytical_mass(bx)
    My = analytical_mass(by)
    M = kron(Mx, My)

    # Stiffness matrix
    Ax = analytical_stiffness(bx)
    Ay = analytical_stiffness(by)
    K = kron(Ax, My) + kron(Mx, Ay)

    B = _assemble_advection(basis, domain, grad_V, Nquad)

    return K, M, B
end

function _gl_points_weights(Nquad::Int, a::Float64, b::Float64)
    ξ, w = gausslegendre(Nquad)
    # Map from [-1, 1] to [a, b]
    mid = (a + b) / 2
    half = (b - a) / 2
    pts = mid .+ half .* ξ
    wts = half .* w
    return pts, wts
end

function _assemble_advection(
    basis::TensorProductBasis{MixedSineBasis1D,HalfCosineBasis1D},
    domain::RectangularDomain,
    grad_V::Function,
    Nquad::Int,
)
    bx, by = basis.basis_x, basis.basis_y
    Nx, Ny = n_modes(bx), n_modes(by)
    N_basis = Nx * Ny

    x_pts, x_wts = _gl_points_weights(Nquad, domain.x_min, domain.x_max)
    y_pts, y_wts = _gl_points_weights(Nquad, domain.y_min, domain.y_max)

    Xvals = Matrix{Float64}(undef, Nquad, Nx)
    dXvals = Matrix{Float64}(undef, Nquad, Nx)
    Yvals = Matrix{Float64}(undef, Nquad, Ny)
    dYvals = Matrix{Float64}(undef, Nquad, Ny)

    kx = [((2 * m - 1) * π) / (2 * bx.width) for m in 1:Nx]
    ky = [(n * π) / by.width for n in 0:(Ny - 1)]

    @inbounds for i in 1:Nquad
        x = x_pts[i]
        for mi in 1:Nx
            s, c = sincos(kx[mi] * x)
            Xvals[i, mi] = s
            dXvals[i, mi] = kx[mi] * c
        end
    end

    @inbounds for j in 1:Nquad
        y = y_pts[j]
        for ni in 1:Ny
            s, c = sincos(ky[ni] * y)
            Yvals[j, ni] = c
            dYvals[j, ni] = -ky[ni] * s
        end
    end

    Vx = Matrix{Float64}(undef, Nquad, Nquad)
    Vy = Matrix{Float64}(undef, Nquad, Nquad)
    @inbounds for i in 1:Nquad
        x = x_pts[i]
        for j in 1:Nquad
            gx, gy = grad_V(x, y_pts[j])
            Vx[i, j] = Float64(gx)
            Vy[i, j] = Float64(gy)
        end
    end

    # Contract the tensor-product form along y first, then finish with two GEMMs.
    # This avoids building Phi/dPhi tables of size (Nquad^2) x (Nx*Ny).
    XdX = Matrix{Float64}(undef, Nx * Nx, Nquad)
    XX = Matrix{Float64}(undef, Nx * Nx, Nquad)
    @inbounds for i in 1:Nquad
        idx = 1
        xw = x_wts[i]
        for mj in 1:Nx
            dxj = dXvals[i, mj]
            xj = Xvals[i, mj]
            for mi in 1:Nx
                xi = xw * Xvals[i, mi]
                XdX[idx, i] = xi * dxj
                XX[idx, i] = xi * xj
                idx += 1
            end
        end
    end

    YY = Matrix{Float64}(undef, Ny * Ny, Nquad)
    YdY = Matrix{Float64}(undef, Ny * Ny, Nquad)
    @views @inbounds for i in 1:Nquad
        col_yy = YY[:, i]
        col_ydy = YdY[:, i]
        fill!(col_yy, 0.0)
        fill!(col_ydy, 0.0)
        for j in 1:Nquad
            wyvx = y_wts[j] * Vx[i, j]
            wyvy = y_wts[j] * Vy[i, j]
            idx = 1
            for nj in 1:Ny
                yj = Yvals[j, nj]
                dyj = dYvals[j, nj]
                for ni in 1:Ny
                    yi = Yvals[j, ni]
                    col_yy[idx] = muladd(wyvx * yi, yj, col_yy[idx])
                    col_ydy[idx] = muladd(wyvy * yi, dyj, col_ydy[idx])
                    idx += 1
                end
            end
        end
    end

    contracted = XdX * YY'
    mul!(contracted, XX, YdY', 1.0, 1.0)

    B_view = PermutedDimsArray(reshape(contracted, Nx, Nx, Ny, Ny), (3, 1, 4, 2))
    return reshape(copy(B_view), N_basis, N_basis)
end

# ─────────────────────────────────────────────
# 9. Problem Struct and Constructor
# ─────────────────────────────────────────────

struct SpectralGalerkinProblem{B<:TensorProductBasis}
    K::Matrix{Float64}
    B_adv::Matrix{Float64}
    M::Matrix{Float64}
    n_dof::Int
    basis::B
    domain::RectangularDomain
end

function SpectralGalerkinProblem(
    basis::TensorProductBasis,
    domain::RectangularDomain,
    grad_V::Function,
    Nquad::Int,
)
    K, M, B = assemble_matrices(basis, domain, grad_V, Nquad)
    return SpectralGalerkinProblem(K, B, M, n_dof(basis), basis, domain)
end

# ─────────────────────────────────────────────
# 10. Eigenvalue Solver
# ─────────────────────────────────────────────

"""
    solve_galerkin(prob; nev=6, solver=:dense)

Solve (K + B) u = λ M u for the smallest eigenvalues.

`solver` can be `:dense` (uses `eigen`) or `:krylov` (uses KrylovKit).
"""
function solve_galerkin(prob::SpectralGalerkinProblem; nev::Int=6, solver::Symbol=:dense, v0=prob.n_dof)
    A = prob.K + prob.B_adv
    M = prob.M

    if solver == :krylov
        A_factored = lu(A)
        matvec(z) = A_factored \ (M * z)
        eigenvalues_inv, eigenvectors, _ = eigsolve(
            matvec, v0, nev, :LM; # :LM asks for Largest Magnitude
            issymmetric=false,
            krylovdim=max(3 * nev, 30),
            maxiter=800,
            verbosity=2
        )
        
        # Convert inverted eigenvalues back to original λ
        vals = 1.0 ./ eigenvalues_inv
        
        # Sort to ensure they are strictly ordered from smallest to largest real part
        perm = sortperm(real.(vals))
        
        eigvec_matrix = hcat(eigenvectors[perm][1:nev]...)
        return real.(vals[perm][1:nev]), real.(eigvec_matrix)
    else
        vals, vecs = eigen(A, M)
        perm = sortperm(real.(vals))
        vals = vals[perm]
        vecs = vecs[:, perm]

        nev_actual = min(nev, length(vals))
        max_imag = maximum(abs.(imag.(vals[1:nev_actual])))
        if max_imag > 1e-10
            @warn "Eigenvalues have non-negligible imaginary parts: max |imag| = $max_imag"
        end

        return real.(vals[1:nev_actual]), real.(vecs[:, 1:nev_actual])
    end
end

# ─────────────────────────────────────────────
# 11. Time Evolution
# ─────────────────────────────────────────────

function _mixed_half_basis(prob::SpectralGalerkinProblem)
    bx, by = prob.basis.basis_x, prob.basis.basis_y
    if !(bx isa MixedSineBasis1D && by isa HalfCosineBasis1D)
        throw(ArgumentError("Time evolution is only implemented for TensorProductBasis(MixedSineBasis1D, HalfCosineBasis1D)"))
    end
    return bx, by
end

function _reshape_coefficients(prob::SpectralGalerkinProblem, coeffs::AbstractVector)
    bx, by = prob.basis.basis_x, prob.basis.basis_y
    Nx, Ny = n_modes(bx), n_modes(by)
    @assert length(coeffs) == Nx * Ny "Coefficient vector length mismatch"
    return reshape(coeffs, Ny, Nx)'
end

"""
    evolve_coefficients(prob, coeffs, λ, t)

Evolve mixed/half-basis coefficients exactly to time `t` for
    ∂_t f = (Δf + λ f) / 8.
"""
function evolve_coefficients(
    prob::SpectralGalerkinProblem,
    coeffs::AbstractVector,
    λ::Real,
    t::Real,
)
    bx, by = _mixed_half_basis(prob)
    C = _reshape_coefficients(prob, coeffs)
    C_t = Matrix{Float64}(undef, size(C))

    for (mi, m) in enumerate(mode_indices(bx))
        kx = (2 * m - 1) * π / (2 * bx.width)
        for (ni, n) in enumerate(mode_indices(by))
            ky = n * π / by.width
            r = (Float64(λ) - kx^2 - ky^2) / 8.0
            C_t[mi, ni] = Float64(C[mi, ni]) * exp(r * Float64(t))
        end
    end

    return vec(C_t')
end

"""
    evaluate_solution(prob, coeffs, λ, x, y, t)

Evaluate the mixed/half-basis solution u(x, y, t) obtained by exact modal
time evolution from the initial coefficients `coeffs`.
"""
function evaluate_solution(
    prob::SpectralGalerkinProblem,
    coeffs::AbstractVector,
    λ::Real,
    x::Real,
    y::Real,
    t::Real,
)
    bx, by = _mixed_half_basis(prob)
    C = _reshape_coefficients(prob, coeffs)

    x_val = Float64(x)
    y_val = Float64(y)
    t_val = Float64(t)
    λ_val = Float64(λ)

    val = 0.0
    for (mi, m) in enumerate(mode_indices(bx))
        X_val = evaluate(bx, m, x_val)
        kx = (2 * m - 1) * π / (2 * bx.width)
        for (ni, n) in enumerate(mode_indices(by))
            Y_val = evaluate(by, n, y_val)
            ky = n * π / by.width
            r = (λ_val - kx^2 - ky^2) / 8.0
            val += Float64(C[mi, ni]) * exp(r * t_val) * X_val * Y_val
        end
    end

    return val
end

# ─────────────────────────────────────────────
# 12. Field Reconstruction
# ─────────────────────────────────────────────

"""
    reconstruct_field(prob, coeffs; nx=100, ny=50)

Reconstruct u(x, y) = Σ c_{m,n} X_m(x) Y_n(y) on a uniform grid.
"""
function reconstruct_field(
    prob::SpectralGalerkinProblem,
    coeffs::AbstractVector;
    nx::Int=100,
    ny::Int=50,
)
    bx, by = prob.basis.basis_x, prob.basis.basis_y
    Nx, Ny = n_modes(bx), n_modes(by)
    C = _reshape_coefficients(prob, coeffs)  # C[m, n]

    x_grid = range(prob.domain.x_min, prob.domain.x_max, length=nx)
    y_grid = range(prob.domain.y_min, prob.domain.y_max, length=ny)

    X_vals = zeros(nx, Nx)
    Y_vals = zeros(ny, Ny)

    for (i, x) in enumerate(x_grid), (mi, m) in enumerate(mode_indices(bx))
        X_vals[i, mi] = evaluate(bx, m, x)
    end
    for (j, y) in enumerate(y_grid), (ni, n) in enumerate(mode_indices(by))
        Y_vals[j, ni] = evaluate(by, n, y)
    end

    u_grid = X_vals * C * Y_vals'

    return collect(x_grid), collect(y_grid), u_grid
end

"""
    reconstruct_full_field(prob, coeffs; nx=100, ny=50)

Reconstruct field on the full domain [-W_x, W_x] x [-W_y, W_y]
using antisymmetry in X and symmetry in Y.
"""
function reconstruct_full_field(
    prob::SpectralGalerkinProblem,
    coeffs::AbstractVector;
    nx::Int=100,
    ny::Int=50,
)
    # 1. Reconstruct on first quadrant
    x_grid_q1, y_grid_q1, u_grid_q1 = reconstruct_field(prob, coeffs; nx=nx, ny=ny)

    # 2. X-direction (antisymmetric)
    if isapprox(x_grid_q1[1], 0.0; atol=1e-12)
        full_x_grid = vcat(reverse(-x_grid_q1[2:end]), x_grid_q1)
        full_u_x    = vcat(-reverse(u_grid_q1[2:end, :], dims=1), u_grid_q1)
    else
        full_x_grid = vcat(reverse(-x_grid_q1), x_grid_q1)
        full_u_x    = vcat(-reverse(u_grid_q1, dims=1), u_grid_q1)
    end

    # 3. Y-direction (symmetric)
    if isapprox(y_grid_q1[1], 0.0; atol=1e-12)
        full_y_grid = vcat(reverse(-y_grid_q1[2:end]), y_grid_q1)
        full_u_grid = hcat(reverse(full_u_x[:, 2:end], dims=2), full_u_x)
    else
        full_y_grid = vcat(reverse(-y_grid_q1), y_grid_q1)
        full_u_grid = hcat(reverse(full_u_x, dims=2), full_u_x)
    end

    return full_x_grid, full_y_grid, full_u_grid
end

end # module SpectralGalerkin
