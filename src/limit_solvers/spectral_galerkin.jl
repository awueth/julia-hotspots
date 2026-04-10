module SpectralGalerkin

using FastGaussQuadrature
using KrylovKit
using LinearAlgebra

export SpectralGalerkinProblem, solve_galerkin, reconstruct_field, reconstruct_full_field
export evolve_coefficients, evaluate_solution
export TensorProductBasis, CosineBasis1D, SymmetricCosineBasis1D, SineWingBasis1D, CustomBasis1D, MixedSineBasis1D, HalfCosineBasis1D
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
analytical_mass(::AbstractBasis1D) = nothing
analytical_stiffness(::AbstractBasis1D) = nothing

# ─────────────────────────────────────────────
# 3. Cosine Basis (Neumann BCs)
# ─────────────────────────────────────────────

"""
    CosineBasis1D(n_modes, half_width)

Cosine basis on [-half_width, half_width] with Neumann BCs:
    φ_m(x) = cos(m π (x + half_width) / (2 half_width)),  m = 0, 1, ..., n_modes-1
"""
struct CosineBasis1D <: AbstractBasis1D
    n_modes::Int
    half_width::Float64
end

mode_indices(b::CosineBasis1D) = 0:(b.n_modes - 1)

@inline function evaluate(b::CosineBasis1D, m::Int, x::Float64)
    k = m * π / (2 * b.half_width)
    return cos(k * (x + b.half_width))
end

@inline function evaluate_deriv(b::CosineBasis1D, m::Int, x::Float64)
    k = m * π / (2 * b.half_width)
    return -k * sin(k * (x + b.half_width))
end

function analytical_mass(b::CosineBasis1D)
    M = zeros(b.n_modes, b.n_modes)
    for (i, m) in enumerate(mode_indices(b))
        M[i, i] = m == 0 ? 2 * b.half_width : b.half_width
    end
    return M
end

function analytical_stiffness(b::CosineBasis1D)
    A = zeros(b.n_modes, b.n_modes)
    for (i, m) in enumerate(mode_indices(b))
        k = m * π / (2 * b.half_width)
        A[i, i] = k^2 * (m == 0 ? 2 * b.half_width : b.half_width)
    end
    return A
end

# ─────────────────────────────────────────────
# 4. Symmetric Cosine Basis (bespoke Y-direction)
# ─────────────────────────────────────────────

"""
    SymmetricCosineBasis1D(n_modes, half_width)

Symmetric cosine basis on [-half_width, half_width]:
    Y_n(y) = cos(2nπy / (2 half_width)),  n = 0, 1, ..., n_modes-1
"""
struct SymmetricCosineBasis1D <: AbstractBasis1D
    n_modes::Int
    half_width::Float64
end

mode_indices(b::SymmetricCosineBasis1D) = 0:(b.n_modes - 1)

@inline function evaluate(b::SymmetricCosineBasis1D, n::Int, y::Float64)
    Ly = 2 * b.half_width
    return cos(2 * n * π * y / Ly)
end

@inline function evaluate_deriv(b::SymmetricCosineBasis1D, n::Int, y::Float64)
    Ly = 2 * b.half_width
    k = 2 * n * π / Ly
    return -k * sin(k * y)
end

function analytical_mass(b::SymmetricCosineBasis1D)
    Ly = 2 * b.half_width
    M = zeros(b.n_modes, b.n_modes)
    M[1, 1] = Ly  # n = 0 mode
    for i in 2:b.n_modes
        M[i, i] = Ly / 2
    end
    return M
end

function analytical_stiffness(b::SymmetricCosineBasis1D)
    Ly = 2 * b.half_width
    A = zeros(b.n_modes, b.n_modes)
    for i in 2:b.n_modes
        n = i - 1  # 0-indexed mode number
        k = 2 * n * π / Ly
        A[i, i] = k^2 * Ly / 2
    end
    return A
end

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
# 5. Sine-Wing Basis (bespoke X-direction)
# ─────────────────────────────────────────────

"""
    SineWingBasis1D(n_modes, a, half_width)

Antisymmetric sine basis with constant wings on [-half_width, half_width]:
    X_m(x) = sin((2m-1)πx / 2a)   for |x| ≤ a
    X_m(x) = ±1                     for |x| > a
Mode indices: m = 1, 2, ..., n_modes
"""
struct SineWingBasis1D <: AbstractBasis1D
    n_modes::Int
    a::Float64
    half_width::Float64

    function SineWingBasis1D(n_modes, a, half_width)
        @assert 0 < a < half_width "Need 0 < a < half_width"
        new(n_modes, Float64(a), Float64(half_width))
    end
end

mode_indices(b::SineWingBasis1D) = 1:b.n_modes

@inline function evaluate(b::SineWingBasis1D, m::Int, x::Float64)
    k = (2 * m - 1) * π / (2 * b.a)
    if x < -b.a
        return sin(k * (-b.a))
    elseif x > b.a
        return sin(k * b.a)
    else
        return sin(k * x)
    end
end

@inline function evaluate_deriv(b::SineWingBasis1D, m::Int, x::Float64)
    if abs(x) > b.a
        return 0.0
    else
        k = (2 * m - 1) * π / (2 * b.a)
        return k * cos(k * x)
    end
end

function analytical_mass(b::SineWingBasis1D)
    Lx = 2 * b.half_width
    wing_contrib = Lx - 2 * b.a
    M = fill(wing_contrib, b.n_modes, b.n_modes)
    for m in 1:b.n_modes
        M[m, m] += b.a
    end
    return M
end

function analytical_stiffness(b::SineWingBasis1D)
    A = zeros(b.n_modes, b.n_modes)
    for m in 1:b.n_modes
        k = (2 * m - 1) * π / (2 * b.a)
        A[m, m] = k^2 * b.a
    end
    return A
end

# ─────────────────────────────────────────────
# 6. Custom Basis (user-provided closures)
# ─────────────────────────────────────────────

"""
    CustomBasis1D(n_modes, indices, eval_fn, eval_deriv_fn)

User-provided 1D basis. `eval_fn(m, x)` and `eval_deriv_fn(m, x)` are closures.
`indices` is the range of mode indices (e.g., `0:N-1` or `1:N`).
"""
struct CustomBasis1D{F1,F2,I} <: AbstractBasis1D
    n_modes::Int
    indices::I
    eval_fn::F1
    eval_deriv_fn::F2
end

mode_indices(b::CustomBasis1D) = b.indices
evaluate(b::CustomBasis1D, m::Int, x::Float64) = b.eval_fn(m, x)
evaluate_deriv(b::CustomBasis1D, m::Int, x::Float64) = b.eval_deriv_fn(m, x)

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

Uses analytical Kronecker products for M and K when the 1D bases provide
`analytical_mass` / `analytical_stiffness`. Falls back to numerical
Gauss-Legendre quadrature otherwise. Advection is always numerical.
"""
function assemble_matrices(
    basis::TensorProductBasis,
    domain::RectangularDomain,
    grad_V::Function,
    Nquad::Int,
)
    bx, by = basis.basis_x, basis.basis_y
    Nx, Ny = n_modes(bx), n_modes(by)
    N_basis = Nx * Ny

    # --- Mass matrix ---
    Mx_ana = analytical_mass(bx)
    My_ana = analytical_mass(by)
    if Mx_ana !== nothing && My_ana !== nothing
        M = kron(Mx_ana, My_ana)
    else
        M = _assemble_mass_numerical(basis, domain, Nquad)
    end

    # --- Stiffness matrix ---
    Ax_ana = analytical_stiffness(bx)
    Ay_ana = analytical_stiffness(by)
    if Ax_ana !== nothing && Ay_ana !== nothing && Mx_ana !== nothing && My_ana !== nothing
        K = kron(Ax_ana, My_ana) + kron(Mx_ana, Ay_ana)
    else
        K = _assemble_stiffness_numerical(basis, domain, Nquad)
    end

    # --- Advection matrix (always numerical) ---
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

function _assemble_mass_numerical(basis::TensorProductBasis, domain::RectangularDomain, Nquad::Int)
    bx, by = basis.basis_x, basis.basis_y
    Nx, Ny = n_modes(bx), n_modes(by)
    N_basis = Nx * Ny

    x_pts, x_wts = _gl_points_weights(Nquad, domain.x_min, domain.x_max)
    y_pts, y_wts = _gl_points_weights(Nquad, domain.y_min, domain.y_max)

    # Evaluate all basis values at quadrature points
    Xvals = zeros(Nquad, Nx)
    Yvals = zeros(Nquad, Ny)
    for (i, x) in enumerate(x_pts), (mi, m) in enumerate(mode_indices(bx))
        Xvals[i, mi] = evaluate(bx, m, x)
    end
    for (j, y) in enumerate(y_pts), (ni, n) in enumerate(mode_indices(by))
        Yvals[j, ni] = evaluate(by, n, y)
    end

    # M_X[m, m'] = Σ_i X_m(x_i) X_{m'}(x_i) w_i
    Mx = Xvals' * Diagonal(x_wts) * Xvals
    My = Yvals' * Diagonal(y_wts) * Yvals
    return kron(Mx, My)
end

function _assemble_stiffness_numerical(basis::TensorProductBasis, domain::RectangularDomain, Nquad::Int)
    bx, by = basis.basis_x, basis.basis_y
    Nx, Ny = n_modes(bx), n_modes(by)

    x_pts, x_wts = _gl_points_weights(Nquad, domain.x_min, domain.x_max)
    y_pts, y_wts = _gl_points_weights(Nquad, domain.y_min, domain.y_max)

    Xvals = zeros(Nquad, Nx)
    dXvals = zeros(Nquad, Nx)
    Yvals = zeros(Nquad, Ny)
    dYvals = zeros(Nquad, Ny)

    for (i, x) in enumerate(x_pts), (mi, m) in enumerate(mode_indices(bx))
        Xvals[i, mi] = evaluate(bx, m, x)
        dXvals[i, mi] = evaluate_deriv(bx, m, x)
    end
    for (j, y) in enumerate(y_pts), (ni, n) in enumerate(mode_indices(by))
        Yvals[j, ni] = evaluate(by, n, y)
        dYvals[j, ni] = evaluate_deriv(by, n, y)
    end

    Mx = Xvals' * Diagonal(x_wts) * Xvals
    Ax = dXvals' * Diagonal(x_wts) * dXvals
    My = Yvals' * Diagonal(y_wts) * Yvals
    Ay = dYvals' * Diagonal(y_wts) * dYvals

    return kron(Ax, My) + kron(Mx, Ay)
end

function _assemble_advection(
    basis::TensorProductBasis,
    domain::RectangularDomain,
    grad_V::Function,
    Nquad::Int,
)
    bx, by = basis.basis_x, basis.basis_y
    Nx, Ny = n_modes(bx), n_modes(by)
    N_basis = Nx * Ny

    x_pts, x_wts = _gl_points_weights(Nquad, domain.x_min, domain.x_max)
    y_pts, y_wts = _gl_points_weights(Nquad, domain.y_min, domain.y_max)
    N_pts = Nquad * Nquad

    # Evaluate 1D basis functions at quadrature points
    Xvals = zeros(Nquad, Nx)
    dXvals = zeros(Nquad, Nx)
    Yvals = zeros(Nquad, Ny)
    dYvals = zeros(Nquad, Ny)

    for (i, x) in enumerate(x_pts), (mi, m) in enumerate(mode_indices(bx))
        Xvals[i, mi] = evaluate(bx, m, x)
        dXvals[i, mi] = evaluate_deriv(bx, m, x)
    end
    for (j, y) in enumerate(y_pts), (ni, n) in enumerate(mode_indices(by))
        Yvals[j, ni] = evaluate(by, n, y)
        dYvals[j, ni] = evaluate_deriv(by, n, y)
    end

    # Build 2D basis, derivative, and weight arrays
    # Flatten 2D grid: point index = (j-1)*Nquad + i  (y-major ordering)
    Phi  = zeros(N_pts, N_basis)  # Φ_{m,n}(x_i, y_j)
    dPhi_x = zeros(N_pts, N_basis)  # ∂_x Φ
    dPhi_y = zeros(N_pts, N_basis)  # ∂_y Φ
    W = zeros(N_pts)  # quadrature weights
    Vx_vec = zeros(N_pts)
    Vy_vec = zeros(N_pts)

    pt = 1
    for j in 1:Nquad
        for i in 1:Nquad
            W[pt] = x_wts[i] * y_wts[j]
            gV = grad_V(x_pts[i], y_pts[j])
            Vx_vec[pt] = gV[1]
            Vy_vec[pt] = gV[2]

            for mi in 1:Nx
                for ni in 1:Ny
                    b_idx = (mi - 1) * Ny + ni
                    Phi[pt, b_idx]    = Xvals[i, mi] * Yvals[j, ni]
                    dPhi_x[pt, b_idx] = dXvals[i, mi] * Yvals[j, ni]
                    dPhi_y[pt, b_idx] = Xvals[i, mi] * dYvals[j, ni]
                end
            end
            pt += 1
        end
    end

    # B_ij = ∫ (∇V · ∇Φ_j) Φ_i dA
    # = Σ_q [ (Vx * ∂_x Φ_j + Vy * ∂_y Φ_j) * Φ_i * w_q ]
    Drift = Vx_vec .* dPhi_x .+ Vy_vec .* dPhi_y  # N_pts × N_basis
    B = Phi' * Diagonal(W) * Drift  # N_basis × N_basis

    return B
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
        # matvec(z) = M_factored \ (A * z)
        # eigenvalues, eigenvectors, _ = eigsolve(
        #     matvec, prob.n_dof, nev, :SR;
        #     issymmetric=false,
        #     krylovdim=max(3 * nev, 30),
        #     maxiter=300,
        # )
        # eigvec_matrix = hcat(eigenvectors[1:nev]...)
        # return real.(eigenvalues[1:nev]), eigvec_matrix
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
