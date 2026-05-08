"""
    PotentialGenerator

Module for generating optimized potentials for the eigenvalue solvers.
"""
module PotentialGenerator

using Integrals
using LinearAlgebra
using Optim
using ForwardDiff
using StaticArrays

export PotentialData, PotentialFunctions, generate_potential
export V₀, V, ∇V, V_extended, ∇V_extended

# ==========================================
# Data Structures
# ==========================================

"""
    PotentialData

Metadata and Fourier-Chebyshev coefficients for a generated potential.
"""
struct PotentialData
    Lx::Float64
    Ly::Float64
    q_coeffs::Vector{Float64}
    J_HARMONICS::Int
    optimal_val::Float64
    fourier_chebyshev_coeffs::Matrix{Float64}  # C[m+1, n+1] for T_m(x/Lx) * cos(nπy/Ly)
    V0_sup_norm::Float64
    Mx_convex::Float64
    My_convex::Float64
    M_convex::Float64
end

"""
    PotentialFunctions

Container for generated potential data and reconstructed x-mode polynomials.
`source_polys` is retained only for diagnostics and tests; runtime evaluation uses
the Fourier-Chebyshev representation exclusively.
"""
struct PotentialFunctions{P}
    data::PotentialData
    mode_polys::Vector{Tuple{P, P, P}}
    source_polys::Any
end

struct HessianMode{P}
    n::Int
    poly::Tuple{P, P, P}
    Ly::Float64
end

# ==========================================
# Constants (defaults)
# ==========================================
const DEFAULT_Lx = 0.5 * pi
const DEFAULT_Ly = 1.0
const DEFAULT_q_coeffs = [0.2, 1.0, 0.0]
const DEFAULT_J_HARMONICS = 7
const DEFAULT_COEFFICIENT_PENALTY_WEIGHT = 5e-5
const DEFAULT_PRIMITIVE_SAMPLE_COUNT = 65
const DEFAULT_INTEGRATION_RELTOL = 1e-9
const DEFAULT_INTEGRATION_ABSTOL = 1e-9

default_fourier_chebyshev_degree(J_HARMONICS::Int) = max(4, 4 * J_HARMONICS)

# ==========================================
# Constraint/Polynomial Building
# ==========================================

"""
Build constraint operators for mode n.

fₙ(x) = Σ_j a_(n, j) T_(2j+1)(x / Lx) with constraints:
1. ∑_j a_(n, j) = 1
2. fₙ'(Lx) = 0
3. fₙ''(Lx) = ((nπ/Ly)² - 1)fₙ(Lx)

We solve for the first 3 coefficients in terms of the remaining free coefficients.
"""
function build_constraint_operators(n::Int, J_HARMONICS::Int; Lx::Float64=DEFAULT_Lx, Ly::Float64=DEFAULT_Ly)
    s = 1.0 / Lx
    M = zeros(3, J_HARMONICS + 1)

    for j in 0:J_HARMONICS
        k = 2j + 1
        M[1, j + 1] = 1.0
        M[2, j + 1] = k^2 * s
        M[3, j + 1] = (k^2 * (k^2 - 1.0) / 3.0) * s^2
    end

    A = M[:, 1:3]
    B = M[:, 4:end]

    target = [1.0, 0.0, (n * pi / Ly)^2 - 1.0]
    return A \ target, A \ B
end

"""
Compute
row[j] = ∫_{-Lx}^{Lx} T_{2j+1}(x/Lx) * sin(x) dx
"""
function zero_mode_orthogonality_row(J_HARMONICS::Int; Lx::Float64=DEFAULT_Lx)
    row = zeros(Float64, J_HARMONICS + 1)

    for j in 1:(J_HARMONICS + 1)
        coeffs = zeros(Float64, J_HARMONICS + 1)
        coeffs[j] = 1.0
        p, _, _, _ = get_basis_funcs(coeffs, Lx)

        prob = IntegralProblem((x, _) -> p(x) * sin(x), -Lx, Lx)
        row[j] = solve(prob, QuadGKJL(); reltol=1e-8, abstol=1e-8).u
    end

    return row
end

"""
Build constraint operators for the x-only zeroth mode.
"""
function build_zero_mode_constraint_operators(J_HARMONICS::Int; Lx::Float64=DEFAULT_Lx)
    s = 1.0 / Lx
    M = zeros(3, J_HARMONICS + 1)
    orth_row = zero_mode_orthogonality_row(J_HARMONICS; Lx=Lx)

    for j in 0:J_HARMONICS
        k = 2j + 1
        M[1, j + 1] = k^2 * s
        M[2, j + 1] = (k^2 * (k^2 - 1.0) / 3.0) * s^2 + 1.0
        M[3, j + 1] = orth_row[j + 1]
    end

    A = M[:, 1:3]
    B = M[:, 4:end]

    target = zeros(3)
    return A \ target, A \ B
end

function get_full_coeffs(free_coeffs::AbstractVector, base::AbstractVector, mult::AbstractMatrix)
    dep_coeffs = base - mult * free_coeffs
    return vcat(dep_coeffs, free_coeffs)
end

function split_polys(polys::AbstractVector, n_modes::Int)
    if length(polys) == n_modes
        return nothing, polys
    elseif length(polys) == n_modes + 1
        return polys[1], polys[2:end]
    else
        error("Expected either $n_modes positive modes or $(n_modes + 1) modes including a zeroth mode, got $(length(polys)).")
    end
end

# ==========================================
# Chebyshev Evaluation
# ==========================================

"""
The function ∑ⱼ cⱼ Tⱼ
"""
struct ChebyshevApprox{T}
    coeffs::Vector{T}
    domain::Tuple{T, T}
end

function (p::ChebyshevApprox)(x)
    a, b = p.domain
    # Map x from [a, b] to u in [-1, 1]
    u = (2x - (b + a)) / (b - a)

    coeffs = p.coeffs
    n = length(coeffs)

    if n == 0
        return zero(x)
    elseif n == 1
        return coeffs[1] * one(x)
    end

    bk1 = zero(u)
    bk2 = zero(u)

    for k in n:-1:2
        bk0 = coeffs[k] + 2u * bk1 - bk2
        bk2 = bk1
        bk1 = bk0
    end

    return coeffs[1] + u * bk1 - bk2
end

function ∂(p::ChebyshevApprox)
    a, b = p.domain
    scaling = 2.0 / (b - a)

    n = length(p.coeffs)
    if n <= 1
        return ChebyshevApprox([0.0], p.domain)
    end

    deriv_coeffs = zeros(eltype(p.coeffs), n - 1)

    c_next_plus = 0.0
    c_next = 0.0

    for i in (n - 1):-1:1
        val = 2 * i * p.coeffs[i + 1] + c_next_plus
        deriv_coeffs[i] = val
        c_next_plus = c_next
        c_next = val
    end

    deriv_coeffs[1] /= 2.0
    return ChebyshevApprox(deriv_coeffs .* scaling, p.domain)
end

function get_basis_funcs(coeffs::AbstractVector, Lx::Float64=DEFAULT_Lx)
    c_full = zeros(eltype(coeffs), length(coeffs) * 2)
    c_full[2:2:end] .= coeffs

    p = ChebyshevApprox(c_full, (-Lx, Lx))
    dp = ∂(p)
    ddp = ∂(dp)
    dddp = ∂(ddp)

    return (p, dp, ddp, dddp)
end

function chebyshev_values(u::Real, degree::Int)
    vals = zeros(Float64, degree + 1)
    vals[1] = 1.0

    if degree >= 1
        vals[2] = Float64(u)
        for m in 2:degree
            vals[m + 1] = 2.0 * u * vals[m] - vals[m - 1]
        end
    end

    return vals
end

even_degree_indices(max_degree::Int) = collect(0:2:max_degree)

function primitive_fit_weight(x::Real, Lx::Float64)
    u = clamp(abs(x) / Lx, 0.0, 1.0)
    return 0.05 + u^8
end

function choose_stable_three_columns(R::AbstractMatrix)
    n_cols = size(R, 2)
    best_cols = Int[]
    best_det = -Inf

    for i in 1:(n_cols - 2), j in (i + 1):(n_cols - 1), k in (j + 1):n_cols
        cols = [i, j, k]
        det_val = abs(det(R[:, cols]))
        if det_val > best_det
            best_det = det_val
            best_cols = cols
        end
    end

    best_det < 1e-12 && error("Unable to construct a stable 3-constraint system for the primitive fit.")
    return best_cols
end

function integrate_zero_to_x(f::Function, x::Real; reltol::Float64=DEFAULT_INTEGRATION_RELTOL, abstol::Float64=DEFAULT_INTEGRATION_ABSTOL)
    abs(x) < 1e-14 && return 0.0
    prob = IntegralProblem((s, _) -> x * f(x * s), 0.0, 1.0)
    return solve(prob, QuadGKJL(); reltol=reltol, abstol=abstol).u
end

# ==========================================
# Integrand Evaluation
# ==========================================

"""
    eval_integrand(n, x, p, ddp)

Compute the modal integrand for the base potential V₀, representing
(-(Δ + 1)β₀,ₙ) / cos(x). For a mode β₀,ₙ(x,y) = fₙ(x)cos(nπy/Ly), this
evaluates [((nπ/Ly)² - 1)fₙ(x) - fₙ''(x)] / cos(x).
"""
function eval_integrand(n::Int, x::Real, p, ddp, Ly::Real=DEFAULT_Ly)
    K2 = (n * pi / Ly)^2 - 1.0
    return (K2 * p(x) - ddp(x)) / cos(x)
end

function eval_integrand_boundary_limit(n::Int, x::Real, p, dp, ddp, dddp, Ly::Real=DEFAULT_Ly)
    K2 = (n * pi / Ly)^2 - 1.0
    N_prime = K2 * dp(x) - dddp(x)
    sin_x = sin(x)
    abs(sin_x) < 1e-12 && error("Boundary limit for the integrand is singular because sin(x) vanished.")
    return -N_prime / sin_x
end

function V₀_integral_reference(source_polys::AbstractVector, q_coeffs::Vector{Float64}, x::Real, y::Real, Ly::Real=DEFAULT_Ly)
    n_modes = length(q_coeffs)
    zero_poly, positive_polys = split_polys(source_polys, n_modes)
    T = promote_type(typeof(x), typeof(y))
    val = zero(T)

    if zero_poly !== nothing
        p, _, ddp, _ = zero_poly
        S0 = integrate_zero_to_x(t -> eval_integrand(0, t, p, ddp, Ly), x)
        val += (-0.5 * pi) * S0
    end

    for n in 1:n_modes
        p, _, ddp, _ = positive_polys[n]
        Sn = integrate_zero_to_x(t -> eval_integrand(n, t, p, ddp, Ly), x)
        val += (-0.5 * pi * q_coeffs[n]) * Sn * cos(n * pi * y / Ly)
    end

    return val
end

# ==========================================
# Primitive Fitting
# ==========================================

function fit_even_chebyshev_primitive(
    xs::AbstractVector,
    values::AbstractVector,
    Lx::Float64,
    fit_degree::Int,
    boundary_value::Float64,
    boundary_slope::Float64,
)
    fit_degree = isodd(fit_degree) ? fit_degree - 1 : fit_degree
    fit_degree < 4 && error("The primitive fit degree must be at least 4 to enforce the endpoint constraints.")

    even_degrees = even_degree_indices(fit_degree)
    n_basis = length(even_degrees)

    A = zeros(Float64, length(xs), n_basis)
    for (i, x) in enumerate(xs)
        vals = chebyshev_values(x / Lx, fit_degree)
        A[i, :] .= vals[even_degrees .+ 1]
    end

    row_zero = chebyshev_values(0.0, fit_degree)[even_degrees .+ 1]
    row_boundary_value = ones(Float64, n_basis)
    row_boundary_slope = (even_degrees .^ 2) ./ Lx
    constraints = reduce(vcat, permutedims.([row_zero, row_boundary_value, row_boundary_slope]))
    targets = [0.0, boundary_value, boundary_slope]

    dep_cols = choose_stable_three_columns(constraints)
    free_cols = [j for j in 1:n_basis if j ∉ dep_cols]

    A_dep = constraints[:, dep_cols]
    A_free = constraints[:, free_cols]
    base = A_dep \ targets
    mult = isempty(free_cols) ? zeros(Float64, 3, 0) : (A_dep \ A_free)

    full_reduced = zeros(Float64, n_basis)
    if isempty(free_cols)
        full_reduced[dep_cols] .= base
    else
        W = Diagonal(sqrt.([primitive_fit_weight(x, Lx) for x in xs]))
        fit_matrix = W * (A[:, free_cols] - A[:, dep_cols] * mult)
        fit_rhs = W * (values - A[:, dep_cols] * base)
        free_coeffs = fit_matrix \ fit_rhs
        dep_coeffs = base - mult * free_coeffs
        full_reduced[dep_cols] .= dep_coeffs
        full_reduced[free_cols] .= free_coeffs
    end

    coeffs = zeros(Float64, fit_degree + 1)
    coeffs[even_degrees .+ 1] .= full_reduced
    return coeffs
end

function primitive_sample_grid(Lx::Float64, fit_degree::Int; sample_count::Int=DEFAULT_PRIMITIVE_SAMPLE_COUNT)
    fit_degree = isodd(fit_degree) ? fit_degree - 1 : fit_degree
    n_samples = max(sample_count, 3 * fit_degree + 1, 33)
    ts = collect(range(0.0, 1.0, length=n_samples))
    return Lx .* (1 .- (1 .- ts) .^ 2)
end

function fit_mode_primitive(
    mode_n::Int,
    source_poly,
    Lx::Float64,
    Ly::Float64,
    fit_degree::Int;
    sample_count::Int=DEFAULT_PRIMITIVE_SAMPLE_COUNT,
)
    p, dp, ddp, dddp = source_poly
    xs = primitive_sample_grid(Lx, fit_degree; sample_count=sample_count)
    values = [integrate_zero_to_x(t -> eval_integrand(mode_n, t, p, ddp, Ly), x) for x in xs]
    boundary_value = values[end]
    boundary_slope = eval_integrand_boundary_limit(mode_n, Lx, p, dp, ddp, dddp, Ly)
    coeffs = fit_even_chebyshev_primitive(xs, values, Lx, fit_degree, boundary_value, boundary_slope)
    return coeffs, (; xs, values, boundary_value, boundary_slope)
end

function build_source_coeffs(
    all_free_coeffs::AbstractVector,
    zero_constraint_base,
    zero_constraint_mult,
    constraint_bases,
    constraint_mults,
    n_modes::Int,
    params_per_mode::Int,
)
    zero_mode_params = size(zero_constraint_mult, 2)
    zero_mode_coeffs = get_full_coeffs(
        all_free_coeffs[1:zero_mode_params],
        zero_constraint_base,
        zero_constraint_mult,
    )

    positive_mode_coeffs = [
        get_full_coeffs(
            all_free_coeffs[zero_mode_params + (n - 1) * params_per_mode + 1:zero_mode_params + n * params_per_mode],
            constraint_bases[n],
            constraint_mults[n],
        ) for n in 1:n_modes
    ]

    return vcat([zero_mode_coeffs], positive_mode_coeffs)
end

function build_source_polys(source_coeffs::Vector{Vector{Float64}}, Lx::Float64)
    return [get_basis_funcs(coeffs, Lx) for coeffs in source_coeffs]
end

function primitive_amplitude(mode_n::Int, q_coeffs::Vector{Float64})
    return mode_n == 0 ? (-0.5 * pi) : (-0.5 * pi * q_coeffs[mode_n])
end

function build_fourier_chebyshev_coeffs(
    source_polys::AbstractVector,
    q_coeffs::Vector{Float64},
    Lx::Float64,
    Ly::Float64,
    fit_degree::Int;
    sample_count::Int=DEFAULT_PRIMITIVE_SAMPLE_COUNT,
)
    fit_degree = isodd(fit_degree) ? fit_degree - 1 : fit_degree
    n_modes = length(q_coeffs)
    length(source_polys) == n_modes + 1 || error("Expected $(n_modes + 1) source modes, got $(length(source_polys)).")

    coeffs = zeros(Float64, fit_degree + 1, n_modes + 1)
    for mode_n in 0:n_modes
        mode_coeffs, _ = fit_mode_primitive(mode_n, source_polys[mode_n + 1], Lx, Ly, fit_degree; sample_count=sample_count)
        coeffs[:, mode_n + 1] .= primitive_amplitude(mode_n, q_coeffs) .* mode_coeffs
    end

    return coeffs
end

function build_runtime_mode_polys(fourier_chebyshev_coeffs::AbstractMatrix, Lx::Float64)
    n_modes = size(fourier_chebyshev_coeffs, 2) - 1
    mode_polys = Vector{Tuple{ChebyshevApprox{Float64}, ChebyshevApprox{Float64}, ChebyshevApprox{Float64}}}(undef, n_modes + 1)

    for mode_n in 0:n_modes
        p = ChebyshevApprox(Vector{Float64}(fourier_chebyshev_coeffs[:, mode_n + 1]), (-Lx, Lx))
        dp = ∂(p)
        ddp = ∂(dp)
        mode_polys[mode_n + 1] = (p, dp, ddp)
    end

    return mode_polys
end

# ==========================================
# Physics Functions
# ==========================================

"""
Compute the base potential V₀(x, y) from the stored Fourier-Chebyshev expansion

    V₀(x, y) = ∑ₙ S̃ₙ(x) cos(nπy/Ly),

where `S̃ₙ` already includes the modal amplitude.
"""
function V₀(pot::PotentialFunctions, x::Real, y::Real)
    T = promote_type(typeof(x), typeof(y))
    val = zero(T)

    for (mode_n, poly) in enumerate(pot.mode_polys)
        p, _, _ = poly
        n = mode_n - 1
        val += p(x) * cos(n * pi * y / pot.data.Ly)
    end

    return val
end

function V(pot::PotentialFunctions, x::Real, y::Real)
    return (V₀(pot, x, y) / pot.data.V0_sup_norm) +
           0.5 * (pot.data.Mx_convex * x^2 + pot.data.My_convex * y^2)
end

function ∇V(pot::PotentialFunctions, x::Real, y::Real)
    V_total = xy -> V(pot, xy[1], xy[2])
    return ForwardDiff.gradient(V_total, [x, y])
end

function smooth_step(x)
    if x <= 0
        return 0.0
    elseif x >= 1
        return 1.0
    else
        f(t) = exp(-1.0 / t)
        return f(x) / (f(x) + f(1.0 - x))
    end
end

function smooth_max(x::Real, y::Real, strength::Real=10.0)
    max_val = max(x, y)
    min_val = min(x, y)
    return max_val + (1.0 / strength) * log1p(exp(strength * (min_val - max_val)))
end

function g(x::Real, y::Real)
    #y_min = 1 / π * acos((3.0 - 5.0 * sqrt(3.0)) / 12.0)
    y_min = 0.6
    return 2.0 * smooth_step(2.5 * x - 1.0) * max(0.0, abs(y) - y_min)^2
end

function V_wing(Lx::Real, x::Real, y::Real)
    Δx = abs(x) - Lx
    return 1e7 * (Δx + g(Δx / 5.0, y))
end

function ∇V_wing(Lx::Real, x::Real, y::Real)
    return ForwardDiff.gradient(xy -> V_wing(Lx, xy[1], xy[2]), SVector(x, y))
end

function V_extended(pot::PotentialFunctions, x::Real, y::Real)
    if abs(x) <= pot.data.Lx - 0.5
        return V(pot, x, y)
    elseif abs(x) <= pot.data.Lx + 0.5
        return max(V(pot, x, y), V(pot, pot.data.Lx, pot.data.Ly) + V_wing(pot.data.Lx, x, y))
    else
        return V(pot, pot.data.Lx, pot.data.Ly) + V_wing(pot.data.Lx, x, y)
    end
end

function ∇V_extended(pot::PotentialFunctions, x::Real, y::Real)
    V_total = xy -> V_extended(pot, xy[1], xy[2])
    return ForwardDiff.gradient(V_total, [x, y])
end

# ==========================================
# Hessian Evaluation
# ==========================================

function build_hessian_modes(mode_polys::Vector, Ly::Float64)
    return [HessianMode(i - 1, mode_polys[i], Ly) for i in eachindex(mode_polys)]
end

function eval_mode_x_data(mode::HessianMode, x::Real)
    p, dp, ddp = mode.poly
    return (; S = p(x), dS = dp(x), ddS = ddp(x))
end

function accumulate_mode_hessian(mode::HessianMode, x_data, y::Real)
    n_pi_over_Ly = mode.n * pi / mode.Ly
    cos_npy = cos(n_pi_over_Ly * y)
    sin_npy = sin(n_pi_over_Ly * y)

    return (
        H11 = cos_npy * x_data.ddS,
        H22 = (-(n_pi_over_Ly^2)) * cos_npy * x_data.S,
        H12 = (-n_pi_over_Ly) * sin_npy * x_data.dS,
        V0 = cos_npy * x_data.S,
    )
end

function precompute_mode_x_data(modes::AbstractVector{<:HessianMode}, x::Real)
    return [eval_mode_x_data(mode, x) for mode in modes]
end

function eval_unnormalized_Hessian_entries(
    modes::AbstractVector{<:HessianMode},
    x_data_per_mode,
    y::Real,
)
    H11, H22, H12, V0_xy = 0.0, 0.0, 0.0, 0.0

    @inbounds for i in eachindex(modes)
        contribution = accumulate_mode_hessian(modes[i], x_data_per_mode[i], y)
        H11 += contribution.H11
        H22 += contribution.H22
        H12 += contribution.H12
        V0_xy += contribution.V0
    end

    return H11, H22, H12, V0_xy
end

function eval_unnormalized_Hessian_point(x::Real, y::Real, modes::AbstractVector{<:HessianMode})
    x_data_per_mode = precompute_mode_x_data(modes, x)
    H11, H22, H12, V0_xy = eval_unnormalized_Hessian_entries(modes, x_data_per_mode, y)
    return [H11 H12; H12 H22], V0_xy
end

function eval_normalized_Hessian_grid(
    all_free_coeffs::AbstractVector,
    zero_constraint_base,
    zero_constraint_mult,
    constraint_bases,
    constraint_mults,
    n_modes::Int,
    params_per_mode::Int,
    q_coeffs::Vector{Float64},
    xs_coarse,
    ys_coarse,
    Lx::Float64,
    Ly::Float64,
    fit_degree::Int;
    normalize::Bool=true,
)
    source_coeffs = build_source_coeffs(
        all_free_coeffs,
        zero_constraint_base,
        zero_constraint_mult,
        constraint_bases,
        constraint_mults,
        n_modes,
        params_per_mode,
    )
    source_polys = build_source_polys(source_coeffs, Lx)
    fourier_chebyshev_coeffs = build_fourier_chebyshev_coeffs(source_polys, q_coeffs, Lx, Ly, fit_degree)
    modes = build_hessian_modes(build_runtime_mode_polys(fourier_chebyshev_coeffs, Lx), Ly)

    min_eig_global = Inf
    V0_sup_norm = normalize ? 0.0 : 1.0

    for x in xs_coarse
        x_data_per_mode = precompute_mode_x_data(modes, x)

        for y in ys_coarse
            H11, H22, H12, V0_xy = eval_unnormalized_Hessian_entries(modes, x_data_per_mode, y)

            if normalize
                V0_sup_norm = max(V0_sup_norm, abs(V0_xy))
            end

            trace_H = H11 + H22
            det_H = H11 * H22 - H12^2
            min_eig = 0.5 * (trace_H - sqrt(max(0.0, trace_H^2 - 4 * det_H)))
            min_eig_global = min(min_eig_global, min_eig)
        end
    end

    return V0_sup_norm < 1e-12 ? -Inf : (min_eig_global / V0_sup_norm)
end

function collect_normalized_hessian_samples(
    mode_polys::AbstractVector,
    V0_sup_norm::Float64,
    Ly::Float64,
    xs::AbstractVector,
    ys::AbstractVector,
)
    V0_sup_norm < 1e-12 && return Tuple{Float64, Float64, Float64}[]

    modes = build_hessian_modes(collect(mode_polys), Ly)
    samples = Tuple{Float64, Float64, Float64}[]

    for x in xs
        x_data_per_mode = precompute_mode_x_data(modes, x)
        for y in ys
            H11, H22, H12, _ = eval_unnormalized_Hessian_entries(modes, x_data_per_mode, y)
            push!(samples, (H11 / V0_sup_norm, H22 / V0_sup_norm, H12 / V0_sup_norm))
        end
    end

    return samples
end

function required_My_for_Mx(
    hessian_samples::AbstractVector{<:Tuple{Float64, Float64, Float64}},
    Mx::Float64;
    tol::Float64=1e-12,
)
    My = 0.0

    for (a, b, c) in hessian_samples
        ax = a + Mx
        if ax < -tol
            return Inf
        elseif ax <= tol
            if abs(c) > tol
                return Inf
            end
            My = max(My, -b)
        else
            My = max(My, (c^2 / ax) - b)
        end
    end

    return max(My, 0.0)
end

function convexification_weighted_cost(
    hessian_samples::AbstractVector{<:Tuple{Float64, Float64, Float64}},
    Mx::Float64,
    weight_x::Float64,
    weight_y::Float64,
)
    My = required_My_for_Mx(hessian_samples, Mx)
    return weight_x * Mx + weight_y * My
end

function solve_weighted_convexification(
    hessian_samples::AbstractVector{<:Tuple{Float64, Float64, Float64}};
    weight_x::Float64=1.0,
    weight_y::Float64=1.0,
)
    weight_x > 0 || error("`weight_x` must be positive.")
    weight_y > 0 || error("`weight_y` must be positive.")
    isempty(hessian_samples) && return (Mx=0.0, My=0.0, cost=0.0)

    lower = 0.0
    for (a, _, c) in hessian_samples
        candidate = -a + (abs(c) > 1e-12 ? 1e-12 : 0.0)
        lower = max(lower, candidate)
    end

    objective = Mx -> convexification_weighted_cost(hessian_samples, Mx, weight_x, weight_y)

    prev_x = lower
    prev_f = objective(prev_x)
    upper = max(lower + 1.0, 1.0)
    upper_f = objective(upper)

    for _ in 1:80
        if isfinite(upper_f) && upper_f >= prev_f
            break
        end
        prev_x = upper
        prev_f = upper_f
        upper = max(upper + 1.0, 2.0 * upper)
        upper_f = objective(upper)
    end

    isfinite(upper_f) || error("Unable to bracket the weighted convexification problem.")
    upper_f >= prev_f || error("Failed to bracket the weighted convexification minimum.")

    res = optimize(objective, lower, upper)
    Mx = Optim.minimizer(res)
    My = required_My_for_Mx(hessian_samples, Mx)
    return (Mx=Mx, My=My, cost=weight_x * Mx + weight_y * My)
end

# ==========================================
# Optimization Engine
# ==========================================

function eval_coefficient_penalty(
    all_free_coeffs::AbstractVector,
    zero_constraint_base,
    zero_constraint_mult,
    constraint_bases,
    constraint_mults,
    n_modes::Int,
    params_per_mode::Int,
)
    zero_mode_params = size(zero_constraint_mult, 2)
    zero_mode_coeffs = get_full_coeffs(
        all_free_coeffs[1:zero_mode_params],
        zero_constraint_base,
        zero_constraint_mult,
    )

    penalty = sum(abs2, zero_mode_coeffs)
    for n in 1:n_modes
        coeffs = get_full_coeffs(
            all_free_coeffs[zero_mode_params + (n - 1) * params_per_mode + 1:zero_mode_params + n * params_per_mode],
            constraint_bases[n],
            constraint_mults[n],
        )
        penalty += sum(abs2, coeffs)
    end

    return penalty
end

function run_optimization(;
    Lx::Float64=DEFAULT_Lx,
    Ly::Float64=DEFAULT_Ly,
    q_coeffs::Vector{Float64}=DEFAULT_q_coeffs,
    J_HARMONICS::Int=DEFAULT_J_HARMONICS,
    M::Int=default_fourier_chebyshev_degree(J_HARMONICS),
    coefficient_penalty_weight::Float64=DEFAULT_COEFFICIENT_PENALTY_WEIGHT,
)
    n_modes = length(q_coeffs)
    params_per_mode = J_HARMONICS - 2
    zero_mode_params = J_HARMONICS - 2

    xs_coarse = range(0.0, Lx - 1e-4, length=32)
    ys_coarse = range(0.0, Ly, length=32)

    zero_constraint_base, zero_constraint_mult = build_zero_mode_constraint_operators(J_HARMONICS; Lx=Lx)
    constraint_data = [build_constraint_operators(n, J_HARMONICS; Lx=Lx, Ly=Ly) for n in 1:n_modes]
    constraint_bases = [d[1] for d in constraint_data]
    constraint_mults = [d[2] for d in constraint_data]

    objective = (x) -> begin
        hessian_term = -eval_normalized_Hessian_grid(
            x,
            zero_constraint_base,
            zero_constraint_mult,
            constraint_bases,
            constraint_mults,
            n_modes,
            params_per_mode,
            q_coeffs,
            xs_coarse,
            ys_coarse,
            Lx,
            Ly,
            M;
            normalize=false,
        )
        coeff_penalty = coefficient_penalty_weight * eval_coefficient_penalty(
            x,
            zero_constraint_base,
            zero_constraint_mult,
            constraint_bases,
            constraint_mults,
            n_modes,
            params_per_mode,
        )
        return hessian_term + coeff_penalty
    end

    initial_guess = zeros(zero_mode_params + n_modes * params_per_mode)
    res = optimize(objective, initial_guess, NelderMead(), Optim.Options(iterations=1000, show_trace=false))

    best_coeffs = Optim.minimizer(res)
    penalized_objective = Optim.minimum(res)
    raw_min_hessian_eig = eval_normalized_Hessian_grid(
        best_coeffs,
        zero_constraint_base,
        zero_constraint_mult,
        constraint_bases,
        constraint_mults,
        n_modes,
        params_per_mode,
        q_coeffs,
        xs_coarse,
        ys_coarse,
        Lx,
        Ly,
        M;
        normalize=false,
    )

    source_coeffs = build_source_coeffs(
        best_coeffs,
        zero_constraint_base,
        zero_constraint_mult,
        constraint_bases,
        constraint_mults,
        n_modes,
        params_per_mode,
    )
    source_polys = build_source_polys(source_coeffs, Lx)
    fourier_chebyshev_coeffs = build_fourier_chebyshev_coeffs(source_polys, q_coeffs, Lx, Ly, M)

    return penalized_objective, raw_min_hessian_eig, source_polys, fourier_chebyshev_coeffs
end

# ==========================================
# Potential Generation
# ==========================================

"""
    build_potential_functions(data; source_polys=nothing)

Build a potential runtime from generated coefficient data.
"""
function build_potential_functions(data::PotentialData; source_polys=nothing)
    mode_polys = build_runtime_mode_polys(data.fourier_chebyshev_coeffs, data.Lx)
    return PotentialFunctions(data, mode_polys, source_polys)
end

"""
    generate_potential(; kwargs...)

Run the optimization and return the generated potential functions directly.

# Keyword Arguments
- `Lx::Float64`: Domain half-width in x (default: π/2)
- `Ly::Float64`: Domain half-width in y (default: 1.0)
- `q_coeffs::Vector{Float64}`: Mode coefficients (default: [0.5, 0.3, -0.2])
- `J_HARMONICS::Int`: Number of harmonics (default: 7)
- `M::Int`: Final Chebyshev degree for the stored primitive approximation

# Returns
- `PotentialFunctions`: Metadata plus reconstructed polynomial runtime data
"""
function generate_potential(;
    Lx::Float64=DEFAULT_Lx,
    Ly::Float64=DEFAULT_Ly,
    q_coeffs::Vector{Float64}=DEFAULT_q_coeffs,
    J_HARMONICS::Int=DEFAULT_J_HARMONICS,
    M::Int=default_fourier_chebyshev_degree(J_HARMONICS),
    convexification_weight_x::Float64=1.0,
    convexification_weight_y::Float64=1.0,
)
    println("Running optimization...")
    penalized_objective, raw_min_hessian_eig, source_polys, fourier_chebyshev_coeffs = run_optimization(
        Lx=Lx,
        Ly=Ly,
        q_coeffs=q_coeffs,
        J_HARMONICS=J_HARMONICS,
        M=M,
    )
    println("Optimization finished. Penalized objective: ", penalized_objective)

    tmp_data = PotentialData(Lx, Ly, q_coeffs, J_HARMONICS, 0.0, fourier_chebyshev_coeffs, 1.0, 0.0, 0.0, 0.0)
    tmp_pot = build_potential_functions(tmp_data; source_polys=source_polys)

    xs_fine = range(-Lx, Lx, length=100)
    ys_fine = range(-Ly, Ly, length=100)
    V0_sup_norm = maximum(abs(V₀(tmp_pot, x, y)) for x in xs_fine, y in ys_fine)
    println("Sup norm of V₀: ", V0_sup_norm)

    optimal_val = V0_sup_norm < 1e-12 ? -Inf : (raw_min_hessian_eig / V0_sup_norm)
    println("Unpenalized normalized min Hessian eigenvalue: ", optimal_val)

    xs_convex = range(0.0, Lx - 1e-4, length=32)
    ys_convex = range(0.0, Ly, length=32)
    hessian_samples = collect_normalized_hessian_samples(tmp_pot.mode_polys, V0_sup_norm, Ly, xs_convex, ys_convex)
    convexification = solve_weighted_convexification(
        hessian_samples;
        weight_x=convexification_weight_x,
        weight_y=convexification_weight_y,
    )
    println("Weighted convexification: Mx = ", convexification.Mx, ", My = ", convexification.My,
        ", cost = ", convexification.cost)

    M_convex = max(convexification.Mx, convexification.My)
    data = PotentialData(
        Lx,
        Ly,
        q_coeffs,
        J_HARMONICS,
        optimal_val,
        fourier_chebyshev_coeffs,
        V0_sup_norm,
        convexification.Mx,
        convexification.My,
        M_convex,
    )

    return build_potential_functions(data; source_polys=source_polys)
end

# ==========================================
# Standalone execution
# ==========================================

function main()
    generate_potential()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
