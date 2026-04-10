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

Metadata and coefficients for a generated potential.
"""
struct PotentialData
    Lx::Float64
    Ly::Float64
    q_coeffs::Vector{Float64}
    J_HARMONICS::Int
    optimal_val::Float64
    chebyshev_coeffs::Vector{Vector{Float64}}  # Full coeffs per mode; may include a leading zeroth mode
    V0_sup_norm::Float64
    M_convex::Float64
end

"""
    PotentialFunctions

Container for generated potential data and reconstructed polynomial modes.
"""
struct PotentialFunctions{P}
    data::PotentialData
    polys::Vector{P}
end

# ==========================================
# Constants (defaults)
# ==========================================
const DEFAULT_Lx = 0.5 * pi
const DEFAULT_Ly = 1.0
const DEFAULT_q_coeffs = [0.5, 0.3, -0.2]
const DEFAULT_J_HARMONICS = 7
const DEFAULT_COEFFICIENT_PENALTY_WEIGHT = 5e-5

# ==========================================
# Constraint/Polynomial Building
# ==========================================

"""
Build constraint operators for mode n.

fₙ(x) = Σ_j a_(n, j) T_(2j+1)(2x / π) with constraints:
1. ∑_j a_(n, j) = 1
2. fₙ'(π/2) = 0 ⇒ ∑_j a_(n, j) (2j+1)² = n²π² - 1
3. fₙ''(π/2) = (n²π²-1)fₙ(π/2) ⇒ ∑_j a_(n, j) (2j+1)²((2j+1)² - 1) = (n²π² - 1) * ∑_j a_(n, j)

this way, β₀(x,y) = ∑ₙ qₙ/fₙ(π/2) * fₙ(x) * cos(nπy) satisfies the necessary boundary conditions and PDE constraints at x=π/2.

We solve for the first 3 coefficients in terms of the remaining free coefficients.
"""
function build_constraint_operators(n::Int, J_HARMONICS::Int)
    s = 1.0 / DEFAULT_Lx
    M = zeros(3, J_HARMONICS + 1)

    for j in 0:J_HARMONICS
        k = 2j + 1
        M[1, j+1] = 1.0                              # Tₖ(π/2)
        M[2, j+1] = k^2 * s                          # Tₖ'(π/2) 
        M[3, j+1] = (k^2 * (k^2 - 1) / 3.0) * s^2    # Tₖ''(π/2) 
    end

    A = M[:, 1:3]
    B = M[:, 4:end]

    target = [1.0, 0.0, (n * pi)^2 - 1.0]
    return A \ target, A \ B
end

"""
Compute
row[j] = ∫_{-Lx}^{Lx} T_{2j+1}(2x/π) * sin(x) dx
"""
function zero_mode_orthogonality_row(J_HARMONICS::Int)
    row = zeros(Float64, J_HARMONICS + 1)

    for j in 1:(J_HARMONICS + 1)
        coeffs = zeros(Float64, J_HARMONICS + 1)
        coeffs[j] = 1.0
        p, _, _, _ = get_basis_funcs(coeffs)
        
        # Define the integrand and interval for Integrals.jl
        prob = IntegralProblem((x, p_params) -> p(x) * sin(x), -DEFAULT_Lx, DEFAULT_Lx)
        
        # Solve using adaptive Gauss-Kronrod quadrature
        row[j] = solve(prob, QuadGKJL(), reltol=1e-8, abstol=1e-8).u
    end

    return row
end

"""
Build constraint operators for the x-only zeroth mode.

The zeroth mode is not normalized to unit boundary trace. It satisfies:
1. f₀'(π/2) = 0
2. f₀''(π/2) = -f₀(π/2)
3. ⟨f₀, sin(x)⟩ = 0

which is the `n = 0` endpoint regularity condition while leaving the boundary
amplitude free.
"""
function build_zero_mode_constraint_operators(J_HARMONICS::Int)
    s = 1.0 / DEFAULT_Lx
    M = zeros(3, J_HARMONICS + 1)
    orth_row = zero_mode_orthogonality_row(J_HARMONICS)

    for j in 0:J_HARMONICS
        k = 2j + 1
        M[1, j+1] = k^2 * s                              # f₀'(π/2)
        M[2, j+1] = (k^2 * (k^2 - 1) / 3.0) * s^2 + 1.0  # f₀''(π/2) + f₀(π/2)
        M[3, j+1] = orth_row[j+1]                        # ⟨f₀, sin(x)⟩
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
    
    if n == 0 return zero(x) end
    if n == 1 return coeffs[1] * one(x) end

    # Clenshaw's recurrence
    # We maintain two intermediate values to avoid computing T_j explicitly
    bk1 = 0.0 # b_{k+1}
    bk2 = 0.0 # b_{k+2}
    
    for k in n:-1:2
        # b_k = c_k + 2u*b_{k+1} - b_{k+2}
        bk0 = coeffs[k] + 2u * bk1 - bk2
        bk2 = bk1
        bk1 = bk0
    end
    
    # Final result: c_0 + u*b_1 - b_2
    return coeffs[1] + u * bk1 - bk2
end

function ∂(p::ChebyshevApprox)
    a, b = p.domain
    # Chain rule: d/dx P(u(x)) = P'(u) * du/dx
    # scaling = u'(x) = 2 / (b - a)
    scaling = 2.0 / (b - a)
    
    n = length(p.coeffs)
    if n <= 1
        return ChebyshevApprox([0.0], p.domain)
    end

    # 1. Compute derivative coefficients in the u-domain [-1, 1]
    # Using the recurrence: c'_{n-1} = c'_{n+1} + 2n*c_n
    deriv_coeffs = zeros(eltype(p.coeffs), n - 1)
    
    c_next_plus = 0.0 # c'_{i+1}
    c_next = 0.0      # c'_i
    
    for i in (n-1):-1:1
        # 2i * c_i + c'_{i+1}
        val = 2 * i * p.coeffs[i+1] + c_next_plus
        deriv_coeffs[i] = val
        c_next_plus = c_next
        c_next = val
    end
    
    # 2. Correct the T0 term: the recurrence gives 2 * c'_0, so we halve it
    deriv_coeffs[1] /= 2.0
    
    # 3. Apply the chain rule scaling directly to the coefficients
    # This returns a new approximation valid on the same domain [a, b]
    return ChebyshevApprox(deriv_coeffs .* scaling, p.domain)
end

function get_basis_funcs(coeffs::AbstractVector)
    c_full = zeros(eltype(coeffs), length(coeffs) * 2)
    c_full[2:2:end] .= coeffs

    p = ChebyshevApprox(c_full, (-DEFAULT_Lx, DEFAULT_Lx))
    dp = ∂(p)
    ddp = ∂(dp)
    dddp = ∂(ddp)

    return (p, dp, ddp, dddp)
end

# ==========================================
# Integrand Evaluation
# ==========================================

"""
    eval_integrand(n, x, p, ddp)

Compute the modal integrand for the base potential V₀, representing
(-(Δ + 1)β₀,ₙ) / cos(x). For a mode β₀,ₙ(x,y) = fₙ(x)cos(nπy), this 
evaluates [(n²π² - 1)fₙ(x) - fₙ''(x)] / cos(x).
"""
function eval_integrand(n::Int, x::Real, p, ddp)
    K2 = n^2 * pi^2 - 1.0

    return (K2 * p(x) - ddp(x)) / cos(x)
end

function eval_integrand_derivative(n::Int, x::Real, p, dp, ddp, dddp)
    K2 = n^2 * pi^2 - 1.0

    N = K2 * p(x) - ddp(x)
    N_prime = K2 * dp(x) - dddp(x)

    cos_x = cos(x)
    return (N_prime * cos_x + N * sin(x)) / (cos_x^2)
end

# ==========================================
# Physics Functions
# ==========================================

"""
Compute the base potential V₀(x,y) as

V₀(x,y) = -∫₀ˣ (Δ+1)β₀(s,y)/cos(s) ds

where β₀(s,y) = Σₙ qₙ/fₙ(π/2) * fₙ(s) * cos(nπy).
"""
function V₀(pot::PotentialFunctions, x::Real, y::Real)
    n_modes = length(pot.data.q_coeffs)
    val = zero(x) * zero(y)
    zero_poly, positive_polys = split_polys(pot.polys, n_modes)

    if zero_poly !== nothing
        A0 = -0.5 * pi
        p, _, ddp, _ = zero_poly
        prob = IntegralProblem((s, p_params) -> x * eval_integrand(0, x * s, p, ddp), 0.0, 1.0)
        S0 = solve(prob, QuadGKJL()).u
        val += A0 * S0
    end

    for n in 1:n_modes
        An = -0.5 * pi * pot.data.q_coeffs[n]
        p, _, ddp, _ = positive_polys[n]
        # Change of variables: t = x * s, dt = x * ds (limits 0.0 to 1.0)
        prob = IntegralProblem((s, p_params) -> x * eval_integrand(n, x * s, p, ddp), 0.0, 1.0)
        Sn = solve(prob, QuadGKJL()).u
        val += An * Sn * cos(n * pi * y)
    end
    return val
end

function V(pot::PotentialFunctions, x::Real, y::Real)
    return (V₀(pot, x, y) / pot.data.V0_sup_norm) + 0.5 * pot.data.M_convex * (x^2 + y^2)
end

function ∇V(pot::PotentialFunctions, x::Real, y::Real)
    V_total = xy -> V(pot, xy[1], xy[2])
    return ForwardDiff.gradient(V_total, [x, y])
end

function smooth_step(x)
    # if x <= 0
    #     return 0.0
    # elseif x >= 1
    #     return 1.0
    # else
    #     f(t) = exp(-1.0 / t)
    #     return f(x) / (f(x) + f(1.0 - x))
    # end
    return max(0.0, x)^2
end

function g(x::Real, y::Real)
    # y_min = 1/π * acos((3.0 - 5.0 * sqrt(3.0)) / 12.0)
    # return 2.0 * s * smooth_step(0.5 * x - 1.0) * max(0.0, abs(y) - y_min)^2
    return 0.5 * max(0.0, x + 3*abs(y)^2 - 6.0)^2 + max(0.0, x - 4.0)^2
end

# function g(x::Real, y::Real, s::Real)
#     t0 = 4.0
#     c = 0.6
#     T = 2.0 * pi - 0.5 * pi
#     α = (1.0 - c) / (T - t0)

#     return 100.0 * s * max(0.0, abs(y) - 1.0 + α * max(0.0, x - t0))^2
# end

function V_wing(Lx::Real, x::Real, y::Real)
    return 1e7 * ((abs(x) - Lx) + g(abs(x) - Lx, y))
end

function ∇V_wing(Lx::Real, x::Real, y::Real)
    return ForwardDiff.gradient(xy -> V_wing(Lx, xy[1], xy[2]), SVector(x, y))
end

function V_extended(pot::PotentialFunctions, x::Real, y::Real)
    if abs(x) <= pot.data.Lx
        # Inside the core: exact evaluation
        return V(pot, x, y)
    else
        return V(pot, pot.data.Lx, 1.0) + V_wing(pot.data.Lx, x, y)
    end
end

function ∇V_extended(pot::PotentialFunctions, x::Real, y::Real)
    V_total = xy -> V_extended(pot, xy[1], xy[2])
    return ForwardDiff.gradient(V_total, [x, y])
end

# ==========================================
# Optimization Engine
# ==========================================

struct HessianMode{P}
    n::Int
    amplitude::Float64
    poly::P
end

"""
    build_hessian_modes(polys, q_coeffs, n_modes)

Build the zeroth and positive Fourier-Chebyshev modes used in Hessian evaluation.
"""
function build_hessian_modes(polys::Vector, q_coeffs::Vector{Float64}, n_modes::Int)
    zero_poly, positive_polys = split_polys(polys, n_modes)
    all_polys = zero_poly === nothing ? positive_polys : vcat([zero_poly], positive_polys)
    poly_type = eltype(all_polys)
    mode_count = length(all_polys)
    modes = Vector{HessianMode{poly_type}}(undef, mode_count)

    offset = zero_poly === nothing ? 0 : 1
    if zero_poly !== nothing
        modes[1] = HessianMode(0, -0.5 * pi, zero_poly)
    end

    for n in 1:n_modes
        modes[n + offset] = HessianMode(n, -0.5 * pi * q_coeffs[n], positive_polys[n])
    end

    return modes
end

"""
    eval_mode_x_data(mode, x)

Compute the x-dependent modal quantities `Sₙ(x)`, `Iₙ(x)`, and `Iₙ′(x)`.
"""
function eval_mode_x_data(mode::HessianMode, x::Real)
    p, dp, ddp, dddp = mode.poly
    prob = IntegralProblem((t, p_params) -> eval_integrand(mode.n, t, p, ddp), 0.0, x)
    S = solve(prob, QuadGKJL()).u
    I = eval_integrand(mode.n, x, p, ddp)
    I_prime = eval_integrand_derivative(mode.n, x, p, dp, ddp, dddp)
    return (; S, I, I_prime)
end

"""
    accumulate_mode_hessian(mode, x_data, y)

Return the contribution of a single mode to `V₀(x, y)` and its Hessian entries.
"""
function accumulate_mode_hessian(mode::HessianMode, x_data, y::Real)
    n_pi = mode.n * pi
    cos_npy = cos(n_pi * y)
    sin_npy = sin(n_pi * y)

    return (
        H11 = mode.amplitude * cos_npy * x_data.I_prime,
        H22 = mode.amplitude * (-(n_pi^2)) * cos_npy * x_data.S,
        H12 = mode.amplitude * (-n_pi) * sin_npy * x_data.I,
        V0 = mode.amplitude * x_data.S * cos_npy,
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

"""
    eval_unnormalized_Hessian_point(x, y, polys, q_coeffs, n_modes)

Compute the Hessian of the unnormalized base potential `V₀` at a single point
`(x, y)`, together with the scalar value `V₀(x, y)`.

The modal expansion used by this file is

    V₀(x, y) = A₀ S₀(x) + ∑ₙ₌₁ᴺ Aₙ Sₙ(x) cos(nπy),

with

    A₀ = -π/2,   Aₙ = -(π/2) qₙ,

and

    Sₙ(x) = ∫₀ˣ Iₙ(t) dt,
    Iₙ(x) = ((n²π² - 1) fₙ(x) - fₙʺ(x)) / cos(x).

Differentiating this separated form gives the Hessian entries

    ∂ₓₓV₀ = A₀ I₀'(x) + ∑ₙ₌₁ᴺ Aₙ Iₙ'(x) cos(nπy),

    ∂yyV₀ = ∑ₙ₌₁ᴺ Aₙ (-(nπ)²) Sₙ(x) cos(nπy),

    ∂xyV₀ = ∑ₙ₌₁ᴺ Aₙ (-nπ) sin(nπy) Iₙ(x).

For the `xx` entry, the derivative of the integrand is computed using

    Nₙ(x) = (n²π² - 1) fₙ(x) - fₙʺ(x),
    Iₙ(x) = Nₙ(x) / cos(x),

so that

    Iₙ'(x) = (Nₙ'(x) cos(x) + Nₙ(x) sin(x)) / cos²(x),
    Nₙ'(x) = (n²π² - 1) fₙ'(x) - fₙ‴(x).

The function returns

    ∇²V₀(x, y) = [∂ₓₓV₀  ∂xyV₀;
                  ∂xyV₀  ∂yyV₀]

along with `V₀(x, y)` itself, which is later used for normalization in
`eval_normalized_Hessian_grid`.
"""
function eval_unnormalized_Hessian_point(x::Real, y::Real, polys::Vector,
    q_coeffs::Vector{Float64}, n_modes::Int)
    return eval_unnormalized_Hessian_point(x, y, build_hessian_modes(polys, q_coeffs, n_modes))
end

"""
    eval_unnormalized_Hessian_point(x, y, modes)

Evaluate the unnormalized Hessian of `V₀` at `(x, y)` using prebuilt modes.
"""
function eval_unnormalized_Hessian_point(x::Real, y::Real, modes::AbstractVector{<:HessianMode})
    x_data_per_mode = precompute_mode_x_data(modes, x)
    H11, H22, H12, V0_xy = eval_unnormalized_Hessian_entries(modes, x_data_per_mode, y)

    return [H11 H12; H12 H22], V0_xy
end

function eval_normalized_Hessian_grid(all_free_coeffs::AbstractVector,
    zero_constraint_base, zero_constraint_mult,
    constraint_bases, constraint_mults,
    n_modes::Int, params_per_mode::Int,
    q_coeffs::Vector{Float64},
    xs_coarse, ys_coarse;
    normalize::Bool=true)

    zero_mode_params = size(zero_constraint_mult, 2)
    zero_poly = get_basis_funcs(get_full_coeffs(
        all_free_coeffs[1:zero_mode_params],
        zero_constraint_base,
        zero_constraint_mult
    ))

    positive_polys = [
        get_basis_funcs(get_full_coeffs(
            all_free_coeffs[zero_mode_params + (n-1)*params_per_mode+1:zero_mode_params + n*params_per_mode],
            constraint_bases[n],
            constraint_mults[n]
        )) for n in 1:n_modes
    ]
    modes = build_hessian_modes(vcat([zero_poly], positive_polys), q_coeffs, n_modes)
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

function eval_coefficient_penalty(all_free_coeffs::AbstractVector,
    zero_constraint_base, zero_constraint_mult,
    constraint_bases, constraint_mults,
    n_modes::Int, params_per_mode::Int)

    zero_mode_params = size(zero_constraint_mult, 2)
    zero_mode_coeffs = get_full_coeffs(
        all_free_coeffs[1:zero_mode_params],
        zero_constraint_base,
        zero_constraint_mult
    )

    penalty = sum(abs2, zero_mode_coeffs)
    for n in 1:n_modes
        coeffs = get_full_coeffs(
            all_free_coeffs[zero_mode_params + (n-1)*params_per_mode+1:zero_mode_params + n*params_per_mode],
            constraint_bases[n],
            constraint_mults[n]
        )
        penalty += sum(abs2, coeffs)
    end

    return penalty
end

function run_optimization(; Lx::Float64=DEFAULT_Lx, Ly::Float64=DEFAULT_Ly,
    q_coeffs::Vector{Float64}=DEFAULT_q_coeffs,
    J_HARMONICS::Int=DEFAULT_J_HARMONICS,
    coefficient_penalty_weight::Float64=DEFAULT_COEFFICIENT_PENALTY_WEIGHT)
    n_modes = length(q_coeffs)
    params_per_mode = J_HARMONICS - 2
    zero_mode_params = J_HARMONICS - 2

    xs_coarse = range(0.0, Lx - 1e-4, length=32)
    ys_coarse = range(0.0, Ly, length=32)

    zero_constraint_base, zero_constraint_mult = build_zero_mode_constraint_operators(J_HARMONICS)
    constraint_data = [build_constraint_operators(n, J_HARMONICS) for n in 1:n_modes]
    constraint_bases = [d[1] for d in constraint_data]
    constraint_mults = [d[2] for d in constraint_data]

    objective = (x) -> begin
        hessian_term = -eval_normalized_Hessian_grid(x, zero_constraint_base, zero_constraint_mult,
            constraint_bases, constraint_mults,
            n_modes, params_per_mode, q_coeffs,
            xs_coarse, ys_coarse, normalize=false)
        coeff_penalty = coefficient_penalty_weight * eval_coefficient_penalty(
            x, zero_constraint_base, zero_constraint_mult,
            constraint_bases, constraint_mults,
            n_modes, params_per_mode
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
        ys_coarse;
        normalize=false
    )

    # Store full Chebyshev coefficients for each mode, with the x-only zeroth mode first.
    zero_mode_coeffs = get_full_coeffs(
        best_coeffs[1:zero_mode_params],
        zero_constraint_base,
        zero_constraint_mult
    )
    positive_mode_coeffs = [
        get_full_coeffs(
            best_coeffs[zero_mode_params + (n-1)*params_per_mode+1:zero_mode_params + n*params_per_mode],
            constraint_bases[n],
            constraint_mults[n]
        ) for n in 1:n_modes
    ]
    chebyshev_coeffs = vcat([zero_mode_coeffs], positive_mode_coeffs)

    return penalized_objective, raw_min_hessian_eig, chebyshev_coeffs
end

# ==========================================
# Potential Generation
# ==========================================

"""
    build_potential_functions(data)

Build a potential runtime from generated coefficient data.
"""
function build_potential_functions(data::PotentialData)
    polys = [get_basis_funcs(coeffs) for coeffs in data.chebyshev_coeffs]
    return PotentialFunctions(data, polys)
end

"""
    generate_potential(; kwargs...)

Run the optimization and return the generated potential functions directly.

# Keyword Arguments
- `Lx::Float64`: Domain half-width in x (default: π/2)
- `Ly::Float64`: Domain half-width in y (default: 1.0)
- `q_coeffs::Vector{Float64}`: Mode coefficients (default: [0.5, 0.3, -0.2])
- `J_HARMONICS::Int`: Number of harmonics (default: 7)

# Returns
- `PotentialFunctions`: Metadata plus reconstructed polynomial runtime data
"""
function generate_potential(;
    Lx::Float64=DEFAULT_Lx,
    Ly::Float64=DEFAULT_Ly,
    q_coeffs::Vector{Float64}=DEFAULT_q_coeffs,
    J_HARMONICS::Int=DEFAULT_J_HARMONICS)
    println("Running optimization...")
    penalized_objective, raw_min_hessian_eig, chebyshev_coeffs = run_optimization(Lx=Lx, Ly=Ly, q_coeffs=q_coeffs, J_HARMONICS=J_HARMONICS)
    println("Optimization finished. Penalized objective: ", penalized_objective)

    # Reconstruct polynomials to compute V0_sup_norm
    tmp_data = PotentialData(Lx, Ly, q_coeffs, J_HARMONICS, 0.0,
        chebyshev_coeffs, 1.0, 0.0)
    tmp_pot = build_potential_functions(tmp_data)

    xs_fine = range(-Lx, Lx, length=100)
    ys_fine = range(-Ly, Ly, length=100)
    V0_sup_norm = maximum(abs(V₀(tmp_pot, x, y)) for x in xs_fine, y in ys_fine)
    println("Sup norm of V₀: ", V0_sup_norm)

    optimal_val = V0_sup_norm < 1e-12 ? -Inf : (raw_min_hessian_eig / V0_sup_norm)
    println("Unpenalized normalized min Hessian eigenvalue: ", optimal_val)

    M_convex = optimal_val < 0 ? ceil(abs(optimal_val)) : 0.0

    data = PotentialData(Lx, Ly, q_coeffs, J_HARMONICS, optimal_val,
        chebyshev_coeffs, V0_sup_norm, M_convex)

    return build_potential_functions(data)
end

# ==========================================
# Standalone execution
# ==========================================

"""
    main()

Generate the default potential when running this file directly.
"""
function main()
    generate_potential()
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
