# The MPS function: its basis, the fitted representation, evaluation, gradient, and IO.
#
# Layering: this module is self-contained (only depends on the Bessel primitives in lambda.jl).
# `solver.jl` is a *consumer* — it uses the basis here to assemble/solve and returns a
# FittedEigenfunction. Evaluation lives here, not in the solver.
#
# The MPS ansatz is  u(x, y, r) = ∑ cₙ · axial_basisₙ(x, y) · ϕₙ(r),  with ϕₙ(1) = 1.
# The axial part is a tensor product of sin (x) and cos (y) modes; the radial part ϕ is a
# Gaussian at d = ∞ and a (modified) Bessel profile at finite d.
#
# Interval arithmetic is supported at d = ∞ only. The generalized `axial_basis` and the
# d == Inf branch of `ϕ` accept intervals; the finite-d Bessel path stays Float64-only.

module MPSFunction

export FittedEigenfunction, InfiniteEvaluator, axial_basis, axial_basis_tables, fitted_eigenvalues,
    get_eigenvalues, intervalize, load_fitted_eigenfunction, prepare_u, save_fitted_eigenfunction,
    u, value_gradient, ϕ

include("lambda.jl")

using Serialization
using IntervalArithmetic

# The literal `v` as a value of the same numeric type as the scalar `x`. For intervals this returns
# a guaranteed interval — `x / 2` with a bare literal would drop the `isguaranteed` flag — and for
# Float64 it is just the exact `oftype(x, v)`.
_lit(x, v) = x isa Interval ? interval(v) : oftype(x, v)

# ---------------------------------------------------------------------------------------------
# Basis functions (the MPS ansatz)
# ---------------------------------------------------------------------------------------------

function ϕ(d, radial_eigenvalue, r)
    if d == Inf
        a = -radial_eigenvalue / _lit(radial_eigenvalue, 4)
        val_normalized = exp.(a .* (r .^ 2 .- _lit(radial_eigenvalue, 1)) ./ _lit(radial_eigenvalue, 2))
        grad_normalized = a .* r .* val_normalized
        return val_normalized, grad_normalized
    end

    # finite d: (modified) Bessel profile, Float64 only
    order = (d - 1) / 2
    k = sqrt(d * abs(radial_eigenvalue)) / 2

    val, grad = radial_eigenvalue < 0 ? Λ_i(order, k .* r) : Λ_j(order, k .* r)
    normalization, _ = radial_eigenvalue < 0 ? Λ_i(order, k) : Λ_j(order, k)

    return val ./ normalization, k .* grad ./ normalization
end

function axial_basis(λx, λy, diam_x, diam_y, x, y)
    kx = sqrt(λx)
    ky = sqrt(λy)

    two = _lit(diam_x, 2)
    norm_x_sq = diam_x / two
    norm_y_sq = iszero(λy) ? diam_y : diam_y / two
    inv_norm = inv(sqrt(norm_x_sq * norm_y_sq))

    val = sin.(kx * x) .* cos.(ky * y) .* inv_norm
    grad = (
        kx .* cos.(kx * x) .* cos.(ky * y) .* inv_norm,
        -ky .* sin.(kx * x) .* sin.(ky * y) .* inv_norm,
    )

    return val, grad
end

function axial_basis_tables(
    xs::AbstractVector{T}, ys::AbstractVector{T},
    n_modes::Tuple{Int, Int},
    diam_x::T, diam_y::T,
) where {T<:AbstractFloat}
    mx, my = n_modes

    Kx = [T(2p - 1) * (T(π) / diam_x) for p in 1:mx]
    Ky = [T(2q - 2) * (T(π) / diam_y) for q in 1:my]

    λx_modes = Kx .^ 2
    λy_modes = Ky .^ 2

    norm_x_sq = T(0.5) * diam_x
    norm_y_sq = [
        iszero(Ky[q]) ? diam_y : diam_y / T(2.0)
        for q in 1:my
    ]
    inv_norms = inv.(sqrt.(norm_x_sq .* norm_y_sq))

    Sx = sin.(xs .* Kx')
    Cx = cos.(xs .* Kx')
    Sy = sin.(ys .* Ky')
    Cy = cos.(ys .* Ky')

    return (; Kx, Ky, λx_modes, λy_modes, inv_norms, Sx, Cx, Sy, Cy)
end

@inline function axial_basis(tables, ix, iy, p, q)
    sx = tables.Sx[ix, p]
    cx = tables.Cx[ix, p]
    sy = tables.Sy[iy, q]
    cy = tables.Cy[iy, q]

    kx = tables.Kx[p]
    ky = tables.Ky[q]
    inv_norm = tables.inv_norms[q]

    av = sx * cy * inv_norm
    agx = kx * cx * cy * inv_norm
    agy = -ky * sx * sy * inv_norm

    return av, (agx, agy)
end

# ---------------------------------------------------------------------------------------------
# Mode eigenvalues of the ansatz
# ---------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------
# Point evaluation of a mode expansion (raw arrays)
# ---------------------------------------------------------------------------------------------

# d = ∞ (radial Gaussian). Generic in the element type: works for Float64 and for intervals.
function u(d::Nothing, coefficients, λx, λy, λr, diam_x, diam_y, x, y, r)
    return sum(
        coefficients[i] *
        first(axial_basis(λx[i], λy[i], diam_x, diam_y, x, y)) *
        first(ϕ(Inf, λr[i], r))
        for i in eachindex(coefficients)
    )
end

# finite d (radial Bessel), Float64 only
function u(
    d::T,
    coefficients::AbstractVector{T},
    λx::AbstractVector{T},
    λy::AbstractVector{T},
    λr::AbstractVector{T},
    diam_x::T, diam_y::T,
    x::T, y::T, r::T
) where {T <: AbstractFloat}
    val = zero(T)
    for i in eachindex(coefficients)
        av, _ = axial_basis(λx[i], λy[i], diam_x, diam_y, x, y)
        rv, _ = ϕ(d, λr[i], r)
        val += coefficients[i] * av * rv
    end
    return val
end

# u without precomputed eigenvalues, avoid if u is computed many times for the same mode expansion.
function u(
    d::T,
    diam_x::T, diam_y::T,
    coefficients::AbstractVector{T},
    λ::T,
    n_modes::Tuple{Int,Int},
    x::T, y::T, r::T
) where {T <: AbstractFloat}
    λx, λy, λr = get_eigenvalues(diam_x, diam_y, n_modes, λ)
    return u(d, coefficients, λx, λy, λr, diam_x, diam_y, x, y, r)
end

# ---------------------------------------------------------------------------------------------
# FittedEigenfunction: the persisted, self-describing MPS function
# ---------------------------------------------------------------------------------------------

struct FittedEigenfunction{T,D}
    coefficients::Vector{T}
    λ::T
    n_modes::Tuple{Int,Int}
    d::D
    diam_x::T
    diam_y::T
    metadata::Dict{String,Any}
end

function FittedEigenfunction(
    coefficients::AbstractVector{T}, λ::T,
    n_modes::Tuple{Int,Int}, d::D,
    diam_x::T, diam_y::T;
    metadata=Dict{String,Any}()
) where {T<:Real,D<:Real}
    return FittedEigenfunction{T,D}(
        Vector{T}(coefficients),
        λ,
        n_modes,
        d,
        diam_x,
        diam_y,
        Dict{String,Any}(string(k) => v for (k, v) in pairs(metadata)),
    )
end

function FittedEigenfunction(
    coefficients::AbstractVector{<:Real}, λ::Real,
    n_modes::Tuple{Int,Int}, d::Real,
    diam_x::Real, diam_y::Real;
    metadata=Dict{String,Any}()
)
    return FittedEigenfunction(
        Float64.(coefficients),
        Float64(λ),
        n_modes,
        Float64(d),
        Float64(diam_x),
        Float64(diam_y);
        metadata,
    )
end

function intervalize(fit::FittedEigenfunction)
    d = fit.d isa Interval ? fit.d : (isinf(fit.d) ? fit.d : interval(fit.d))
    return FittedEigenfunction(
        interval.(fit.coefficients),
        interval(fit.λ),
        fit.n_modes,
        d,
        interval(fit.diam_x),
        interval(fit.diam_y);
        metadata=fit.metadata,
    )
end

function fitted_eigenvalues(fit::FittedEigenfunction{T}) where {T}
    mx, my = fit.n_modes
    c(x) = T <: Interval ? interval(x) : T(x)
    pi_T = c(π)
    Kx = [c(2p - 1) * pi_T / fit.diam_x for p in 1:mx]
    Ky = [c(2q - 2) * pi_T / fit.diam_y for q in 1:my]
    λx = [kx * kx for _ in Ky for kx in Kx]
    λy = [ky * ky for ky in Ky for _ in Kx]
    return λx, λy, fit.λ .- (λx .+ λy)
end

# value and gradient of u at r = 1, for d = ∞ only.
function value_gradient(fit::FittedEigenfunction{T}, x, y) where {T}
    !(fit.d isa Interval) && isinf(fit.d) ||
        throw(ArgumentError("value_gradient currently supports only d = Inf fitted eigenfunctions."))

    λx, λy, _ = fitted_eigenvalues(fit)

    local u, ux, uy
    for i in eachindex(fit.coefficients)
        c = fit.coefficients[i]
        av, (agx, agy) = axial_basis(λx[i], λy[i], fit.diam_x, fit.diam_y, x, y)
        if i == 1
            u, ux, uy = c * av, c * agx, c * agy
        else
            u += c * av
            ux += c * agx
            uy += c * agy
        end
    end

    return u, ux, uy
end

# Point-independent data for repeated evaluation of a d = ∞ fitted eigenfunction.
struct InfiniteEvaluator{T}
    weights::Matrix{T}
    kx::Vector{T}
    ky::Vector{T}
    kx_squared::Vector{T}
    ky_squared::Vector{T}
    λ::T
end

"""
    prepare_u(fit::FittedEigenfunction)

Precompute the point-independent mode data for repeated evaluation of a fitted
eigenfunction at `d = Inf`. Evaluators prepared from floating-point and interval
fits are evaluated through the same `u(evaluator, x, y, r)` method.
"""
function prepare_u(fit::FittedEigenfunction{T}) where {T<:Union{AbstractFloat,Interval}}
    !(fit.d isa Interval) && isinf(fit.d) ||
        throw(ArgumentError("prepare_u currently supports only d = Inf"))

    mx, my = fit.n_modes
    mx > 0 && my > 0 || throw(ArgumentError("n_modes must be positive"))
    expected = mx * my
    length(fit.coefficients) == expected ||
        throw(DimensionMismatch("expected $expected coefficients, got $(length(fit.coefficients))"))

    half = _lit(fit.λ, 0.5)
    pi_T = _lit(fit.λ, π)

    kx = [_lit(fit.λ, 2p - 1) * pi_T / fit.diam_x for p in 1:mx]
    ky = [_lit(fit.λ, 2q - 2) * pi_T / fit.diam_y for q in 1:my]
    kx_squared = kx .^ 2
    ky_squared = ky .^ 2

    norm_x_squared = half * fit.diam_x
    weights = Matrix{T}(undef, mx, my)
    @inbounds for q in 1:my
        norm_y_squared = q == 1 ? fit.diam_y : half * fit.diam_y
        inv_norm = inv(sqrt(norm_x_squared * norm_y_squared))
        for p in 1:mx
            weights[p, q] = fit.coefficients[p + (q - 1) * mx] * inv_norm
        end
    end

    return InfiniteEvaluator(weights, kx, ky, kx_squared, ky_squared, fit.λ)
end

function u(evaluator::InfiniteEvaluator{T}, x, y, r) where {T}
    radial_argument = _lit(evaluator.λ, 0.125) * (r^2 - _lit(evaluator.λ, 1.0))

    # Reuse the x-dependent factors for every y mode. Keeping only this one
    # temporary vector avoids repeated transcendental evaluations and a second allocation.
    x_terms = Vector{T}(undef, length(evaluator.kx))
    @inbounds for p in eachindex(evaluator.kx)
        x_terms[p] = sin(evaluator.kx[p] * x) *
                     exp(evaluator.kx_squared[p] * radial_argument)
    end

    value = _lit(evaluator.λ, 0.0)
    # ky[1] is exactly zero, so its y-dependent factor is exactly one.
    @inbounds for p in eachindex(evaluator.kx)
        value += evaluator.weights[p, 1] * x_terms[p]
    end
    @inbounds for q in 2:length(evaluator.ky)
        y_term = cos(evaluator.ky[q] * y) *
                 exp(evaluator.ky_squared[q] * radial_argument)
        for p in eachindex(evaluator.kx)
            value += evaluator.weights[p, q] * x_terms[p] * y_term
        end
    end
    return exp(-evaluator.λ * radial_argument) * value
end

function u(fit::FittedEigenfunction{T}, x, y, r) where {T<:AbstractFloat}
    if isinf(fit.d)
        return u(prepare_u(fit), T(x), T(y), T(r))
    end

    λx, λy, λr = fitted_eigenvalues(fit)
    return u(fit.d, fit.coefficients,
        λx, λy, λr,
        fit.diam_x, fit.diam_y,
        T(x), T(y), T(r),
    )
end

function u(fit::FittedEigenfunction{T}, x, y, r) where {T<:Interval}
    !(fit.d isa Interval) && isinf(fit.d) ||
        throw(ArgumentError("interval evaluation currently supports only d = Inf"))
    return u(prepare_u(fit), interval(x), interval(y), interval(r))
end

# ---------------------------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------------------------

function save_fitted_eigenfunction(path::AbstractString, fit::FittedEigenfunction)
    open(path, "w") do io
        serialize(io, (
            coefficients=fit.coefficients,
            λ=fit.λ,
            n_modes=fit.n_modes,
            d=fit.d,
            diam_x=fit.diam_x,
            diam_y=fit.diam_y,
            metadata=fit.metadata,
        ))
    end
    return path
end

function load_fitted_eigenfunction(path::AbstractString; intervalized::Bool=false)
    payload = try
        open(deserialize, path)
    catch
        error(
            "Could not deserialize fitted eigenfunction checkpoint at $path. " *
            "Regenerate it with save_fitted_eigenfunction to use the current payload format."
        )
    end

    fit = payload isa FittedEigenfunction ? payload : FittedEigenfunction(
        payload.coefficients,
        payload.λ,
        payload.n_modes,
        payload.d,
        payload.diam_x,
        payload.diam_y;
        metadata=payload.metadata,
    )
    return intervalized ? intervalize(fit) : fit
end

end
