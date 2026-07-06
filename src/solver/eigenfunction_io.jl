if !isdefined(@__MODULE__, :Geometry)
    include("solver.jl")
end

using Serialization
using IntervalArithmetic

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
    coefficients::AbstractVector{T},
    λ::T,
    n_modes::Tuple{Int,Int},
    d::D,
    diam_x::T,
    diam_y::T;
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
    coefficients::AbstractVector{<:Real},
    λ::Real,
    n_modes::Tuple{Int,Int},
    d::Real,
    diam_x::Real,
    diam_y::Real;
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

function value_gradient(fit::FittedEigenfunction{T}, x, y) where {T}
    !(fit.d isa Interval) && isinf(fit.d) ||
        throw(ArgumentError("value_gradient currently supports only d = Inf fitted eigenfunctions."))

    mx, my = fit.n_modes
    c(v) = T <: Interval ? interval(v) : T(v)
    pi_T = c(π)
    half = c(0.5)

    Kx = [c(2p - 1) * pi_T / fit.diam_x for p in 1:mx]
    Ky = [c(2q - 2) * pi_T / fit.diam_y for q in 1:my]
    inv_norm = [
        inv(sqrt((half * fit.diam_x) * (q == 1 ? fit.diam_y : half * fit.diam_y)))
        for q in 1:my
    ]
    W = reshape(fit.coefficients, mx, my) .* inv_norm'

    sx = [sin(Kx[p] * x) for p in 1:mx]
    cx = [cos(Kx[p] * x) for p in 1:mx]
    sy = [sin(Ky[q] * y) for q in 1:my]
    cy = [cos(Ky[q] * y) for q in 1:my]

    local u, ux, uy
    for q in 1:my
        Sp = W[1, q] * sx[1]
        SPx = W[1, q] * Kx[1] * cx[1]
        for p in 2:mx
            Sp += W[p, q] * sx[p]
            SPx += W[p, q] * Kx[p] * cx[p]
        end

        term_u = cy[q] * Sp
        term_ux = cy[q] * SPx
        term_uy = -Ky[q] * sy[q] * Sp
        if q == 1
            u, ux, uy = term_u, term_ux, term_uy
        else
            u += term_u
            ux += term_ux
            uy += term_uy
        end
    end

    return u, ux, uy
end

function u(fit::FittedEigenfunction, x, y, r)
    if !(fit.d isa Interval) && isinf(fit.d)
        return value_gradient(fit, x, y)[1]
    end

    λx, λy, λr = fitted_eigenvalues(fit)
    return u(fit.d, fit.coefficients, λx, λy, λr, fit.diam_x, fit.diam_y, x, y, r)
end

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
