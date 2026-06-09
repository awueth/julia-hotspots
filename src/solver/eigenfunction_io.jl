if !isdefined(@__MODULE__, :Geometry)
    include("solver.jl")
end

using Serialization

struct FittedEigenfunction
    coefficients::Vector{Float64}
    λ::Float64
    n_modes::Tuple{Int,Int}
    d::Float64
    diam_x::Float64
    diam_y::Float64
    metadata::Dict{String,Any}
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
        Float64(diam_y),
        Dict{String,Any}(string(k) => v for (k, v) in pairs(metadata))
    )
end

function fitted_eigenvalues(fit::FittedEigenfunction)
    return get_eigenvalues(fit.diam_x, fit.diam_y, fit.n_modes, fit.λ)
end

function u(fit::FittedEigenfunction, x::Float64, y::Float64, r::Float64)
    λx, λy, λr = fitted_eigenvalues(fit)
    return u(fit.d, fit.coefficients, λx, λy, λr, x, y, r)
end

function save_fitted_eigenfunction(path::AbstractString, fit::FittedEigenfunction)
    open(path, "w") do io
        serialize(io, fit)
    end
    return path
end

function load_fitted_eigenfunction(path::AbstractString)
    fit = open(deserialize, path)
    fit isa FittedEigenfunction || error("Expected FittedEigenfunction in $path, got $(typeof(fit))")
    return fit
end
