if !isdefined(@__MODULE__, :PotentialGenerator)
    include("potential_generator.jl")
end
if !isdefined(@__MODULE__, :LSEModel)
    include("lse_regression.jl")
end

module PotentialInterface

using ForwardDiff

using ..PotentialGenerator

const _lse_predict = getfield(parentmodule(@__MODULE__), :predict)
const _lse_gradient = getfield(parentmodule(@__MODULE__), :gradient)
const _load_lse_model = getfield(parentmodule(@__MODULE__), :load_lse_model)
const _LSEModel = getfield(parentmodule(@__MODULE__), :LSEModel)

export AbstractCorePotential, AbstractWingPotential, AbstractPotential
export SmoothMaxPotential
export GeneratedCorePotential
export LSECorePotential, load_lse_core_potential
export HandmadeWingPotential, NonConvexWingPotential
export LSEWingPotential, load_lse_wing_potential
export JoinedLSEPotential, join_lse_potentials
export core_value, core_gradient, wing_value, wing_gradient
export potential_value, potential_gradient, potential_domain, potential_functions

abstract type AbstractCorePotential end
abstract type AbstractWingPotential end
abstract type AbstractPotential end

"""
    core_value(core, x, y)

Evaluate an unscaled core potential at `(x, y)`.
"""
function core_value end

"""
    core_gradient(core, x, y)

Evaluate the unscaled core potential gradient at `(x, y)`, returned as `(gx, gy)`.
"""
function core_gradient end

"""
    wing_value(wing, x, y)

Evaluate an unscaled wing potential at `(x, y)`.
"""
function wing_value end

"""
    wing_gradient(wing, x, y)

Evaluate the unscaled wing potential gradient at `(x, y)`, returned as `(gx, gy)`.
"""
function wing_gradient end

"""
    potential_value(p, x, y)

Evaluate the unscaled solver-facing potential at `(x, y)`.
"""
function potential_value end

"""
    potential_gradient(p, x, y)

Evaluate the unscaled solver-facing potential gradient at `(x, y)`, returned as `(gx, gy)`.
"""
function potential_gradient end

"""
    potential_domain(p)

Return domain metadata as a named tuple with at least `Lx` and `Ly`.
"""
function potential_domain end

function _gradient_tuple(g)
    return (g[1], g[2])
end

function _smooth_max(x::Real, y::Real, strength::Real=10.0)
    max_val = max(x, y)
    min_val = min(x, y)
    return max_val + (1.0 / strength) * log1p(exp(strength * (min_val - max_val)))
end

"""
    potential_functions(p; scale=1.0)

Return solver-ready closures `(V, gradV)`. Scaling is applied here so stored
potential objects always represent the unscaled potential.
"""
function potential_functions(p::AbstractPotential; scale::Real=1.0)
    scale_value = Float64(scale)
    V = (x, y) -> scale_value * potential_value(p, x, y)
    gradV = (x, y) -> begin
        gx, gy = potential_gradient(p, x, y)
        return (scale_value * gx, scale_value * gy)
    end
    return V, gradV
end

struct SmoothMaxPotential{C<:AbstractCorePotential,W<:AbstractWingPotential} <: AbstractPotential
    core::C
    wing::W
    smooth_max_strength::Float64
end

function SmoothMaxPotential(
    core::C,
    wing::W;
    smooth_max_strength::Real=10.0,
) where {C<:AbstractCorePotential,W<:AbstractWingPotential}
    return SmoothMaxPotential{C,W}(core, wing, Float64(smooth_max_strength))
end

function potential_value(p::SmoothMaxPotential, x::Real, y::Real)
    return _smooth_max(
        core_value(p.core, x, y),
        wing_value(p.wing, x, y),
        p.smooth_max_strength,
    )
end

potential_gradient(p::SmoothMaxPotential, x::Real, y::Real) =
    _gradient_tuple(ForwardDiff.gradient(xy -> potential_value(p, xy[1], xy[2]), [x, y]))

potential_domain(p::SmoothMaxPotential) = potential_domain(p.core)

struct GeneratedCorePotential{P} <: AbstractCorePotential
    pot::P
end

core_value(p::GeneratedCorePotential, x::Real, y::Real) =
    PotentialGenerator.V(p.pot, x, y)

core_gradient(p::GeneratedCorePotential, x::Real, y::Real) =
    _gradient_tuple(PotentialGenerator.∇V(p.pot, x, y))

function potential_domain(p::GeneratedCorePotential)
    data = p.pot.data
    return (Lx=data.Lx, Ly=data.Ly)
end

const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_LSE_CORE_LX = 0.5 * pi
const DEFAULT_LSE_CORE_LY = 1.0
const DEFAULT_LSE_WING_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_LSE_WING_LX = DEFAULT_LSE_CORE_LX
const DEFAULT_LSE_WING_LY = DEFAULT_LSE_CORE_LY
const DEFAULT_LSE_WING_SCALE = 5e6

struct LSECorePotential{M} <: AbstractCorePotential
    model::M
    Lx::Float64
    Ly::Float64
end

function LSECorePotential(model; Lx::Real=DEFAULT_LSE_CORE_LX, Ly::Real=DEFAULT_LSE_CORE_LY)
    return LSECorePotential{typeof(model)}(model, Float64(Lx), Float64(Ly))
end

core_value(p::LSECorePotential, x::Real, y::Real) = _lse_predict(p.model, x, y)

core_gradient(p::LSECorePotential, x::Real, y::Real) = _lse_gradient(p.model, x, y)

potential_domain(p::LSECorePotential) = (Lx=p.Lx, Ly=p.Ly)

function load_lse_core_potential(;
    checkpoint_path::AbstractString=DEFAULT_LSE_CORE_CHECKPOINT_PATH,
    Lx::Real=DEFAULT_LSE_CORE_LX,
    Ly::Real=DEFAULT_LSE_CORE_LY,
    model_loader=_load_lse_model,
)
    return LSECorePotential(model_loader(checkpoint_path); Lx=Lx, Ly=Ly)
end

struct LSEWingPotential{M} <: AbstractWingPotential
    model::M
    Lx::Float64
    Ly::Float64
    scale::Float64
end

function LSEWingPotential(
    model;
    Lx::Real=DEFAULT_LSE_WING_LX,
    Ly::Real=DEFAULT_LSE_WING_LY,
    scale::Real=DEFAULT_LSE_WING_SCALE,
)
    return LSEWingPotential{typeof(model)}(model, Float64(Lx), Float64(Ly), Float64(scale))
end

wing_value(p::LSEWingPotential, x::Real, y::Real) =
    p.scale * _lse_predict(p.model, x, y)

function wing_gradient(p::LSEWingPotential, x::Real, y::Real)
    gx, gy = _lse_gradient(p.model, x, y)
    return (p.scale * gx, p.scale * gy)
end

potential_domain(p::LSEWingPotential) = (Lx=p.Lx, Ly=p.Ly)

function load_lse_wing_potential(;
    checkpoint_path::AbstractString=DEFAULT_LSE_WING_CHECKPOINT_PATH,
    Lx::Real=DEFAULT_LSE_WING_LX,
    Ly::Real=DEFAULT_LSE_WING_LY,
    scale::Real=DEFAULT_LSE_WING_SCALE,
    model_loader=_load_lse_model,
)
    return LSEWingPotential(model_loader(checkpoint_path); Lx=Lx, Ly=Ly, scale=scale)
end

struct JoinedLSEPotential{M} <: AbstractPotential
    model::M
    Lx::Float64
    Ly::Float64
end

function JoinedLSEPotential(model; Lx::Real=DEFAULT_LSE_CORE_LX, Ly::Real=DEFAULT_LSE_CORE_LY)
    return JoinedLSEPotential{typeof(model)}(model, Float64(Lx), Float64(Ly))
end

potential_value(p::JoinedLSEPotential, x::Real, y::Real) = _lse_predict(p.model, x, y)

potential_gradient(p::JoinedLSEPotential, x::Real, y::Real) = _lse_gradient(p.model, x, y)

potential_domain(p::JoinedLSEPotential) = (Lx=p.Lx, Ly=p.Ly)

function join_lse_potentials(core::LSECorePotential, wing::LSEWingPotential)
    core_temperature = core.model.temperature
    wing_temperature = wing.model.temperature
    isapprox(core_temperature, wing_temperature) ||
        throw(ArgumentError("Cannot join LSE potentials with different temperatures: core=$core_temperature, wing=$wing_temperature."))

    model = _LSEModel(
        hcat(core.model.A, wing.scale .* wing.model.A),
        vcat(core.model.b, wing.scale .* wing.model.b),
        core_temperature,
    )
    domain = potential_domain(core)
    return JoinedLSEPotential(model; Lx=domain.Lx, Ly=domain.Ly)
end

struct HandmadeWingPotential <: AbstractWingPotential
    Lx::Float64
    anchor::Float64
    scale::Float64
end

function HandmadeWingPotential(Lx::Real; anchor::Real=0.0, scale::Real=5e6)
    return HandmadeWingPotential(Float64(Lx), Float64(anchor), Float64(scale))
end

function _handmade_wing_shape(x::Real, y::Real)
    return 0.5 * max(0.0, x + 2 * max(0, y - 0.42) - 1.2)^2 +
           0.1 * max(0.0, x - 0.5)^2
end

function wing_value(p::HandmadeWingPotential, x::Real, y::Real)
    Δx = abs(x) - p.Lx
    return p.anchor + p.scale * (Δx + _handmade_wing_shape(Δx / 5.0, y))
end

wing_gradient(p::HandmadeWingPotential, x::Real, y::Real) =
    _gradient_tuple(ForwardDiff.gradient(xy -> wing_value(p, xy[1], xy[2]), [x, y]))

struct NonConvexWingPotential <: AbstractWingPotential
    Lx::Float64
    anchor::Float64
    scale::Float64
end

function NonConvexWingPotential(Lx::Real; anchor::Real=0.0, scale::Real=1e7)
    return NonConvexWingPotential(Float64(Lx), Float64(anchor), Float64(scale))
end

function _smooth_step(x)
    if x <= 0
        return 0.0
    elseif x >= 1
        return 1.0
    else
        f(t) = exp(-1.0 / t)
        return f(x) / (f(x) + f(1.0 - x))
    end
end

function _g(x::Real, y::Real)
    #y_min = 1 / π * acos((3.0 - 5.0 * sqrt(3.0)) / 12.0)
    #y_min = 0.6
    y_min = 0.54
    return 2.0 * _smooth_step(2.5 * x - 1.0) * max(0.0, abs(y) - y_min)^2
end

function wing_value(p::NonConvexWingPotential, x::Real, y::Real)
    Δx = abs(x) - p.Lx
    return p.anchor + p.scale * (Δx + _g(Δx / 5.0, y))
end

wing_gradient(p::NonConvexWingPotential, x::Real, y::Real) =
    _gradient_tuple((p.scale / 1e7) .* PotentialGenerator.∇V_wing(p.Lx, x, y))

end # module
