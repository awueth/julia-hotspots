# Saved log-sum-exp potential representation.
#
# Defines the `LSEPotential` product type: a fitted log-sum-exp model, its domain
# metadata, and an optional verified normalization enclosure for
# `Z = ∫ exp(-V)`. This is the representation loaded by downstream solver code.
#
# Scalar and interval/Taylor evaluation use the same stable log-sum-exp
# expression. The implementations are kept separate so the interval arithmetic
# used for validation remains explicit.

if !isdefined(@__MODULE__, :LSERegression)
    include("lse_regression.jl")
end
if !isdefined(@__MODULE__, :ValidatedQuadrature)
    include(joinpath(@__DIR__, "..", "quadrature", "validated_quadrature.jl"))
end

module LSEPotentials

using Serialization
using TaylorModels

using ..LSERegression
using ..LSERegression: _plane_value
using ..ValidatedQuadrature

export LSEPotential
export default_domain
export potential_value, potential_gradient, potential_functions
export neg_potential, log_density, density, density_cell_bounds
export verified_normalization
export save_lse_potential, load_lse_potential

# ---------------------------------------------------------------------------
# Domain metadata.
#
# The global construction is symmetric: the potential is even in both x and y,
# so integrals over the full domain are `quadrant_factor` (= 4) times the
# integral over the first quadrant. `Lx`/`Ly` are the core half-extents and
# `x_max` is the outer edge of the wing.
# ---------------------------------------------------------------------------

"""
    default_domain(; Lx=π/2, Ly=1.0, x_max=2π, quadrant_factor=4.0)

Domain metadata for the global core+wing potential. Stored alongside the model
so consumers no longer hardcode `π/2`, `2π`, or the symmetry factor.
"""
function default_domain(;
    Lx::Real=0.5 * pi,
    Ly::Real=1.0,
    x_max::Real=2.0 * pi,
    quadrant_factor::Real=4.0,
)
    return (
        Lx=Float64(Lx),
        Ly=Float64(Ly),
        x_max=Float64(x_max),
        quadrant_factor=Float64(quadrant_factor),
    )
end

struct LSEPotential{M<:LSEModel,D,N}
    model::M
    domain::D
    normalization::N
end

# The normalization is stored as plain `Float64` bounds `(lo, hi)` enclosing
# `Z = ∫ exp(-V)`, not as an interval object.
"""
    LSEPotential(model; domain=default_domain(), normalization=nothing)

Wrap an [`LSEModel`](@ref) as the saved product potential. `normalization` is
the verified mass `Z = ∫ exp(-V)` over the whole domain (stored as `(lo, hi)`
bounds), or `nothing` if it has not been computed yet.
"""
LSEPotential(model::LSEModel; domain=default_domain(), normalization=nothing) =
    LSEPotential(model, domain, normalization)

# ---------------------------------------------------------------------------
# Stable log-sum-exp evaluation.
#
# `V = LSE_T(planes)` and the density `exp(-V) = (Σ exp(lᵢ/T))^(-T)`. We factor
# out a single reference plane `l_ref` so every exponential lies in `(0, 1]`:
#
#     -V = -l_ref - T · log( Σᵢ exp((lᵢ - l_ref)/T) ).
#
# For interval boxes, the reference plane is chosen at the box midpoint.
# ---------------------------------------------------------------------------

_plane(model::LSEModel, i::Integer, x, y) =
    model.A[1, i] * x + model.A[2, i] * y + model.b[i]

_interval_plane(model::LSEModel, i::Integer, x, y) =
    interval(model.A[1, i]) * x + interval(model.A[2, i]) * y + interval(model.b[i])

# Index of the plane that dominates at the scalar point (cx, cy).
_reference_index(model::LSEModel, cx, cy) =
    argmax(i -> _plane_value(model, i, cx, cy), eachindex(model.b))

"""
    neg_potential(p, x, y)

Return `-V(x, y)` (the log-density `log exp(-V)`) computed with the stable
log-sum-exp formula.
"""
function neg_potential(p::LSEPotential, x::Real, y::Real)
    model = p.model
    T = model.temperature
    reference_index = _reference_index(model, x, y)
    reference = _plane(model, reference_index, x, y)
    shifted_total = one(reference)

    for i in eachindex(model.b)
        i == reference_index && continue
        shifted_total += exp((_plane(model, i, x, y) - reference) / T)
    end

    return -reference - T * log(shifted_total)
end

# Interval density: pick the dominant plane at the box centre (a fixed plane,
# not an interval-valued maximum) and evaluate the stable formula over the box.
function neg_potential(p::LSEPotential, x::Interval, y::Interval)
    model = p.model
    T = interval(model.temperature)
    reference_index = _reference_index(model, mid(x), mid(y))
    reference = _interval_plane(model, reference_index, x, y)
    shifted_total = one(reference)

    for i in eachindex(model.b)
        i == reference_index && continue
        shifted_total += exp((_interval_plane(model, i, x, y) - reference) / T)
    end

    return -reference - T * log(shifted_total)
end

"""
    potential_value(p, x, y)

Evaluate the potential `V(x, y)`.
"""
potential_value(p::LSEPotential, x::Real, y::Real) = -neg_potential(p, x, y)

"""
    potential_gradient(p, x, y)

Evaluate `∇V(x, y)` as `(gx, gy)`.
"""
potential_gradient(p::LSEPotential, x::Real, y::Real) = gradient(p.model, x, y)

"""
    density(p, x, y)

Evaluate the unnormalized density `exp(-V(x, y))`. This is the function used by
the verified normalization and by interval-arithmetic Rayleigh quotients; the
same formula serves `Float64` and interval/Taylor inputs.
"""
density(p::LSEPotential, x, y) = exp(neg_potential(p, x, y))

# Integrand for zeroth-order interval-box quadrature. On first-quadrant cells,
# convexity plus evenness imply that V is nondecreasing in both coordinates.
# Thus exp(-V) is enclosed by evaluating only the lower-left and upper-right
# corners.
function density_cell_bounds(p::LSEPotential, x::Interval, y::Interval)
    inf(x) >= 0 && inf(y) >= 0 ||
        throw(ArgumentError("corner density enclosure requires first-quadrant cells"))

    xlo = interval(Float64, inf(x), inf(x))
    xhi = interval(Float64, sup(x), sup(x))
    ylo = interval(Float64, inf(y), inf(y))
    yhi = interval(Float64, sup(y), sup(y))

    density_upper = density(p, xlo, ylo)
    density_lower = density(p, xhi, yhi)
    return hull(density_lower, density_upper)
end

_cell_density(p::LSEPotential, x::Interval, y::Interval) = density_cell_bounds(p, x, y)

"""
    verified_normalization(p; cells_core=(16, 16), cells_wing=(16, 16))

Compute a guaranteed enclosure of the normalization `Z = ∫ exp(-V)` for the
saved potential `p`, using validated interval-box quadrature over the stored
domain. Returns a named tuple

    (Z, Z_core, Z_wing, quadrant_core, quadrant_wing)

where `Z = quadrant_factor · (quadrant_core + quadrant_wing)` accounts for the
even-even symmetry, and `quadrant_core` / `quadrant_wing` are the first-quadrant
masses (useful for the wing-mass fraction, where the symmetry factor cancels).
"""
function verified_normalization(
    p::LSEPotential;
    cells_core=(16, 16),
    cells_wing=(16, 16),
)
    dom = p.domain
    factor = interval(dom.quadrant_factor)
    core_domain = [interval(0.0, dom.Lx), interval(0.0, dom.Ly)]
    wing_domain = [interval(dom.Lx, dom.x_max), interval(0.0, dom.Ly)]

    integrand(x, y) = _cell_density(p, x, y)

    quadrant_core = integrate_box_cells(integrand, core_domain; cells=cells_core)
    quadrant_wing = integrate_box_cells(integrand, wing_domain; cells=cells_wing)

    return (
        Z=factor * (quadrant_core + quadrant_wing),
        Z_core=factor * quadrant_core,
        Z_wing=factor * quadrant_wing,
        quadrant_core=quadrant_core,
        quadrant_wing=quadrant_wing,
    )
end

"""
    potential_functions(p; scale=1.0)

Return solver-ready closures `(V, gradV)`.
"""
function potential_functions(p::LSEPotential; scale::Real=1.0)
    scale_value = Float64(scale)
    V = (x, y) -> scale_value * potential_value(p, x, y)
    gradV = (x, y) -> begin
        gx, gy = potential_gradient(p, x, y)
        return (scale_value * gx, scale_value * gy)
    end
    return V, gradV
end

# ---------------------------------------------------------------------------
# The full product (planes, temperature, domain, normalization) is
# serialized so the saved file captures everything the verified computation
# depends on.
# ---------------------------------------------------------------------------

function save_lse_potential(path::AbstractString, p::LSEPotential)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, (
            A=Matrix(p.model.A),
            b=Vector(p.model.b),
            temperature=p.model.temperature,
            domain=p.domain,
            normalization=p.normalization,
        ))
    end
    return path
end

function load_lse_potential(path::AbstractString)
    payload = open(deserialize, path)
    model = LSEModel(payload.A, payload.b, payload.temperature)
    return LSEPotential(model; domain=payload.domain, normalization=payload.normalization)
end

end # module
