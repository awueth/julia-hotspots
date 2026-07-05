# Saved log-sum-exp potential representation.
#
# Defines the `LSEPotential` product type: a fitted log-sum-exp model, its domain
# metadata, an optional verified normalization enclosure for `Z = ∫ exp(-V)`,
# and an optional verified wing-mass enclosure. This is the representation
# loaded by downstream solver code.
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
export interval_model
export default_domain
export potential_value, potential_gradient, potential_functions
export neg_potential, log_density, density, density_cell_bounds
export potential_derivatives, potential_d1, potential_d2, potential_d11, potential_d22
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

struct LSEPotential{M<:LSEModel,D,N,W}
    model::M
    domain::D
    normalization::N
    wing_mass::W
end

# The normalization is stored as plain `Float64` bounds `(lo, hi)` enclosing
# `Z = ∫ exp(-V)`, not as an interval object. The wing mass is stored the same
# way as bounds for `Z_wing / Z`.
"""
    LSEPotential(model; domain=default_domain(), normalization=nothing, wing_mass=nothing)

Wrap an [`LSEModel`](@ref) as the saved product potential. `normalization` is
the verified mass `Z = ∫ exp(-V)` over the whole domain (stored as `(lo, hi)`
bounds), or `nothing` if it has not been computed yet. `wing_mass` stores the
verified fraction `Z_wing / Z` as `(lo, hi)` bounds, or `nothing`.
"""
LSEPotential(model::LSEModel; domain=default_domain(), normalization=nothing, wing_mass=nothing) =
    LSEPotential(model, domain, normalization, wing_mass)

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

# Index of the plane that dominates at the scalar point (cx, cy). The ranking key
# is reduced to a `Float64` first: for an interval model the plane values are (thin)
# enclosures, and comparing them directly would throw `InconclusiveBooleanOperation`
# whenever two planes tie. Any consistent scalar summary works — the reference only
# controls conditioning, not correctness — so we rank by the enclosure's `sup`.
_reference_key(v::Interval) = sup(v)
_reference_key(v) = v
_reference_index(model::LSEModel, cx, cy) =
    argmax(i -> _reference_key(_plane_value(model, i, cx, cy)), eachindex(model.b))

# Evaluate `-V` on a model directly, delegating to the shared `neg_lse`. At a
# scalar point the reference is the dominant plane there; over a box it is the
# plane dominant at the midpoint (a fixed plane, not an interval-valued maximum).
neg_potential(model::LSEModel, x::Real, y::Real) =
    neg_lse(model, x, y, _reference_index(model, x, y))
neg_potential(model::LSEModel, x::Interval, y::Interval) =
    neg_lse(model, x, y, _reference_index(model, mid(x), mid(y)))

"""
    neg_potential(p, x, y)

Return `-V(x, y)` (the log-density `log exp(-V)`) computed with the stable
log-sum-exp formula. For validated interval evaluation, wrap the model once with
[`interval_model`](@ref) and call `neg_potential(interval_model(p.model), x, y)`
directly — this scalar method exists only for the Float64 path.
"""
neg_potential(p::LSEPotential, x::Real, y::Real) = neg_potential(p.model, x, y)

# ---------------------------------------------------------------------------
# First and second partial derivatives of V = LSE_T(planes).
#
# With softmax weights wᵢ ∝ exp(lᵢ/T) over the affine planes lᵢ, the gradient is
# a weighted average of plane slopes and the diagonal Hessian entries are 1/T
# times the weighted variances of the slopes:
#
#     ∂₁V = Σ wᵢ A₁ᵢ,   ∂₁₁V = Var_w(A₁)/T ≥ 0   (and likewise for the 2-axis).
#
# These are evaluated over an interval box with the same reference-plane shift as
# `neg_potential` (so every exponential lies in (0, 1]). The variances use
# centered moments — slopes are measured relative to the reference plane's slope
# — which keeps the interval enclosure away from catastrophic cancellation; they
# are nonnegative in exact arithmetic (V is convex by construction).
# ---------------------------------------------------------------------------

"""
    potential_derivatives(p, x::Interval, y::Interval)

Validated enclosures of the partial derivatives of `V` over the interval box
`(x, y)`, returned as a named tuple `(d1, d2, d11, d22)` for
`∂₁V, ∂₂V, ∂₁₁V, ∂₂₂V`. Computed from the log-sum-exp planes in a single pass
with the stable reference-plane shift.
"""
function potential_derivatives(p::LSEPotential, x::Interval, y::Interval)
    model = interval_model(p.model)
    T = model.temperature

    # Stable shift: subtract a scalar upper bound `shift` of max_i lᵢ over the box,
    # so every exponent has nonpositive supremum and exp(...) stays in (0, 1] (no
    # overflow). Unlike `neg_potential`'s midpoint-dominant reference — valid only
    # at points — this holds over a wide cell box.
    shift = -Inf
    for i in eachindex(model.b)
        shift = max(shift, sup(_plane_value(model, i, x, y)))
    end
    shift_iv = interval(shift)

    # Center the variance moments on the mean slopes E_w[A] at the box midpoint
    # (the scalar gradient). This keeps E_w[A − a0] ≈ 0 across the box, so the
    # subtracted (E_w[A] − a0)² term is tiny and Σ wᵢ(Aᵢ − a0)² ≥ 0 dominates —
    # giving a tight, nonnegative ∂₁₁V enclosure (V is convex). Centering on a
    # far-off plane slope instead would let interval cancellation push inf(∂₁₁V)
    # spuriously negative.
    g1, g2 = gradient(p.model, mid(x), mid(y))
    a0 = interval(g1)
    b0 = interval(g2)

    total = zero(T)               # Σ wᵢ,  wᵢ = exp((lᵢ − shift)/T) ∈ (0, 1]
    sum1 = zero(T)                # Σ wᵢ (A₁ᵢ − a0)
    sum2 = zero(T)                # Σ wᵢ (A₂ᵢ − b0)
    sq1 = zero(T)                 # Σ wᵢ (A₁ᵢ − a0)²
    sq2 = zero(T)                 # Σ wᵢ (A₂ᵢ − b0)²

    for i in eachindex(model.b)
        weight = exp((_plane_value(model, i, x, y) - shift_iv) / T)
        da = model.A[1, i] - a0
        db = model.A[2, i] - b0
        total += weight
        sum1 += weight * da
        sum2 += weight * db
        sq1 += weight * da * da
        sq2 += weight * db * db
    end

    inv_total = inv(total)
    mean1 = sum1 * inv_total
    mean2 = sum2 * inv_total

    d1 = a0 + mean1
    d2 = b0 + mean2
    d11 = (sq1 * inv_total - mean1 * mean1) / T
    d22 = (sq2 * inv_total - mean2 * mean2) / T

    return (d1=d1, d2=d2, d11=d11, d22=d22)
end

"""
    potential_d1(p, x, y)

`∂₁V(x, y)` enclosure (see [`potential_derivatives`](@ref)).
"""
potential_d1(p::LSEPotential, x::Interval, y::Interval) = potential_derivatives(p, x, y).d1

"""
    potential_d2(p, x, y)

`∂₂V(x, y)` enclosure.
"""
potential_d2(p::LSEPotential, x::Interval, y::Interval) = potential_derivatives(p, x, y).d2

"""
    potential_d11(p, x, y)

`∂₁₁V(x, y)` enclosure (nonnegative in exact arithmetic).
"""
potential_d11(p::LSEPotential, x::Interval, y::Interval) = potential_derivatives(p, x, y).d11

"""
    potential_d22(p, x, y)

`∂₂₂V(x, y)` enclosure.
"""
potential_d22(p::LSEPotential, x::Interval, y::Interval) = potential_derivatives(p, x, y).d22

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
the verified normalization and by interval-arithmetic Rayleigh quotients. Scalar
and interval/Taylor inputs share the formula, but interval evaluation runs on an
interval-coefficient model, so wrap once with [`interval_model`](@ref) and call
`density(interval_model(p.model), x, y)`; the `LSEPotential` method is Float64-only.
"""
density(model::LSEModel, x, y) = exp(neg_potential(model, x, y))

density(p::LSEPotential, x::Real, y::Real) = exp(neg_potential(p, x, y))

# Integrand for zeroth-order interval-box quadrature. On first-quadrant cells,
# convexity plus evenness imply that V is nondecreasing in both coordinates.
# Thus exp(-V) is enclosed by evaluating only the lower-left and upper-right
# corners. The model-based method operates on an already interval-wrapped model
# (built once by the caller); the `LSEPotential` method wraps on demand.
function density_cell_bounds(model::LSEModel, x::Interval, y::Interval)
    inf(x) >= 0 && inf(y) >= 0 ||
        throw(ArgumentError("corner density enclosure requires first-quadrant cells"))

    xlo = interval(Float64, inf(x), inf(x))
    xhi = interval(Float64, sup(x), sup(x))
    ylo = interval(Float64, inf(y), inf(y))
    yhi = interval(Float64, sup(y), sup(y))

    density_upper = density(model, xlo, ylo)
    density_lower = density(model, xhi, yhi)
    return hull(density_lower, density_upper)
end

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
    im = interval_model(p.model)
    dom = p.domain
    factor = interval(dom.quadrant_factor)
    core_domain = [interval(0.0, dom.Lx), interval(0.0, dom.Ly)]
    wing_domain = [interval(dom.Lx, dom.x_max), interval(0.0, dom.Ly)]

    integrand(x, y) = density_cell_bounds(im, x, y)

    quadrant_core = integrate_box_cells(integrand, core_domain; cells=cells_core)
    quadrant_wing = integrate_box_adaptive(integrand, wing_domain; init=(4, 8), atol=1e-14)

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
# The full product (planes, temperature, domain, normalization, wing mass) is
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
            wing_mass=p.wing_mass,
        ))
    end
    return path
end

function load_lse_potential(path::AbstractString)
    payload = open(deserialize, path)
    model = LSEModel(payload.A, payload.b, payload.temperature)
    wing_mass = hasproperty(payload, :wing_mass) ? payload.wing_mass : nothing
    return LSEPotential(model; domain=payload.domain, normalization=payload.normalization, wing_mass=wing_mass)
end

end # module
