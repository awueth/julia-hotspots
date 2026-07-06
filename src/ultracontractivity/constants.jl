# Wang–Li–Yau ultracontractivity constants (d = ∞).
#
# This module exposes verified computations of the ultracontractivity constants
# from `writeup/appendix.typ`. It replaces the non-rigorous
# `src/examples/li_yau_constants.jl`, which used `HCubatureJL`.
#
# First constant. For a fixed time `t > 0`, `C₁(t)` is the smallest constant with
#
#     1/Z ∫_Q e^(-|x|²/t) e^(-V(x)) dx ≥ 1/C₁²,     Z = ∫_Q e^(-V) dx.
#
# Writing I(t) = ∫_Q e^(-|x|²/t) e^(-V), the normalized integral is N = I/Z and
# C₁ = 1/√N = √(Z/I). We enclose both I and Z with validated interval-box
# quadrature, so `√(Z/I)` is an enclosure of C₁ and `sup` of it is a rigorous
# upper bound: C₁ ≤ √(sup(Z)/inf(I)).
#
# Second constant. With α = 1/t₁, `C₂(t₁,t₂)` bounds ‖P_{t₂} e^(α‖x‖²)‖_∞ for the
# semigroup of L = -Δ + ∇V·∇. Instead of a barrier we use the maximum principle on
# the core (writeup/ultracontractivity.typ, §"Reduction to the core"). Separating
# x₂ gives ‖P_{t₂} e^(α‖x‖²)‖_∞ ≤ e^(α L_y²) ‖P_{t₂} e^(α x₁²)‖_∞. The steep wing
# drift pushes mass into the buffered core C_δ = [0, ℓ+δ] (ℓ = Lx), where the max
# principle bounds e^(α x₁²) by its core maximum e^(α(ℓ+δ)²), plus a reduction error
# from the small probability of still being in the far wing. The result is
#
#   C₂ ≤ e^(α(L_y² + (ℓ+δ)²))                                  (main, core maximum)
#      + e^(α(L_y² + R²)) (e^(-(Λt₂ - w)²/(4t₂)) + e^(-Λδ)),   (reduction error)
#
# with R = x_max the wing end, w = R − ℓ the wing width, and Λ ≤ ∂₁V a certified
# lower bound on the wing steepness (min of the interval enclosure of ∂₁V over the
# wing cells). Both error terms decrease in Λ, so the certified lower bound yields a
# rigorous upper bound on C₂. Here we take the flow time τ = t₂: t₂ enters the bound
# only through the constraint τ < t₂, and τ = t₂ is admissible (flow the whole time,
# then the remaining semigroup is the identity). The main term is the requested
# exp((1 + (π/2)²)/t₁) up to the buffer δ ≪ ℓ.

if !isdefined(@__MODULE__, :LSEPotentials)
    include(joinpath(@__DIR__, "..", "potentials", "lse_potential.jl"))
end

module UltracontractivityConstants

using TaylorModels

using ..LSEPotentials
using ..LSEPotentials: density
using ..ValidatedQuadrature

export first_constant, second_constant, total_constant

# Per-cell enclosure of the weighted integrand g(x,y) = e^(-|x|²/t) e^(-V(x,y)).
#
# On first-quadrant cells g is nonincreasing in both coordinates: it is exp of a
# sum of two nonincreasing exponents, -(x²+y²)/t and -V (the latter by convexity
# plus evenness, the same assumption `density_cell_bounds` relies on). So g is
# enclosed by its values at the lower-left corner (the maximum) and the
# upper-right corner (the minimum). Evaluating the *combined* integrand at each
# point corner keeps the Gaussian and e^(-V) factors correlated — both extremal
# at the same corner — which is tighter than multiplying independent interval
# range enclosures of the two factors over the cell.
function weighted_density_cell_bounds(model, x::Interval, y::Interval, t::Interval)
    inf(x) >= 0 && inf(y) >= 0 ||
        throw(ArgumentError("corner weighted-density enclosure requires first-quadrant cells"))

    xlo = interval(Float64, inf(x), inf(x))
    xhi = interval(Float64, sup(x), sup(x))
    ylo = interval(Float64, inf(y), inf(y))
    yhi = interval(Float64, sup(y), sup(y))

    g_upper = density(model, xlo, ylo) * exp(-(xlo^2 + ylo^2) / t)
    g_lower = density(model, xhi, yhi) * exp(-(xhi^2 + yhi^2) / t)
    return hull(g_lower, g_upper)
end

"""
    first_constant(p; t, cells_core=(128, 128), cells_wing=(128, 128))

Verified enclosure of the first ultracontractivity constant `C₁(t)` for the
saved potential `p`, defined by `1/Z ∫_Q e^(-|x|²/t) e^(-V) ≥ 1/C₁²`.

Returns a named tuple `(C1, normalized_integral, I, Z, t)` where every field
except `t` is an interval enclosure: `I = ∫_Q e^(-|x|²/t) e^(-V)`, `Z = ∫_Q e^(-V)`,
`normalized_integral = I/Z`, and `C1 = √(Z/I)`. A rigorous upper bound on the
true constant is `sup(C1)`.

`I` is computed with validated box quadrature; `Z` is the verified normalization
saved with the potential (recomputed via [`verified_normalization`](@ref) only if
the checkpoint stored none). The symmetry factor `quadrant_factor` is applied to
both, so it cancels in `normalized_integral` and `C1`.
"""
function first_constant(
    p::LSEPotential;
    t::Real,
    cells_core=(128, 128),
    cells_wing=(8, 8),
)
    t > 0 || throw(ArgumentError("t must be positive."))

    dom = p.domain
    factor = interval(dom.quadrant_factor)
    t_iv = interval(t)
    core_box = [interval(0.0, dom.Lx), interval(0.0, dom.Ly)]
    wing_box = [interval(dom.Lx, dom.x_max), interval(0.0, dom.Ly)]

    im = interval_model(p.model)   # wrap the planes once, not per cell
    weighted(x, y) = weighted_density_cell_bounds(im, x, y, t_iv)

    I = factor * (
        integrate_box_cells(weighted, core_box; cells=cells_core) +
        integrate_box_adaptive(weighted, wing_box; init=cells_wing)
    )

    # Z = ∫_Q e^(-V) is the verified normalization saved with the potential; only
    # recompute it if the checkpoint did not store one.
    Z = if p.normalization === nothing
        verified_normalization(p; cells_core=cells_core, cells_wing=cells_wing).Z
    else
        interval(p.normalization.lo, p.normalization.hi)
    end

    normalized_integral = I / Z
    C1 = sqrt(Z / I)

    return (
        C1=C1,
        normalized_integral=normalized_integral,
        I=I,
        Z=Z,
        t=Float64(t),
    )
end

# ---------------------------------------------------------------------------
# Second constant: max principle on the core, with a reduction error controlled
# by the certified wing steepness Λ.
# ---------------------------------------------------------------------------

"""
    wing_steepness(p; ny=128)

Certified `Float64` lower bound `Λ ≤ ∂₁V` over the wing `[Lx, x_max] × [0, Ly]`.

`V` is a log-sum-exp of affine planes, hence convex, so `∂₁V` is nondecreasing in
`x₁` and its minimum over the wing is attained at the inner edge `x₁ = Lx`. We
therefore evaluate `∂₁V` at the *thin* inner edge over `ny` sub-intervals in `y`
and take the minimum of the enclosures. Evaluating on a thin `x`-slice avoids the
interval-softmax overflow that a wide wing cell would trigger (the steep wing
planes divided by the small temperature blow past `exp`'s range). Since `V` is even
in `y`, `∂₁V` is even in `y`, so the first-quadrant minimum is the full-wing minimum.
"""
function wing_steepness(p::LSEPotential, ell::Interval; ny::Integer=128)
    dom = p.domain
    y0, yLmax = 0.0, dom.Ly
    ye(j) = y0 + (yLmax - y0) * j / ny

    Lambda = Inf
    for j in 1:ny
        d1 = potential_d1(p, ell, interval(ye(j - 1), ye(j)))
        Lambda = min(Lambda, inf(d1))
    end
    return Lambda
end

"""
    second_constant(p; t1, t2, delta=1e-3, ny_wing=128)

Verified bound on the second ultracontractivity constant `C₂(t₁,t₂)`, i.e.
`‖P_{t₂} e^(‖x‖²/t₁)‖_∞ ≤ C₂`, via the maximum principle on the buffered core (no
barrier). With `α = 1/t₁`, core half-widths `ℓ = Lx`, `L_y`, wing end `R = x_max`,
wing width `w = R − ℓ`, buffer `δ`, and certified wing steepness `Λ`
([`wing_steepness`](@ref)),

    C₂ ≤ e^(α(L_y² + (ℓ+δ)²)) + e^(α(L_y² + R²)) (e^(-(Λt₂ - w)²/(4t₂)) + e^(-Λδ)).

The flow time is taken equal to `t₂`. Returns a named tuple
`(C2, main, error, Lambda, delta, t1, t2)` with interval fields (except the
`Float64` `Lambda`, `delta`, `t1`, `t2`); `sup(C2)` is the rigorous upper bound.
Both error terms are decreasing in `Λ`, so the certified lower bound on `Λ` keeps
the result an upper bound.
"""
function second_constant(
    p::LSEPotential;
    t1::Real,
    t2::Real,
    delta::Real=1e-3,
    ny_wing::Integer=256,
)
    (t1 > 0 && t2 > 0 && delta > 0) ||
        throw(ArgumentError("t1, t2 and delta must be positive."))

    dom = p.domain
    alpha = inv(interval(t1))
    Ly = interval(dom.Ly)
    ell = interval(dom.Lx + 0.0001)
    R = interval(dom.x_max)
    w = R - ell
    del = interval(delta)
    t2i = interval(t2)

    Lambda = wing_steepness(p, ell; ny=ny_wing)   # certified lower bound
    Lam = interval(Lambda)

    main = exp(alpha * (Ly^2 + (ell + del)^2))

    # Reduction error e^(α(L_y²+R²)) q_δ(t₂). The wing-escape tail bound
    # exp(-(Λt₂-w)²/(4t₂)) is only a valid probability bound when Λt₂ > w; if not,
    # fall back to the trivial P ≤ 1.
    tail_wing = Lambda * t2 > dom.x_max - dom.Lx ?
        exp(-(Lam * t2i - w)^2 / (interval(4.0) * t2i)) : interval(1.0)
    q = tail_wing + exp(-Lam * del)
    error_term = exp(alpha * (Ly^2 + R^2)) * q

    C2 = main + error_term

    return (
        C2=C2,
        main=main,
        error=error_term,
        Lambda=Lambda,
        delta=Float64(delta),
        t1=Float64(t1),
        t2=Float64(t2),
    )
end

function total_constant(
    p::LSEPotential,
    t1::Real,
    t2::Real;
    delta::Real=1e-3,
    cells_core=(128, 128),
    cells_wing=(8, 8),
    ny_wing::Integer=256,
)
    C1 = first_constant(p; t=t1, cells_core=cells_core, cells_wing=cells_wing).C1
    C2 = second_constant(p; t1=t1, t2=t2, delta=delta, ny_wing=ny_wing).C2

    return C1 * C2
end

end # module