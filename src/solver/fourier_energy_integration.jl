if !isdefined(@__MODULE__, :MPSFunction)
    include("../functions/mps_function.jl")
end

using .MPSFunction: FittedEigenfunction, prepare_u
using IntervalArithmetic
using LinearAlgebra

"""
    FourierEnergyExpansion

Guaranteed cosine-series coefficients for `u^2` and `|grad(u)|^2`, where `u` is
an infinite-dimensional fitted eigenfunction. Frequencies are indexed from zero:
entry `(rx + 1, ry + 1)` multiplies

    cos(2rx * pi * x / diam_x) * cos(2ry * pi * y / diam_y).
"""
struct FourierEnergyExpansion{T<:Interval}
    numerator_coefficients::Matrix{T}
    denominator_coefficients::Matrix{T}
    diam_x::T
    diam_y::T
end

"""
    fourier_linf_bounds(fit::FittedEigenfunction{<:Interval})

Compute guaranteed global upper bounds for `|u|`, `|ux|`, `|uy|`, and
`|grad(u)|` from the absolute sums of the normalized Fourier coefficients.
"""
function fourier_linf_bounds(fit::FittedEigenfunction{T}) where {T<:Interval}
    data = prepare_u(fit)
    mx, my = fit.n_modes
    u_bound = interval(0.0)
    ux_bound = interval(0.0)
    uy_bound = interval(0.0)

    for q in 1:my, p in 1:mx
        coefficient_bound = abs(data.weights[p, q])
        u_bound += coefficient_bound
        ux_bound += coefficient_bound * abs(data.kx[p])
        uy_bound += coefficient_bound * abs(data.ky[q])
    end

    return (
        u=u_bound,
        ux=ux_bound,
        uy=uy_bound,
        gradient=sqrt(ux_bound * ux_bound + uy_bound * uy_bound),
    )
end

"""
    prepare_fourier_energy(fit::FittedEigenfunction{<:Interval})

Construct guaranteed cosine-series expansions of `u^2` and `|grad(u)|^2`.
The fitted eigenfunction must already have been loaded with
`intervalized=true`.
"""
function prepare_fourier_energy(fit::FittedEigenfunction{T}) where {T<:Interval}
    data = prepare_u(fit)
    mx, my = fit.n_modes
    quarter = interval(0.25)

    coefficient_shape = (2mx, 2my - 1)
    numerator = fill(zero(T), coefficient_shape)
    denominator = fill(zero(T), coefficient_shape)

    # Ordered pairs reproduce the double sum obtained by squaring the series.
    for q in 1:my, p in 1:mx, s in 1:my, r in 1:mx
        dx = abs(p - r)
        sx = p + r - 1
        dy = abs(q - s)
        sy = q + s - 2

        wpq = data.weights[p, q]
        wrs = data.weights[r, s]

        # u²: sin(a)sin(b) in x and cos(a)cos(b) in y.
        denominator_product = quarter * wpq * wrs
        denominator[dx + 1, dy + 1] += denominator_product
        denominator[dx + 1, sy + 1] += denominator_product
        denominator[sx + 1, dy + 1] -= denominator_product
        denominator[sx + 1, sy + 1] -= denominator_product

        # ux²: differentiating the x-sines gives cos(a)cos(b) in both axes.
        ux_product = quarter * (wpq * data.kx[p]) * (wrs * data.kx[r])
        numerator[dx + 1, dy + 1] += ux_product
        numerator[dx + 1, sy + 1] += ux_product
        numerator[sx + 1, dy + 1] += ux_product
        numerator[sx + 1, sy + 1] += ux_product

        # uy²: differentiating the y-cosines gives sin(a)sin(b) in both axes.
        uy_product = quarter * (wpq * data.ky[q]) * (wrs * data.ky[s])
        numerator[dx + 1, dy + 1] += uy_product
        numerator[dx + 1, sy + 1] -= uy_product
        numerator[sx + 1, dy + 1] -= uy_product
        numerator[sx + 1, sy + 1] += uy_product
    end

    return FourierEnergyExpansion(numerator, denominator, fit.diam_x, fit.diam_y)
end

function _cosine_cell_integrals(diameter::Interval, edges, frequency_count::Integer)
    result = Matrix{typeof(diameter)}(undef, length(edges) - 1, frequency_count)
    pi_interval = interval(pi)

    for i in axes(result, 1)
        lo = interval(edges[i])
        hi = interval(edges[i + 1])
        result[i, 1] = hi - lo
        for harmonic in 1:(frequency_count - 1)
            omega = interval(2harmonic) * pi_interval / diameter
            result[i, harmonic + 1] = (sin(omega * hi) - sin(omega * lo)) / omega
        end
    end
    return result
end

function _fourier_energy_values(expansion::FourierEnergyExpansion, x::Interval, y::Interval)
    numerator = zero(eltype(expansion.numerator_coefficients))
    denominator = zero(eltype(expansion.denominator_coefficients))
    pi_interval = interval(pi)

    for ry in axes(expansion.denominator_coefficients, 2),
        rx in axes(expansion.denominator_coefficients, 1)
        harmonic_x = rx - 1
        harmonic_y = ry - 1
        basis = cos(interval(2harmonic_x) * pi_interval * x / expansion.diam_x) *
                cos(interval(2harmonic_y) * pi_interval * y / expansion.diam_y)
        numerator += expansion.numerator_coefficients[rx, ry] * basis
        denominator += expansion.denominator_coefficients[rx, ry] * basis
    end
    return (numerator=numerator, denominator=denominator)
end

"""
    fourier_cell_energies(expansion, domain; cells=(10, 10))

Return guaranteed matrices containing the exact integrals of `|grad(u)|^2` and
`u^2` on every cell of a Cartesian partition. The returned grid edges are used
by [`integrate_weighted_fourier_energy`](@ref) to construct identical density
boxes.
"""
function fourier_cell_energies(
    expansion::FourierEnergyExpansion,
    domain;
    cells=(10, 10),
)
    x_axis, y_axis = domain
    nx, ny = cells
    x_edges = collect(range(Float64(inf(x_axis)), Float64(sup(x_axis)); length=nx + 1))
    y_edges = collect(range(Float64(inf(y_axis)), Float64(sup(y_axis)); length=ny + 1))

    x_integrals = _cosine_cell_integrals(
        expansion.diam_x,
        x_edges,
        size(expansion.denominator_coefficients, 1),
    )
    y_integrals = _cosine_cell_integrals(
        expansion.diam_y,
        y_edges,
        size(expansion.denominator_coefficients, 2),
    )

    denominator = x_integrals * expansion.denominator_coefficients * transpose(y_integrals)
    numerator = x_integrals * expansion.numerator_coefficients * transpose(y_integrals)

    return (
        numerator=numerator,
        denominator=denominator,
        x_edges=x_edges,
        y_edges=y_edges,
    )
end

"""
    integrate_weighted_fourier_energy(expansion, weight_bounds, domain; cells=(10, 10))

Integrate both Fourier energies against a nonnegative per-cell weight enclosure.
`weight_bounds(xbox, ybox)` is called exactly once per cell.
"""
function integrate_weighted_fourier_energy(
    expansion::FourierEnergyExpansion,
    weight_bounds,
    domain;
    cells=(10, 10),
)
    energies = fourier_cell_energies(expansion, domain; cells)
    numerator = interval(0.0)
    denominator = interval(0.0)

    for iy in axes(energies.numerator, 2), ix in axes(energies.numerator, 1)
        xbox = interval(Float64, energies.x_edges[ix], energies.x_edges[ix + 1])
        ybox = interval(Float64, energies.y_edges[iy], energies.y_edges[iy + 1])
        weight = weight_bounds(xbox, ybox)
        weight isa Interval || throw(ArgumentError("weight_bounds must return an interval"))
        isguaranteed(weight) || throw(ArgumentError("weight enclosure must be guaranteed"))
        inf(weight) >= 0 || throw(ArgumentError("weight enclosure must be nonnegative"))

        numerator += energies.numerator[ix, iy] * weight
        denominator += energies.denominator[ix, iy] * weight
    end

    return (numerator=numerator, denominator=denominator)
end
