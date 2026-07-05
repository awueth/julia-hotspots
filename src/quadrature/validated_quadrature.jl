module ValidatedQuadrature

using TaylorModels

export integrate_box_cells, integrate_box_adaptive, integrate_taylor_cells

function _validated_domain(domain)
    length(domain) == 2 ||
        throw(ArgumentError("domain must contain exactly two intervals"))

    validated = Vector{Interval{Float64}}(undef, 2)
    for i in eachindex(validated)
        domain[i] isa Interval ||
            throw(ArgumentError("domain entries must be intervals"))
        isguaranteed(domain[i]) ||
            throw(ArgumentError("domain intervals must be guaranteed"))

        lo = inf(domain[i])
        hi = sup(domain[i])
        isfinite(lo) && isfinite(hi) ||
            throw(ArgumentError("domain intervals must have finite endpoints"))
        hi > lo ||
            throw(ArgumentError("domain intervals must have positive width"))

        validated[i] = interval(Float64, lo, hi)
    end
    return validated
end

function _cell_area(cell)
    return prod(interval.(sup.(cell)) .- interval.(inf.(cell)))
end

function _integrate_taylor_model(tm::TaylorModelN{2})
    centered = centered_dom(tm)
    lo = interval.(inf.(centered))
    hi = interval.(sup.(centered))

    antiderivative = integrate(integrate(polynomial(tm), 1), 2)
    polynomial_integral =
        antiderivative(hi) -
        antiderivative([lo[1], hi[2]]) -
        antiderivative([hi[1], lo[2]]) +
        antiderivative(lo)

    return polynomial_integral + remainder(tm) * _cell_area(domain(tm))
end

function _integrate_box_value(value::Interval, cell)
    return value * _cell_area(cell)
end

function _integrate_box_value(value::Real, cell)
    return convert(Interval{Float64}, value) * _cell_area(cell)
end

function _integrate_taylor_value(value::TaylorModelN{2})
    return _integrate_taylor_model(value)
end

"""
    integrate_box_cells(f, domain; cells=(10, 10))

Compute a validated interval-box enclosure of the integral of the scalar
function `f(x, y)` over a two-dimensional rectangular interval `domain`.

The first domain axis is divided into `cells[1]` pieces and the second into
`cells[2]` pieces. On each cell, `f` is evaluated on the interval box itself,
and the resulting range enclosure is multiplied by the cell area.
"""
function integrate_box_cells(f, domain; cells=(10, 10))
    validated_domain = _validated_domain(domain)
    total = 0.0 .. 0.0

    for cell in mince(validated_domain, cells)
        total += _integrate_box_value(f(cell[1], cell[2]), cell)
    end

    return total
end

"""
    integrate_box_adaptive(f, domain; init=(4, 8), split_axis=1, atol=1e-6, maxcells=100_000)

Compute a validated interval-box enclosure of the integral of the scalar
function `f(x, y)` over a two-dimensional rectangular interval `domain`, using
adaptive refinement instead of a fixed grid.

Integration starts from a coarse `init[1] × init[2]` grid. On each pass the cells
whose contribution has the widest interval enclosure are bisected along
`split_axis` (1 for x, 2 for y). Refinement stops once the total enclosure width
`diam` drops to `atol`, or once the cell count reaches `maxcells`. Because the
budget caps subdivision, the returned interval may be wider than `atol` when
`maxcells` is hit first.

Returns the enclosing `Interval`, matching [`integrate_box_cells`](@ref).
"""
function integrate_box_adaptive(f, domain; init=(4, 4), split_axis::Integer=1, atol=1e-6, maxcells=1000)
    validated_domain = _validated_domain(domain)
    1 <= split_axis <= 2 ||
        throw(ArgumentError("split_axis must be 1 or 2"))

    cells = mince(validated_domain, init)
    contribs = [_integrate_box_value(f(cell[1], cell[2]), cell) for cell in cells]

    while length(cells) < maxcells
        total = sum(contribs)
        diam(total) <= atol && break

        wmax = maximum(diam.(contribs))
        wmax == 0 && break

        new_cells = eltype(cells)[]
        new_contribs = eltype(contribs)[]
        for (cell, contrib) in zip(cells, contribs)
            if diam(contrib) >= wmax / 2
                for child in bisect(cell, split_axis, 0.5)
                    push!(new_cells, child)
                    push!(new_contribs, _integrate_box_value(f(child[1], child[2]), child))
                end
            else
                push!(new_cells, cell)
                push!(new_contribs, contrib)
            end
        end
        cells, contribs = new_cells, new_contribs
    end

    return sum(contribs)
end

"""
    integrate_taylor_cells(f, domain; order=16, cells=(10, 10))

Compute a validated enclosure of the integral of the scalar function `f(x, y)`
over a two-dimensional rectangular interval `domain`.

The first domain axis is divided into `cells[1]` pieces and the second into
`cells[2]` pieces. A separate Taylor model of order `order` is constructed at
the midpoint of every grid cell. The polynomial is integrated exactly and its
absolute remainder is multiplied by the cell area before all cell enclosures
are summed.

This function may update TaylorSeries' process-global default jet space when
it does not already support two variables at the required order.
"""
function integrate_taylor_cells(f, domain; order=16, cells=(10, 10))
    validated_domain = _validated_domain(domain)

    required_order = max(2order, order + 2)
    if TaylorModels.TS.get_numvars() != 2 || TaylorModels.TS.order() < required_order
        jet_order = max(TaylorModels.TS.order(), required_order)
        variables!(
            Interval{Float64},
            "x y";
            order=jet_order,
            nowarn=true,
        )
    end

    total = 0.0 .. 0.0

    for cell in mince(validated_domain, cells)
        center = interval.(mid.(cell))
        x = TaylorModelN(1, order, center, cell)
        y = TaylorModelN(2, order, center, cell)
        total += _integrate_taylor_value(f(x, y))
    end

    return total
end

end
