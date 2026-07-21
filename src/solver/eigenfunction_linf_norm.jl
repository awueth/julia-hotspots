if !isdefined(@__MODULE__, :MPSFunction)
    include("../functions/mps_function.jl")
end

module EigenfunctionLinfNorm

using IntervalArithmetic
using ..MPSFunction: FittedEigenfunction, prepare_u, u

export sampled_linf_norm, interval_linf_norm

"""
    sampled_linf_norm(fit::FittedEigenfunction, axes)

Estimate the `L∞` norm on the Cartesian product of explicit nonempty sample
axes `(x=..., y=..., r=...)`. The first maximizer in deterministic `(r, y, x)`
traversal order is returned.
"""
function sampled_linf_norm(fit::FittedEigenfunction{T}, axes::NamedTuple{(:x,:y,:r)}) where {T<:AbstractFloat}
    evaluator = isinf(fit.d) ? prepare_u(fit) : fit

    best_abs = T(-Inf)
    best_value = zero(T)
    best_location = (x=T(first(axes.x)), y=T(first(axes.y)), r=T(first(axes.r)))

    for r in axes.r, y in axes.y, x in axes.x
        location = (x=T(x), y=T(y), r=T(r))
        value = u(evaluator, location.x, location.y, location.r)
        abs_value = abs(value)
        if abs_value > best_abs
            best_abs = abs_value
            best_value = value
            best_location = location
        end
    end

    return (linf=best_abs, location=best_location, value=best_value)
end

"""
    sampled_linf_norm(fit::FittedEigenfunction, r, grid_size)

Estimate the `L∞` norm of a fitted eigenfunction on the fixed radial slice `r`
by sampling an endpoint-inclusive Cartesian grid on the symmetry-reduced
rectangle `[0, diam_x / 2] × [0, diam_y / 2]`.

The result contains the sampled norm, the first maximizing grid point in
deterministic `(y, x)` traversal order, and the signed value at that point.
"""
function sampled_linf_norm(fit::FittedEigenfunction{T}, r::Real, grid_size::Tuple{Int,Int}) where {T<:AbstractFloat}
    nx, ny = grid_size
    xs = range(zero(T), fit.diam_x / T(2); length=nx)
    ys = range(zero(T), fit.diam_y / T(2); length=ny)
    return sampled_linf_norm(fit, (x=xs, y=ys, r=T[T(r)]))
end

"""
    interval_linf_norm(fit::FittedEigenfunction{<:Interval}, atol)

Compute a guaranteed enclosure of the continuous `L∞` norm on the full
symmetry-reduced `(x, y, r)` domain. The search starts with eight cells, discards cells below
the best midpoint lower bound, and bisects possible maximizers until the norm
enclosure has diameter strictly smaller than the absolute tolerance `atol`.
The fitted eigenfunction must already be intervalized and must have `d = Inf`.

The returned `maximizing_cells` are all cells tied for the largest upper bound
on `u`. This routine uses that the eigenfunction is nonnegative on the
symmetry-reduced rectangle.
"""
function interval_linf_norm(fit::FittedEigenfunction{T}, atol::Real) where {T<:Interval}
    @assert 0 < atol < Inf

    domain = [
        interval(Float64, 0.0, sup(fit.diam_x / interval(2))),
        interval(Float64, 0.0, sup(fit.diam_y / interval(2))),
        interval(0.0, 1.0),
    ]
    evaluator = prepare_u(fit)

    function bound_cell(box)
        enclosure = u(evaluator, box[1], box[2], box[3])
        midpoint_value = u(
            evaluator,
            interval(mid(box[1])),
            interval(mid(box[2])),
            interval(mid(box[3])),
        )
        return (
            box=box,
            lower=max(0.0, inf(midpoint_value)),
            upper=max(0.0, sup(enclosure)),
        )
    end

    cells = bound_cell.(mince(domain, (2, 2, 2)))
    lower_bound = maximum(cell.lower for cell in cells)
    filter!(cell -> cell.upper >= lower_bound, cells)
    relative_widths(box) = diam.(box) ./ diam.(domain)
    relative_width(cell) = maximum(relative_widths(cell.box))

    while true
        upper_bound = maximum(cell.upper for cell in cells)
        upper_bound - lower_bound < atol && break

        top_indices = findall(cell -> cell.upper == upper_bound, cells)
        split_index = top_indices[argmax(relative_width.(cells[top_indices]))]

        box = cells[split_index].box
        split_axis = argmax(relative_widths(box))
        child_boxes = bisect(box, split_axis)
        can_split = all(child -> diam(child[split_axis]) < diam(box[split_axis]), child_boxes)
        can_split ||
            throw(ErrorException("interval cells cannot be bisected further to reach atol=$atol"))
        children = bound_cell.(child_boxes)
        deleteat!(cells, split_index)
        append!(cells, children)

        lower_bound = max(lower_bound, maximum(child.lower for child in children))
        filter!(cell -> cell.upper >= lower_bound, cells)
    end

    upper_bound = maximum(cell.upper for cell in cells)
    maximizing_cells = [
        (x=cell.box[1], y=cell.box[2], r=cell.box[3])
        for cell in cells if cell.upper == upper_bound
    ]
    return (
        linf=interval(Float64, lower_bound, upper_bound),
        maximizing_cells=maximizing_cells,
    )
end

end
