if !isdefined(@__MODULE__, :EigenfunctionLinfNorm)
    include("eigenfunction_linf_norm.jl")
end

module EigenfunctionHotSpot

using ..EigenfunctionLinfNorm

const _PARENT = parentmodule(@__MODULE__)
const FittedEigenfunction = getfield(_PARENT, :FittedEigenfunction)

export sampled_hot_spot_difference

"""
    sampled_hot_spot_difference(fit, grid_size)

Compute the sampled difference between the interior axis witness and the
physical-boundary maximum for a positive fitted eigenfunction.
`grid_size = (nx, ny, nr)` specifies endpoint-inclusive Cartesian axes.

Both the `d = Inf` limit and finite `d` are supported: the radial axis `r`
runs over `[0, 1]`, so at finite `d` the outer shell is evaluated at the
radial-basis normalization point `r = 1`, which sits above the barrel wall
`r = 1 - V/d` by `O(V/d)` — negligible at the large `d` of interest and giving
an interior-boundary gap directly comparable to the `d = Inf` value.
"""
function sampled_hot_spot_difference(
    fit::FittedEigenfunction{T},
    grid_size::NTuple{3,Int},
) where {T<:AbstractFloat}
    nx, ny, nr = grid_size
    xs = range(zero(T), fit.diam_x / T(2); length=nx)
    ys = range(zero(T), fit.diam_y / T(2); length=ny)
    rs = range(zero(T), one(T); length=nr)

    interior = sampled_linf_norm(fit, (x=xs[1:(end - 1)], y=T[0], r=T[0]))
    outer_sample = sampled_linf_norm(fit, (x=xs, y=ys, r=T[1]))
    x_side_sample = sampled_linf_norm(fit, (x=T[last(xs)], y=ys, r=rs))
    y_side_sample = sampled_linf_norm(fit, (x=xs, y=T[last(ys)], r=rs))

    boundary_faces = (
        merge((name="outer_shell",), outer_sample),
        merge((name="x_side",), x_side_sample),
        merge((name="y_side",), y_side_sample),
    )
    samples = (interior, boundary_faces...)
    all(sample -> sample.value >= zero(T), samples) ||
        throw(ArgumentError("expected a nonnegative eigenfunction on the reduced domain"))

    boundary = boundary_faces[argmax(face.value for face in boundary_faces)]
    return (
        interior=interior,
        boundary=boundary,
        boundary_faces=boundary_faces,
        effect=interior.value - boundary.value,
    )
end

end
