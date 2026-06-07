include("lambda.jl")

function ϕ(d::T, radial_eigenvalue::T, r::Union{T, AbstractArray{T}}) where T<:AbstractFloat
    if d == Inf
        a = T(-0.25) * radial_eigenvalue
        val_normalized = exp.(a .* (r.^2 .- one(T)) .* T(0.5))
        grad_normalized = a .* r .* val_normalized
        return val_normalized, grad_normalized
    end

    order = T(0.5) * (d - one(T))
    k = T(0.5) * sqrt(d * abs(radial_eigenvalue))

    val, grad = radial_eigenvalue < zero(T) ? Λ_i(order, k * r) : Λ_j(order, k * r)
    normalization, _ = radial_eigenvalue < zero(T) ? Λ_i(order, k * one(T)) : Λ_j(order, k * one(T))

    return val ./ normalization, k * grad ./ normalization
end

function axial_basis(λx, λy, diam_x, diam_y, x::Union{T, AbstractArray{T}}, y::Union{T, AbstractArray{T}}) where T<:AbstractFloat
    kx = sqrt(λx)
    ky = sqrt(λy)

    norm_x_sq = T(0.5) * diam_x
    norm_y_sq = (λy == zero(eltype(λy))) ? diam_y : (diam_y / T(2.0))
    inv_norm = inv(sqrt(norm_x_sq * norm_y_sq))

    val = sin.(kx * x) .* cos.(ky * y) .* inv_norm
    grad = (kx .* cos.(kx * x) .* cos.(ky * y) .* inv_norm, -ky .* sin.(kx * x) .* sin.(ky * y) .* inv_norm)

    return val, grad
end

function axial_basis_tables(
    xs::AbstractVector{T},
    ys::AbstractVector{T},
    n_modes::Tuple{Int, Int},
    diam_x::T,
    diam_y::T,
) where {T<:AbstractFloat}
    mx, my = n_modes

    Kx = [T(2p - 1) * (T(π) / diam_x) for p in 1:mx]
    Ky = [T(2q - 2) * (T(π) / diam_y) for q in 1:my]

    λx_modes = Kx .^ 2
    λy_modes = Ky .^ 2

    norm_x_sq = T(0.5) * diam_x
    norm_y_sq = [
        iszero(Ky[q]) ? diam_y : diam_y / T(2.0)
        for q in 1:my
    ]
    inv_norms = inv.(sqrt.(norm_x_sq .* norm_y_sq))

    Sx = sin.(xs .* Kx')
    Cx = cos.(xs .* Kx')
    Sy = sin.(ys .* Ky')
    Cy = cos.(ys .* Ky')

    return (; Kx, Ky, λx_modes, λy_modes, inv_norms, Sx, Cx, Sy, Cy)
end

@inline function axial_basis(
    tables,
    ix,
    iy,
    p,
    q
)
    sx = tables.Sx[ix, p]
    cx = tables.Cx[ix, p]
    sy = tables.Sy[iy, q]
    cy = tables.Cy[iy, q]

    kx = tables.Kx[p]
    ky = tables.Ky[q]
    inv_norm = tables.inv_norms[q]

    av = sx * cy * inv_norm
    agx = kx * cx * cy * inv_norm
    agy = -ky * sx * sy * inv_norm

    return av, (agx, agy)
end