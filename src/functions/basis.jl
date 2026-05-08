include("lambda.jl")

function ϕ(d::T, radial_eigenvalue::T, r::Union{T, AbstractArray{T}}) where T<:AbstractFloat
    if d == Inf
        a = -0.25 * radial_eigenvalue
        val_normalized = exp.(a .* (r.^2 .- 1.0) .* 0.5)
        grad_normalized = a .* r .* val_normalized
        return val_normalized, grad_normalized
    end

    order = 0.5 * (d - 1)
    k = 0.5 * sqrt(d * abs(radial_eigenvalue))

    val, grad = radial_eigenvalue < 0 ? Λ_i(order, k * r) : Λ_j(order, k * r)
    normalization, _ = radial_eigenvalue < 0 ? Λ_i(order, k * one(T)) : Λ_j(order, k * one(T))

    return val ./ normalization, k * grad ./ normalization
end

function axial_basis(λx, λy, diam_x, diam_y, x::Union{T, AbstractArray{T}}, y::Union{T, AbstractArray{T}}) where T<:AbstractFloat
    kx = sqrt(λx)
    ky = sqrt(λy)

    norm_x_sq = 0.5 * diam_x
    norm_y_sq = (λy == 0.0) ? diam_y : (diam_y / 2.0)
    inv_norm = 1.0 / sqrt(norm_x_sq * norm_y_sq)

    val = sin.(kx * x) .* cos.(ky * y) .* inv_norm
    grad = (kx .* cos.(kx * x) .* cos.(ky * y) .* inv_norm, -ky .* sin.(kx * x) .* sin.(ky * y) .* inv_norm)

    return val, grad
end