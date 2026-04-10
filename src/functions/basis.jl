include("lambda.jl")
using Plots

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

function axial_basis(λx, λy, x::Union{T, AbstractArray{T}}, y::Union{T, AbstractArray{T}}) where T<:AbstractFloat
    kx = sqrt(λx)
    ky = sqrt(λy)
    val = sin.(kx * x) .* cos.(ky * y)
    grad = (kx .* cos.(kx * x) .* cos.(ky * y), -ky .* sin.(kx * x) .* sin.(ky * y))

    return val, grad
end

function main()
    d = Inf
    rs = range(0, 1, length=100)

    plot()
    for l in [-20.0, -2.0, -1.0, 0.0, 1.0, 2.0, 20.0]
        vals, grads = ϕ(d, l, rs)
        plot!(rs, grads, label="ϕ (λ=$(l))")
    end
    display(plot!())

    # zs = range(-20, 20, length=100)
    # l = π / 40

    # plot()
    # for mode in [1, 3, 5, 7]
    #     vals, grads = axial_basis(mode * l, zs)
    #     plot!(zs, vals, label="axial basis (mode=$(mode))")
    # end
    # display(plot!())
end

#main()