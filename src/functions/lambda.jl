using Plots

"""
Computes the modified Jahnke-Emden function Γ(v+1) * I_v(x) / (x/2)^v
and its derivative with respect to x.
"""
function Λ_i(v::T, x::T) where T<:AbstractFloat
    bl = one(T)
    dl = one(T)

    if abs(x) < eps(T)
        return (bl, zero(T))
    end

    rb = one(T)
    rd = one(T)
    x2_4 = T(0.25) * x * x

    for i in T(1):T(50)
        rb *= x2_4 / (i * (i + v))
        rd *= x2_4 / (i * (i + v + 1))
        bl += rb
        dl += rd
        if abs(rb) < abs(bl) * eps(T)
            break
        end
    end 

    dl *= 0.5 * x / (v + 1)

    return (bl, dl)
end

function Λ_i(v::T, xs::AbstractArray{T}) where T<:AbstractFloat
    s_vals = similar(xs, T)
    sp_vals = similar(xs, T)

    @inbounds for i in eachindex(xs)
        s_vals[i], sp_vals[i] = Λ_i(v, xs[i])
    end

    return s_vals, sp_vals
end

"""
Computes the modified Jahnke-Emden function Γ(v+1) * J_v(x) / (x/2)^v
and its derivative with respect to x.
"""
function Λ_j(v::T, x::T) where T<:AbstractFloat
    x2_4 = T(0.25) * x * x
    r = one(T)
    s = one(T)
    rp = one(T)
    sp = one(T)

    for i in 1:500
        r *= -x2_4 / (i * (i + v))
        rp *= -x2_4 / (i * (i + v + 1))
        s += r
        sp += rp
        if abs(r) < abs(s) * eps(T)
            break
        end
    end 

    sp *= -0.5 * x / (v + 1)

    return (s, sp)
end

function Λ_j(v::T, xs::AbstractArray{T}) where T<:AbstractFloat
    s_vals = similar(xs, T)
    sp_vals = similar(xs, T)

    @inbounds for i in eachindex(xs)
        s_vals[i], sp_vals[i] = Λ_j(v, xs[i])
    end

    return s_vals, sp_vals
end

function main()
    v = 23.5
    xs = range(0, 10, length=100)
    vals_i, grads_i = Λ_i(v, xs)
    vals_j, grads_j = Λ_j(v, xs)

    plot(xs, vals_i, label="Λ_i")
    #plot(xs, vals_j, label="Λ_j")

    #plot!(xs, grads_i, label="grad Λ_i")
    #plot!(xs, grads_j, label="grad Λ_j")
end
