using Plots
using ForwardDiff
using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Revise

includet("../potential_generator.jl")

using .PotentialGenerator

pot = generate_potential()
data = pot.data
const Lx = data.Lx
const Ly = data.Ly

function smooth_step(x)
    # if x <= 0
    #     return 0.0
    # elseif x >= 1
    #     return 1.0
    # else
    #     f(t) = exp(-1.0 / t)
    #     return f(x) / (f(x) + f(1.0 - x))
    # end
    return max(0.0, x)^2
end

function g(x::Real, y::Real, s::Real)
    #return 2.0 * s * smooth_step(0.5 * x - 1.0) * max(0.0, abs(y) - 2.0/3.0)^2
    return s * max(0.0, x+2*y^2-5.0)^2
end

# function g(x::Real, y::Real, s::Real)
#     t0 = 3.0
#     c = 0.6
#     T = 2.0 * pi - 0.5 * pi
#     α = (1.0 - c) / (T - t0)

#     return s * max(0.0, abs(y) - 1.0 + α * max(0.0, x - t0))^2
# end

function V_wing(x, y)
    # s = 1e6

    # return s * abs(x-pi/2) + g((abs(x) - 0.5 * pi), y, s)
    return PotentialGenerator.V_wing(Lx, x, y)
end

function ∇V_wing(x, y)
    return PotentialGenerator.∇V_wing(Lx, x, y)
end

xs = range(pi/2, 2*pi, length=128)
ys = range(-1.0, 1.0, length=128)

surface(xs, ys, V_wing)

function flow_field!(du, u, p, t)
    x, y = u
    grad = ∇V_wing(x, y)
    du[1] = -grad[1]/norm(grad)
    du[2] = -grad[2]/norm(grad)
end

function plot_flow_lines()
    x0 = 2*pi
    x1 = 5.5
    x2 = 4.5
    y0s = LinRange(0.0, 1.0, 10)

    plt = plot(title="Flow Lines in Wing Potential", xlabel="x", ylabel="y", xlims=(pi/2, 2*pi), ylims=(-1.1, 1.1))

    for y0 in y0s
        u0 = [x0, y0]
        prob = ODEProblem(flow_field!, u0, (0.0, 10.0))
        sol = solve(prob, Tsit5())
        plot!(plt, sol[1, :], sol[2, :], label="", color=:green)

        u0 = [x1, y0]
        prob = ODEProblem(flow_field!, u0, (0.0, 10.0))
        sol = solve(prob, Tsit5())
        plot!(plt, sol[1, :], sol[2, :], label="", color=:blue)

        u0 = [x2, y0]
        prob = ODEProblem(flow_field!, u0, (0.0, 10.0))
        sol = solve(prob, Tsit5())
        plot!(plt, sol[1, :], sol[2, :], label="", color=:orange)
    end


    # Draw horizontal lines at y = 2/3 and y = -2/3
    hline!(plt, [2.0/3.0, -2.0/3.0], color=:red, linestyle=:dash, label="y = ±2/3")

    return plt
end

function plot_g()
    xs = range(0.5*pi, 2*pi, length=32)
    ys = range(0.0, 1.0, length=32)
    wireframe(xs, ys, (x, y) -> g((abs(x) - 0.5 * pi), y, 1e6), title="g(x,y)", xlabel="x", ylabel="y")
end

plot_flow_lines()
plot_g()

points = Iterators.product(xs, ys)

for (x, y) in points
    H = ForwardDiff.hessian((p) -> g(p[1], p[2], 1e6), [x, y])
    λ = eigmin(Symmetric(H))

    if λ < 0
        print("Negative eigenvalue at (x, y) = ($x, $y): $λ\n")
    end
end
