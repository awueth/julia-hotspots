using Plots
using ForwardDiff
using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Revise
using Base.Threads

#includet("../potential_generator.jl")

#using .PotentialGenerator

#pot = generate_potential()
#data = pot.data
const Lx = 0.5 * pi
xs = range(Lx, Lx + 5.0, length=128)
ys = range(-1.0, 1.0, length=128)

function smooth_step(x)
    if x <= 0
        return 0.0
    elseif x >= 1
        return 1.0
    else
        f(t) = exp(-1.0 / t)
        return f(x) / (f(x) + f(1.0 - x))
    end
    # return max(0.0, x)^2
end

function g(x::Real, y::Real)
    return 0.5 * max(0.0, x + 2 * max(0, y - 0.42) - 1.2)^2 +
           0.1 * max(0.0, x - 0.5)^2
    #y_min = 1/π * acos((3.0 - 5.0 * sqrt(3.0)) / 12.0)
    #return 2.0 * smooth_step(2.5 * x - 1.0) * max(0.0, abs(y) - y_min)^2
end

function V_wing(x, y)
    Δx = abs(x) - Lx
    return 1e7 * (Δx + g(Δx/5.0, y))
end

function ∇V_wing(x, y)
    return ForwardDiff.gradient(xy -> V_wing(xy[1], xy[2]), SVector(x, y))
end

function flow_field!(du, u, p, t)
    x, y = u
    grad = ∇V_wing(x, y)
    du[1] = -grad[1]/norm(grad)
    du[2] = -grad[2]/norm(grad)
end

function plot_flow_lines()
    x0 = Lx + 5.0
    x1 = Lx + 4.0
    x2 = Lx + 3.0
    x3 = Lx + 2.0
    x4 = Lx + 1.0
    y0s = LinRange(0.0, 1.0, 10)

    sols = Vector{Any}(undef, length(y0s) * 5)

    Threads.@threads for i in eachindex(y0s)
        y0 = y0s[i]
        
        prob0 = ODEProblem(flow_field!, [x0, y0], (0.0, 2.0))
        sols[(i-1)*5 + 1] = solve(prob0, Tsit5())

        prob1 = ODEProblem(flow_field!, [x1, y0], (0.0, 2.0))
        sols[(i-1)*5 + 2] = solve(prob1, Tsit5())

        prob2 = ODEProblem(flow_field!, [x2, y0], (0.0, 2.0))
        sols[(i-1)*5 + 3] = solve(prob2, Tsit5())

        prob3 = ODEProblem(flow_field!, [x3, y0], (0.0, 2.0))
        sols[(i-1)*5 + 4] = solve(prob3, Tsit5())

        prob4 = ODEProblem(flow_field!, [x4, y0], (0.0, 2.0))
        sols[(i-1)*5 + 5] = solve(prob4, Tsit5())
    end

    plt = plot(title="Flow Lines in Wing Potential", xlabel="x", ylabel="y", xlims=(Lx, Lx + 5.0), ylims=(-1.1, 1.1))
    
    for (i, sol) in enumerate(sols)
        col = (i % 5 == 1) ? :green : 
              (i % 5 == 2) ? :blue : 
              (i % 5 == 3) ? :orange :
              (i % 5 == 4) ? :purple : :cyan
        plot!(plt, sol[1, :], sol[2, :], label="", color=col)
    end

    y_min = 0.6 #1 / π * acos((3.0 - 5.0 * sqrt(3.0)) / 12.0)

    hline!(plt, [-y_min, y_min], color=:red, linestyle=:dash, label="y = ±2/3")

    return plt
end

function plot_g()
    xs = range(0.0, 1.0, length=32)
    ys = range(0.0, 1.0, length=32)
    plt = wireframe(xs, ys, (x, y) -> g(x, y),
     xlabel="x", ylabel="y", camera=(-20, 30))

    y0 = 2.0/3.0
    zline = [g(x, y0) for x in xs]
    plot!(plt, xs, fill(y0, length(xs)), zline;
          color=:red, lw=2, seriestype=:path, label="y = 2/3")
end

plot_flow_lines()
plot_g()
surface(xs, ys, V_wing)

points = Iterators.product(range(0.0, 1.0, length=64), range(0.0, 1.0, length=64))

for (x, y) in points
    H = ForwardDiff.hessian((p) -> g(p[1], p[2]), [x, y])
    λ = eigmin(Symmetric(H))

    if λ < 0
        print("Negative eigenvalue at (x, y) = ($x, $y): $λ\n")
    end
end
