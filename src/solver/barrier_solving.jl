if !isdefined(@__MODULE__, :Geometry)
    include("solver.jl")
end

using JuMP
using OSQP

function solve_barrier(geometry::Geometry, n_modes::Tuple{Int,Int}, λ::Float64, neumann_vals)
    A = get_matrix(geometry, n_modes, λ)

    model = Model(OSQP.Optimizer)
    @variable(model, c[1:size(A, 2)])
    @objective(model, Min, sum((A * c) .^ 2))
    @constraint(model, A * c .>= neumann_vals)
    optimize!(model)

    return value.(c)
end
