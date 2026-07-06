using TOML

project_root = normpath(joinpath(@__DIR__, "..", ".."))
result_path = joinpath(project_root, "results", "1-fem-eigenvalues.toml")

include(joinpath(@__DIR__, "..", "solver", "eigenvalue_fem.jl"))

using .EigenvalueFEM: compute_fem_eigenvalues
using .LSEPotentials: load_lse_potential
pot = load_lse_potential(joinpath(@__DIR__, "..", "..", "checkpoints", "lse_global_potential.chk"))


partition = (256, 128)
h_K = sqrt((pi/partition[1])^2 + (2.0/partition[2])^2)
values, vectors, oscillation = compute_fem_eigenvalues(pot; partition=partition, nev=4)

println("Maximum second-quadrant simplex oscillation of V: ", oscillation.value)
println("Attained on cell ", oscillation.cell, " with vertices ", oscillation.vertices)


println("Found Symmetric Neumann eigenvalues: ", values)

mkpath(dirname(result_path))
open(result_path, "w") do io
    TOML.print(io, Dict(
        "step" => "1-fem-eigenvalues",
        "inputs" => Dict(
            "potential_checkpoint" => "checkpoints/lse_global_potential.chk",
        ),
        "parameters" => Dict(
            "partition" => collect(partition),
            "nev" => 4,
        ),
        "result" => Dict(
            "eigenvalues" => collect(values),
            "max_oscillation" => oscillation.value,
            "h_K" => h_K,
        ),
    ); sorted=true)
end

println("Result certificate saved to ", result_path)
