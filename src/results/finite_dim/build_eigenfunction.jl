# Phase 2: fit the finite-dimensional MPS candidate eigenfunction and serialize it.
#
# Computed ONCE and reused by phase 3 (bounds.jl) for every tuning run. Reuses
# the shared log_concave potential artifact (finite_dim never builds its own).
# Output (fixed path, loaded by bounds.jl):
#   checkpoints/first_eigenfunction_finite_dim.chk
#
# Run once from the repository root:
#   julia --project=. src/results/finite_dim/build_eigenfunction.jl [config.toml]
# Reads the [fit] table; defaults to finite-dim.toml.

using TOML

include(joinpath(@__DIR__, "..", "..", "potentials", "lse_potential.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "solver.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "eigenfunction_io.jl"))

using .LSEPotentials: load_lse_potential, potential_functions

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const POTENTIAL_CHK = joinpath(
    PROJECT_ROOT, "checkpoints", "log_concave_extension", "high-resolution",
    "lse_global_potential.chk",
)
const FIT_CHK = joinpath(PROJECT_ROOT, "checkpoints", "first_eigenfunction_finite_dim.chk")

function build_eigenfunction(config_path::AbstractString)
    c = TOML.parsefile(config_path)["fit"]
    d = Float64(c["d"])
    diam_x = Float64(c["diam_x"])
    diam_y = Float64(c["diam_y"])
    lambda = Float64(c["lambda"])
    n_modes = Tuple(Int.(c["n_modes"]))
    sampler_size = Tuple(Int.(c["sampler"]))
    lowercase(String(c["solver"])) == "dense" ||
        throw(ArgumentError("only solver='dense' is supported"))

    potential = load_lse_potential(POTENTIAL_CHK)
    V, gradV = potential_functions(potential)
    geometry = make_geometry(d, diam_x, diam_y, V, gradV, GridSampler(sampler_size...))
    coefficients, residual = solve(geometry, n_modes, lambda, DenseSolver())
    residual_inf = maximum(abs, residual)

    fit = FittedEigenfunction(
        coefficients, lambda, n_modes, d, diam_x, diam_y;
        metadata=Dict(
            "potential_checkpoint" => relpath(POTENTIAL_CHK, PROJECT_ROOT),
            "boundary_sampler" => collect(sampler_size),
        ),
    )
    mkpath(dirname(FIT_CHK))
    save_fitted_eigenfunction(FIT_CHK, fit)

    println("Boundary residual (fit grid) infinity norm: ", residual_inf)
    println("Saved fitted eigenfunction to ", FIT_CHK)
    return FIT_CHK
end

config_path = isempty(ARGS) ? joinpath(@__DIR__, "finite-dim.toml") : ARGS[1]
build_eigenfunction(config_path)
