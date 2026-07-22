# Lightweight finite-dimensional diagnostic at d = 10^9.
#
# Fits an MPS candidate in memory, samples its physical Neumann residual and
# interior-minus-boundary hot-spot gap, and writes a standalone summary for the
# introduction. The d = 10^18 checkpoint and certificate summary are untouched.
#
# Run from the repository root:
#   julia --project=. src/results/finite_dim/observe_hot_spot.jl

using TOML

include(joinpath(@__DIR__, "..", "..", "potentials", "lse_potential.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "solver.jl"))
include(joinpath(@__DIR__, "..", "..", "solver", "eigenfunction_hot_spot.jl"))

using .LSEPotentials: load_lse_potential, potential_functions
using .MPSFunction: FittedEigenfunction
using .EigenfunctionHotSpot: sampled_hot_spot_difference

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const CONFIG_TOML = joinpath(@__DIR__, "finite-dim.toml")
const POTENTIAL_CHK = joinpath(
    PROJECT_ROOT, "checkpoints", "log_concave_extension", "high-resolution",
    "lse_global_potential.chk",
)
const OUTPUT_TOML = joinpath(
    PROJECT_ROOT, "writeup", "results", "finite_dim", "hot-spot-d1e9", "summary.toml",
)
const OBSERVATION_DIMENSION = 1.0e9

int_tuple(values) = Tuple(Int.(values))

sample_table(sample, face) = Dict(
    "value" => sample.value,
    "face" => String(face),
    "x" => sample.location.x,
    "y" => sample.location.y,
    "r" => sample.location.r,
)

config = TOML.parsefile(CONFIG_TOML)
fit_config = config["fit"]
evaluate_config = config["evaluate"]

d = OBSERVATION_DIMENSION
diam_x = Float64(fit_config["diam_x"])
diam_y = Float64(fit_config["diam_y"])
lambda = Float64(fit_config["lambda"])
n_modes = int_tuple(fit_config["n_modes"])
fit_grid = int_tuple(fit_config["sampler"])
residual_grid = int_tuple(evaluate_config["residual_grid"])
hot_spot_grid = int_tuple(evaluate_config["hot_spot_grid"])
lowercase(String(fit_config["solver"])) == "dense" ||
    throw(ArgumentError("only solver='dense' is supported"))

potential = load_lse_potential(POTENTIAL_CHK)
V, gradV = potential_functions(potential)
geometry = make_geometry(d, diam_x, diam_y, V, gradV, GridSampler(fit_grid...))
coefficients, _ = solve(geometry, n_modes, lambda, DenseSolver())
fit = FittedEigenfunction(
    coefficients, lambda, n_modes, d, diam_x, diam_y;
    metadata=Dict(
        "potential_checkpoint" => relpath(POTENTIAL_CHK, PROJECT_ROOT),
        "boundary_sampler" => collect(fit_grid),
    ),
)
residuals, _, _ = boundary_residual(geometry, coefficients, lambda, n_modes, residual_grid)
residual_inf = maximum(abs, residuals)
hot_spot = sampled_hot_spot_difference(fit, hot_spot_grid)

all(isfinite, (
    residual_inf,
    hot_spot.interior.value,
    hot_spot.boundary.value,
    hot_spot.effect,
)) || error("diagnostic produced a non-finite result")
hot_spot.effect > 0 ||
    error("sampled hot-spot effect is not positive: $(hot_spot.effect)")

summary = Dict(
    "step" => "summary",
    "inputs" => Dict(
        "config" => relpath(CONFIG_TOML, PROJECT_ROOT),
        "potential_checkpoint" => relpath(POTENTIAL_CHK, PROJECT_ROOT),
        "fit_grid" => collect(fit_grid),
        "residual_grid" => collect(residual_grid),
        "hot_spot_grid" => collect(hot_spot_grid),
    ),
    "mps_candidate" => Dict(
        "dimension" => d,
        "lambda" => lambda,
        "n_modes" => collect(n_modes),
    ),
    "residual" => Dict(
        "normal_derivative_inf" => residual_inf,
    ),
    "eigenfunction" => Dict(
        "hot_spot_effect" => hot_spot.effect,
        "interior_maximum" => sample_table(hot_spot.interior, "interior_axis"),
        "boundary_maximum" => sample_table(hot_spot.boundary, hot_spot.boundary.name),
        "boundary_face_maxima" => Dict(
            face.name => sample_table(face, face.name) for face in hot_spot.boundary_faces
        ),
    ),
)

mkpath(dirname(OUTPUT_TOML))
open(io -> TOML.print(io, summary; sorted=true), OUTPUT_TOML, "w")
println("Physical Neumann residual infinity norm: ", residual_inf)
println("Sampled hot-spot effect: ", hot_spot.effect)
println("Summary saved to ", OUTPUT_TOML)
