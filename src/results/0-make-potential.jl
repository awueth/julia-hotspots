# Global potential checkpoint builder.
#
# Builds the saved core+wing `LSEPotential` from the generated core potential
# and the hand-built wing potential. The core and wing are fitted separately by
# log-sum-exp regression, then combined by taking the LSE over the union of their
# affine plane sets, matching `writeup/construction.typ` ("Parametrizing the
# potential").
#
# The resulting model is stored with its domain metadata and a validated
# quadrature enclosure of its normalization constant, then serialized to
# `CHECKPOINT_PATH`. Downstream code should load that checkpoint instead of
# running this builder.

include("../potentials/potential_generator.jl")
include("../potentials/lse_regression.jl")
include("../potentials/potential_lab.jl")
include("../potentials/lse_potential.jl")

using TOML
using .PotentialGenerator
using .PotentialLab: HandmadeWingPotential, wing_value
using .LSERegression: LSEModel, fit_lse_model
using .LSEPotentials: LSEPotential, default_domain, save_lse_potential, verified_normalization
using TaylorModels: inf, sup

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_global_potential.chk")
const RESULT_DIR = joinpath(PROJECT_ROOT, "results")
const RESULT_PATH = joinpath(RESULT_DIR, "0-make-potential.toml")

project_relative(path) = relpath(normpath(path), PROJECT_ROOT)
interval_bounds(x) = Dict("lo" => inf(x), "hi" => sup(x))

x_domain = (-0.5 * pi, 0.5 * pi)
y_domain = (-1.0, 1.0)
wing_x_max = 2.0 * pi
temperature = 0.2
core_strength = 10.0
wing_strength = 1e6

# --- Core and wing target potentials  --------------------------
core_potential0 = PotentialGenerator.generate_potential()
V_core(x, y) = core_strength * PotentialGenerator.V(core_potential0, x, y)

wing_potential0 = HandmadeWingPotential(0.5 * pi; anchor=V_core(0.5 * pi, 0.0), scale=wing_strength)
V_wing(x, y) = wing_value(wing_potential0, x, y)

# --- Fit LSE plane sets and glue them --------------------------------------
core_lse_model = fit_lse_model(
    V_core;
    x_domain=x_domain,
    y_domain=y_domain,
    nx=128,
    ny=128,
    temperature=temperature,
)

wing_lse_model = fit_lse_model(
    V_wing;
    x_domain=(0.5 * pi, wing_x_max),
    y_domain=y_domain,
    nx=128,
    ny=128,
    temperature=temperature,
)

full_lse_model = LSEModel(
    hcat(core_lse_model.A, wing_lse_model.A),
    vcat(core_lse_model.b, wing_lse_model.b),
    temperature,
)

# --- Wrap as the product, normalize, and save ------------------------------
domain = default_domain(Lx=0.5 * pi, Ly=1.0, x_max=wing_x_max)
potential = LSEPotential(full_lse_model; domain=domain)

normalization = verified_normalization(
    potential;
    cells_core=(128, 128),
    cells_wing=(128, 128),
)
wing_mass = normalization.Z_wing / normalization.Z
potential = LSEPotential(
    full_lse_model;
    domain=domain,
    normalization=(lo=inf(normalization.Z), hi=sup(normalization.Z)),
    wing_mass=(lo=inf(wing_mass), hi=sup(wing_mass)),
)

save_lse_potential(CHECKPOINT_PATH, potential)

mkpath(RESULT_DIR)
open(RESULT_PATH, "w") do io
    TOML.print(io, Dict(
        "step" => "0-make-potential",
        "outputs" => Dict(
            "potential_checkpoint" => project_relative(CHECKPOINT_PATH),
        ),
        "parameters" => Dict(
            "x_domain" => collect(x_domain),
            "y_domain" => collect(y_domain),
            "wing_x_max" => wing_x_max,
            "temperature" => temperature,
            "core_strength" => core_strength,
            "wing_strength" => wing_strength,
            "core_fit_grid" => [128, 128],
            "wing_fit_grid" => [128, 128],
            "normalization_core_cells" => [128, 128],
            "normalization_wing_cells" => [128, 128],
        ),
        "result" => Dict(
            "unnormalized_core_mass" => interval_bounds(normalization.Z_core),
            "unnormalized_wing_mass" => interval_bounds(normalization.Z_wing),
            "normalization_constant" => interval_bounds(normalization.Z),
            "relative_wing_mass" => interval_bounds(wing_mass),
        ),
    ); sorted=true)
end

println("Saved global LSE potential to ", CHECKPOINT_PATH)
println("Result certificate saved to ", RESULT_PATH)
println("Interval arithmetic normalization Z = ", normalization.Z)
println("Interval arithmetic core mass Z_core = ", normalization.Z_core)
println("Interval arithmetic wing mass Z_wing = ", normalization.Z_wing)
println("Interval arithmetic wing mass Z_wing/Z = ", wing_mass)
