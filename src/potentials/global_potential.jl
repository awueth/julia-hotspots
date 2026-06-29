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

include("potential_generator.jl")
include("lse_regression.jl")
include("potential_lab.jl")
include("lse_potential.jl")

using .PotentialGenerator
using .PotentialLab: HandmadeWingPotential, wing_value
using .LSERegression: LSEModel, fit_lse_model
using .LSEPotentials: LSEPotential, default_domain, save_lse_potential, verified_normalization

const CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_global_potential.chk")

x_domain = (-0.5 * pi, 0.5 * pi)
y_domain = (-1.0, 1.0)
wing_x_max = 2.0 * pi
temperature = 0.1
core_strength = 1.0
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
potential = LSEPotential(
    full_lse_model;
    domain=domain,
    normalization=(lo=inf(normalization.Z), hi=sup(normalization.Z)),
)

save_lse_potential(CHECKPOINT_PATH, potential)

println("Saved global LSE potential to ", CHECKPOINT_PATH)
println("Interval arithmetic normalization Z = ", normalization.Z)
