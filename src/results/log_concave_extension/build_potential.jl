# Phase 1: build the global log-sum-exp potential and its verified normalization.
#
# This is computed ONCE and serialized. Both this (log-concave) pipeline and the
# finite_dim pipeline load the resulting artifacts by their fixed paths, so the
# output locations below must not change:
#   checkpoints/log_concave_extension/high-resolution/lse_global_potential.chk
#   results/log_concave_extension/high-resolution/0-make-potential.toml   (nested
#     result.potential_constants.{sup_pot_val,sup_pot_grad} is read by finite_dim)
#
# Run once from the repository root:
#   julia --project=. src/results/log_concave_extension/build_potential.jl [config.toml]
# The config defaults to high-resolution.toml (the shared, canonical potential).

using IntervalArithmetic
using TaylorModels: inf, sup
using TOML

include(joinpath(@__DIR__, "..", "..", "potentials", "potential_generator.jl"))
include(joinpath(@__DIR__, "..", "..", "potentials", "lse_regression.jl"))
include(joinpath(@__DIR__, "..", "..", "potentials", "potential_lab.jl"))
include(joinpath(@__DIR__, "..", "..", "potentials", "lse_potential.jl"))

using .PotentialGenerator
using .PotentialLab: HandmadeWingPotential, wing_value
using .LSERegression: LSEModel, fit_lse_model
using .LSEPotentials: LSEPotential, default_domain, potential_gradient, potential_value
using .LSEPotentials: save_lse_potential, verified_normalization

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const POTENTIAL_CHK = joinpath(
    PROJECT_ROOT, "checkpoints", "log_concave_extension", "high-resolution",
    "lse_global_potential.chk",
)
const POTENTIAL_TOML = joinpath(
    PROJECT_ROOT, "results", "log_concave_extension", "high-resolution",
    "0-make-potential.toml",
)

int_tuple(v) = Tuple(Int.(v))
interval_table(x) = Dict("lo" => inf(x), "hi" => sup(x))

function build_potential(config_path::AbstractString)
    p = TOML.parsefile(config_path)["potential"]

    x_domain = Tuple(Float64.(p["x_domain"]))
    y_domain = Tuple(Float64.(p["y_domain"]))
    wing_x_max = Float64(p["wing_x_max"])
    quadrant_factor = Float64(p["quadrant_factor"])
    core_Lx = maximum(abs, x_domain)
    core_Ly = maximum(abs, y_domain)
    temperature = Float64(p["temperature"])
    core_strength = Float64(p["core_strength"])
    wing_strength = Float64(p["wing_strength"])
    core_fit_grid = int_tuple(p["core_fit_grid"])
    wing_fit_grid = int_tuple(p["wing_fit_grid"])
    normalization_core_cells = int_tuple(p["normalization_core_cells"])
    normalization_wing_cells = int_tuple(p["normalization_wing_cells"])

    core_potential0 = PotentialGenerator.generate_potential()
    V_core(x, y) = core_strength * (
        PotentialGenerator.V(core_potential0, x, y) -
        PotentialGenerator.V(core_potential0, 0.0, 0.0)
    )
    wing_potential0 = HandmadeWingPotential(
        core_Lx; anchor=V_core(core_Lx, 0.0), scale=wing_strength,
    )
    V_wing(x, y) = wing_value(wing_potential0, x, y)

    core_lse_model = fit_lse_model(
        V_core; x_domain, y_domain,
        nx=core_fit_grid[1], ny=core_fit_grid[2], temperature,
    )
    wing_lse_model = fit_lse_model(
        V_wing; x_domain=(core_Lx, wing_x_max), y_domain,
        nx=wing_fit_grid[1], ny=wing_fit_grid[2], temperature,
    )
    full_lse_model = LSEModel(
        hcat(core_lse_model.A, wing_lse_model.A),
        vcat(core_lse_model.b, wing_lse_model.b),
        temperature,
    )

    domain = default_domain(; Lx=core_Lx, Ly=core_Ly, x_max=wing_x_max, quadrant_factor)
    potential = LSEPotential(full_lse_model; domain)
    sup_pot_val = potential_value(potential, domain.x_max, domain.Ly)
    gx_corner, gy_corner = potential_gradient(potential, domain.x_max, domain.Ly)
    sup_pot_grad = hypot(gx_corner, gy_corner)
    normalization = verified_normalization(
        potential;
        cells_core=normalization_core_cells, cells_wing=normalization_wing_cells,
    )
    wing_mass = normalization.Z_wing / normalization.Z
    potential = LSEPotential(
        full_lse_model; domain,
        normalization=(lo=inf(normalization.Z), hi=sup(normalization.Z)),
        wing_mass=(lo=inf(wing_mass), hi=sup(wing_mass)),
    )

    mkpath(dirname(POTENTIAL_CHK))
    save_lse_potential(POTENTIAL_CHK, potential)

    data = Dict(
        "step" => "0-make-potential",
        "outputs" => Dict("potential_checkpoint" => relpath(POTENTIAL_CHK, PROJECT_ROOT)),
        "parameters" => Dict(
            "x_domain" => collect(x_domain),
            "y_domain" => collect(y_domain),
            "wing_x_max" => wing_x_max,
            "quadrant_factor" => quadrant_factor,
            "temperature" => temperature,
            "core_strength" => core_strength,
            "wing_strength" => wing_strength,
            "core_fit_grid" => collect(core_fit_grid),
            "wing_fit_grid" => collect(wing_fit_grid),
            "normalization_core_cells" => collect(normalization_core_cells),
            "normalization_wing_cells" => collect(normalization_wing_cells),
        ),
        "result" => Dict(
            # NOTE: nested schema below is a hard contract with finite_dim.
            "potential_constants" => Dict(
                "sup_pot_val" => sup_pot_val,
                "sup_pot_grad" => sup_pot_grad,
            ),
            "unnormalized_core_mass" => interval_table(normalization.Z_core),
            "unnormalized_wing_mass" => interval_table(normalization.Z_wing),
            "normalization_constant" => interval_table(normalization.Z),
            "relative_wing_mass" => interval_table(wing_mass),
        ),
    )
    mkpath(dirname(POTENTIAL_TOML))
    open(io -> TOML.print(io, data; sorted=true), POTENTIAL_TOML, "w")

    println("Saved global LSE potential to ", POTENTIAL_CHK)
    println("Result saved to ", POTENTIAL_TOML)
    return POTENTIAL_TOML
end

config_path = isempty(ARGS) ? joinpath(@__DIR__, "high-resolution.toml") : ARGS[1]
build_potential(config_path)
