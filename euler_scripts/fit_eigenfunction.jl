include("../src/solver.jl")
include("../src/eigenfunction_io.jl")
include("../src/potential_interface.jl")

using .PotentialInterface

const DEFAULT_EPSILON = 0.1
const DEFAULT_WING_LENGTH = 5.0
const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_LSE_WING_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_WING_SCALE = 5e6
const DEFAULT_OUTPUT_PATH = joinpath(@__DIR__, "..", "checkpoints", "fitted_eigenfunction.chk")

ε = DEFAULT_EPSILON
core_pot = load_lse_core_potential(checkpoint_path=DEFAULT_LSE_CORE_CHECKPOINT_PATH)
core_domain = potential_domain(core_pot)
wing_pot = load_lse_wing_potential(
    checkpoint_path=DEFAULT_LSE_WING_CHECKPOINT_PATH;
    Lx=core_domain.Lx,
    Ly=core_domain.Ly,
    scale=DEFAULT_WING_SCALE,
)
pot = join_lse_potentials(core_pot, wing_pot)
domain = potential_domain(pot)
d = Inf
wing_length = DEFAULT_WING_LENGTH
diam_x = 2.0 * (domain.Lx + wing_length)
diam_y = 2.0 * domain.Ly
n_boundary_points = 28_672
n_modes = (320, 32)
λ = 1.0539350209179175

V, gradV = potential_functions(pot; scale=ε)

geometry = make_geometry(d, diam_x, diam_y, V, gradV, n_boundary_points)

coefficients, residual = solve_iterative(geometry, n_modes, λ)
println("Infinity norm of boundary residual: ", maximum(abs.(residual)))

fit = FittedEigenfunction(
    coefficients,
    λ,
    n_modes,
    d,
    diam_x,
    diam_y;
    metadata=Dict(
        "core_potential_checkpoint" => DEFAULT_LSE_CORE_CHECKPOINT_PATH,
        "wing_potential_checkpoint" => DEFAULT_LSE_WING_CHECKPOINT_PATH,
        "epsilon" => ε,
        "wing_length" => wing_length,
        "wing_scale" => DEFAULT_WING_SCALE,
        "n_boundary_points" => n_boundary_points,
        "potential" => "JoinedLSEPotential",
    )
)

mkpath(dirname(DEFAULT_OUTPUT_PATH))
save_fitted_eigenfunction(DEFAULT_OUTPUT_PATH, fit)
println("Saved fitted eigenfunction to ", DEFAULT_OUTPUT_PATH)
