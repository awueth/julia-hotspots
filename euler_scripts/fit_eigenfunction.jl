include("../src/solver/solver.jl")
include("../src/solver/eigenfunction_io.jl")
include("../src/potentials/potential_interface.jl")

using .PotentialInterface

const DEFAULT_OUTPUT_PATH = joinpath(@__DIR__, "..", "checkpoints", "fitted_eigenfunction.chk")

const DEFAULT_EPSILON = 10.0
const DEFAULT_WING_LENGTH = 1.5 * pi
const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_SMOOTH_MAX_STRENGTH = 1e-4
# const DEFAULT_LSE_WING_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_WING_SCALE = 1e6

ε = DEFAULT_EPSILON
core = load_lse_core_potential(checkpoint_path=DEFAULT_LSE_CORE_CHECKPOINT_PATH; Ly=1.0)
domain = potential_domain(core)
wing = HandmadeWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE)
#wing = NonConvexWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE, anchor=core_value(core, domain.Lx, domain.Ly))
pot = SmoothMaxPotential(core, wing; smooth_max_strength=DEFAULT_SMOOTH_MAX_STRENGTH)
# wing = load_lse_wing_potential(
#     checkpoint_path=DEFAULT_LSE_WING_CHECKPOINT_PATH;
#     Lx=domain.Lx,
#     Ly=domain.Ly,
#     scale=DEFAULT_WING_SCALE,
# )
# pot = join_lse_potentials(core, wing)
domain = potential_domain(pot)
d = Inf
diam_x = 2.0 * (domain.Lx + DEFAULT_WING_LENGTH)
diam_y = 2.0 * domain.Ly
# sampler = FibonacciSampler(256 * 64)
sampler = GridSampler(512, 128)
n_modes = (256, 64)
λ = 3.87081387296969

V, gradV = potential_functions(pot; scale=ε)

geometry = make_geometry(d, diam_x, diam_y, V, gradV, sampler)

# solver = QRSolver(geometry, n_modes, λ, FibonacciSampler(512))
solver = DenseSolver()

# λ, _ = optimize_eigenvalue(geometry, n_modes, (3.85, 4.0), solver)
coefficients, residual = solve(geometry, n_modes, λ, solver)

fit = FittedEigenfunction(
    coefficients,
    λ,
    n_modes,
    d,
    diam_x,
    diam_y;
    metadata=Dict(
        "core_potential_checkpoint" => DEFAULT_LSE_CORE_CHECKPOINT_PATH,
        "epsilon" => ε,
        "wing_length" => DEFAULT_WING_LENGTH,
        "wing_scale" => DEFAULT_WING_SCALE,
        "n_boundary_points" => n_modes,
        "potential" => "JoinedLSEPotential",
    )
)

mkpath(dirname(DEFAULT_OUTPUT_PATH))
save_fitted_eigenfunction(DEFAULT_OUTPUT_PATH, fit)
println("Saved fitted eigenfunction to ", DEFAULT_OUTPUT_PATH)
