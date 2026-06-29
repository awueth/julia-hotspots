if !isdefined(@__MODULE__, :PotentialLab)
    include(joinpath(@__DIR__, "..", "potentials", "potential_lab.jl"))
end
if !isdefined(@__MODULE__, :LSERegression)
    include(joinpath(@__DIR__, "..", "potentials", "lse_regression.jl"))
end

using .PotentialLab
using .LSERegression

const DEFAULT_LSE_WING_CHECKPOINT = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_LSE_WING_NX = 128
const DEFAULT_LSE_WING_NY = 128
const DEFAULT_LSE_WING_TEMPERATURE = 1e-2
const DEFAULT_LSE_WING_LX = 0.5 * pi
const DEFAULT_LSE_WING_LY = 1.0
const DEFAULT_LSE_WING_LENGTH = 5.0
const DEFAULT_LSE_WING_FIT_SCALE = 1.0

function fit_lse_wing_potential(;
    nx::Integer=DEFAULT_LSE_WING_NX,
    ny::Integer=DEFAULT_LSE_WING_NY,
    temperature::Real=DEFAULT_LSE_WING_TEMPERATURE,
    Lx::Real=DEFAULT_LSE_WING_LX,
    Ly::Real=DEFAULT_LSE_WING_LY,
    wing_length::Real=DEFAULT_LSE_WING_LENGTH,
    scale::Real=DEFAULT_LSE_WING_FIT_SCALE,
    anchor::Real=0.0,
    checkpoint_path::AbstractString=DEFAULT_LSE_WING_CHECKPOINT,
)
    wing = HandmadeWingPotential(Lx; anchor=anchor, scale=scale)
    x_min = Float64(Lx)
    x_max = x_min + Float64(wing_length)
    Ly_value = Float64(Ly)

    model = fit_lse_model(
        (x, y) -> wing_value(wing, x, y);
        x_domain=(x_min, x_max),
        y_domain=(-Ly_value, Ly_value),
        nx=nx,
        ny=ny,
        temperature=temperature,
    )

    save_lse_model(checkpoint_path, model)
    return (model=model, potential=wing, checkpoint_path=checkpoint_path)
end

function main()
    fit = fit_lse_wing_potential()
    println("Saved LSE wing potential: ", fit.checkpoint_path)
    return fit
end

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
