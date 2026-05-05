if !isdefined(@__MODULE__, :PotentialGenerator)
    include(joinpath(@__DIR__, "..", "potential_generator.jl"))
end
if !isdefined(@__MODULE__, :LSEModel)
    include(joinpath(@__DIR__, "..", "lse_regression.jl"))
end

using .PotentialGenerator

const DEFAULT_LSE_CORE_CHECKPOINT = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_LSE_CORE_NX = 65
const DEFAULT_LSE_CORE_NY = 129
const DEFAULT_LSE_CORE_TEMPERATURE = 1e-4

function fit_lse_core_potential(;
    nx::Integer=DEFAULT_LSE_CORE_NX,
    ny::Integer=DEFAULT_LSE_CORE_NY,
    temperature::Real=DEFAULT_LSE_CORE_TEMPERATURE,
    checkpoint_path::AbstractString=DEFAULT_LSE_CORE_CHECKPOINT,
)
    pot = PotentialGenerator.generate_potential()
    model = fit_lse_model(
        (x, y) -> PotentialGenerator.V(pot, x, y);
        x_domain=(-pot.data.Lx, pot.data.Lx),
        y_domain=(-pot.data.Ly, pot.data.Ly),
        nx=nx,
        ny=ny,
        temperature=temperature,
    )

    save_lse_model(checkpoint_path, model)
    return (model=model, potential=pot, checkpoint_path=checkpoint_path)
end

function main()
    fit = fit_lse_core_potential()
    println("Saved LSE core potential: ", fit.checkpoint_path)
    return fit
end

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
