include("../src/potentials/potential_generator.jl")
include("../src/potentials/potential_interface.jl")

using .PotentialGenerator
using .PotentialInterface
using Test

smooth_max_reference(x, y, strength) = max(x, y) + log1p(exp(strength * (min(x, y) - max(x, y)))) / strength

struct TestCorePotential <: AbstractCorePotential
    Lx::Float64
    Ly::Float64
end

PotentialInterface.core_value(p::TestCorePotential, x::Real, y::Real) = x + 2y
PotentialInterface.core_gradient(p::TestCorePotential, x::Real, y::Real) = (1.0, 2.0)
PotentialInterface.potential_domain(p::TestCorePotential) = (Lx=p.Lx, Ly=p.Ly)

struct TestWingPotential <: AbstractWingPotential end

PotentialInterface.wing_value(p::TestWingPotential, x::Real, y::Real) = 3x - y
PotentialInterface.wing_gradient(p::TestWingPotential, x::Real, y::Real) = (3.0, -1.0)

@testset "SmoothMaxPotential composition" begin
    core = TestCorePotential(2.0, 1.0)
    wing = TestWingPotential()
    pot = SmoothMaxPotential(core, wing; smooth_max_strength=7.0)
    x, y = 0.25, 0.5
    expected = smooth_max_reference(core_value(core, x, y), wing_value(wing, x, y), 7.0)

    @test potential_domain(pot) == (Lx=2.0, Ly=1.0)
    @test potential_value(pot, x, y) ≈ expected

    gx, gy = potential_gradient(pot, x, y)
    @test isfinite(gx)
    @test isfinite(gy)
    @test (gx, gy) != core_gradient(core, x, y)
    @test (gx, gy) != wing_gradient(wing, x, y)
end

@testset "Generated potential core adapter" begin
    data = PotentialData(
        0.5 * pi,
        1.0,
        Float64[],
        0,
        0.0,
        zeros(Float64, 3, 1),
        1.0,
        1.0,
        2.0,
        2.0,
    )
    raw = PotentialGenerator.build_potential_functions(data)
    core = GeneratedCorePotential(raw)

    @test core_value(core, 0.25, 0.4) == PotentialGenerator.V(raw, 0.25, 0.4)
    @test core_gradient(core, 0.25, 0.4) == Tuple(PotentialGenerator.∇V(raw, 0.25, 0.4))
    @test potential_domain(core) == (Lx=raw.data.Lx, Ly=raw.data.Ly)
end

@testset "LSE core potential adapter" begin
    model = LSEModel(
        [1.0 0.0; 0.0 1.0],
        [0.0, 0.2],
        0.5,
    )
    core = LSECorePotential(model; Lx=3.0, Ly=4.0)

    @test core_value(core, 0.25, -0.5) == predict(model, 0.25, -0.5)
    @test core_gradient(core, 0.25, -0.5) == gradient(model, 0.25, -0.5)
    @test potential_domain(core) == (Lx=3.0, Ly=4.0)

    mktempdir() do dir
        path = joinpath(dir, "lse_core.chk")
        save_lse_model(path, model)

        loaded = load_lse_core_potential(checkpoint_path=path, Lx=3.0, Ly=4.0)
        @test loaded isa LSECorePotential
        @test potential_domain(loaded) == (Lx=3.0, Ly=4.0)
        @test core_value(loaded, 0.25, -0.5) == core_value(core, 0.25, -0.5)
        @test core_gradient(loaded, 0.25, -0.5) == core_gradient(core, 0.25, -0.5)
    end
end

@testset "LSE wing potential adapter" begin
    model = LSEModel(
        [1.0 0.0; 0.0 1.0],
        [0.0, 0.2],
        0.5,
    )
    wing = LSEWingPotential(model; Lx=3.0, Ly=4.0, scale=7.0)

    @test wing_value(wing, 3.25, -0.5) == 7.0 * predict(model, 3.25, -0.5)
    gx, gy = wing_gradient(wing, 3.25, -0.5)
    expected_gx, expected_gy = gradient(model, 3.25, -0.5)
    @test (gx, gy) == (7.0 * expected_gx, 7.0 * expected_gy)
    @test potential_domain(wing) == (Lx=3.0, Ly=4.0)

    mktempdir() do dir
        path = joinpath(dir, "lse_wing.chk")
        save_lse_model(path, model)

        loaded = load_lse_wing_potential(checkpoint_path=path, Lx=3.0, Ly=4.0, scale=7.0)
        @test loaded isa LSEWingPotential
        @test potential_domain(loaded) == (Lx=3.0, Ly=4.0)
        @test wing_value(loaded, 3.25, -0.5) == wing_value(wing, 3.25, -0.5)
        @test wing_gradient(loaded, 3.25, -0.5) == wing_gradient(wing, 3.25, -0.5)
    end
end

@testset "Joined LSE potential" begin
    core_model = LSEModel(
        [1.0 0.0; 0.0 1.0],
        [0.0, 0.2],
        0.5,
    )
    wing_model = LSEModel(
        [2.0 -1.0; 3.0 4.0],
        [0.1, -0.3],
        0.5,
    )
    core = LSECorePotential(core_model; Lx=3.0, Ly=4.0)
    wing = LSEWingPotential(wing_model; Lx=3.0, Ly=4.0, scale=7.0)
    joined = join_lse_potentials(core, wing)

    expected_model = LSEModel(
        hcat(core_model.A, 7.0 .* wing_model.A),
        vcat(core_model.b, 7.0 .* wing_model.b),
        core_model.temperature,
    )

    @test joined isa JoinedLSEPotential
    @test joined.model.A == expected_model.A
    @test joined.model.b == expected_model.b
    @test joined.model.temperature == expected_model.temperature
    @test potential_value(joined, 0.25, -0.5) == predict(expected_model, 0.25, -0.5)
    @test potential_gradient(joined, 0.25, -0.5) == gradient(expected_model, 0.25, -0.5)
    @test potential_domain(joined) == (Lx=3.0, Ly=4.0)

    mismatched_wing = LSEWingPotential(
        LSEModel(wing_model.A, wing_model.b, 0.25);
        Lx=3.0,
        Ly=4.0,
        scale=7.0,
    )
    @test_throws ArgumentError join_lse_potentials(core, mismatched_wing)
end

@testset "Wing potential adapters" begin
    handmade = HandmadeWingPotential(1.0; scale=10.0)
    nonconvex = NonConvexWingPotential(1.0; anchor=0.5, scale=10.0)

    for wing in (handmade, nonconvex)
        value = wing_value(wing, 1.5, 0.25)
        gx, gy = wing_gradient(wing, 1.5, 0.25)

        @test isfinite(value)
        @test isfinite(gx)
        @test isfinite(gy)
    end
end
