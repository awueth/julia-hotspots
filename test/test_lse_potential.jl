include("../src/potentials/lse_potential.jl")

using .LSERegression
using .LSEPotentials
using TaylorModels
using Test

encloses(interval_result, expected) = inf(interval_result) <= expected <= sup(interval_result)

@testset "LSEPotential evaluation matches the model" begin
    model = LSEModel([1.0 0.0; 0.0 1.0], [0.0, 0.2], 0.5)
    p = LSEPotential(model; domain=default_domain(Lx=1.0, Ly=1.0, x_max=2.0))

    for (x, y) in ((0.25, -0.5), (-0.3, 0.7), (1.4, 0.1))
        @test potential_value(p, x, y) ≈ predict(model, x, y)
        @test neg_potential(p, x, y) ≈ -predict(model, x, y)
        @test density(p, x, y) ≈ exp(-predict(model, x, y))
        @test all(potential_gradient(p, x, y) .≈ gradient(model, x, y))
    end

    @test p.domain == default_domain(Lx=1.0, Ly=1.0, x_max=2.0)
end

@testset "Single-plane potential is exactly affine" begin
    model = LSEModel(reshape([1.0, 1.0], 2, 1), [0.0], 0.5)
    p = LSEPotential(model)
    @test potential_value(p, 0.3, 0.4) ≈ 0.7
    @test density(p, 0.3, 0.4) ≈ exp(-0.7)
end

@testset "Interval density encloses the Float64 density" begin
    model = LSEModel([1.0 0.0; 0.0 1.0], [0.0, 0.2], 0.5)
    p = LSEPotential(model; domain=default_domain(Lx=1.0, Ly=1.0, x_max=2.0))

    x = interval(0.2, 0.3)
    y = interval(-0.1, 0.4)
    d_interval = density(p, x, y)

    @test isguaranteed(d_interval)
    @test encloses(d_interval, density(p, 0.25, 0.15))
    @test encloses(d_interval, density(p, mid(x), mid(y)))
end

@testset "Verified normalization of an affine potential" begin
    # Single plane: V = x + y, so density = exp(-(x+y)) integrated exactly.
    model = LSEModel(reshape([1.0, 1.0], 2, 1), [0.0], 0.5)
    p = LSEPotential(model; domain=default_domain(Lx=1.0, Ly=1.0, x_max=2.0))

    normalization = verified_normalization(
        p;
        cells_core=(8, 8),
        cells_wing=(8, 8),
    )
    expected_core = (1 - exp(-1.0))^2
    expected_wing = (exp(-1.0) - exp(-2.0)) * (1 - exp(-1.0))
    expected_Z = 4.0 * (expected_core + expected_wing)

    @test isguaranteed(normalization.Z)
    @test encloses(normalization.quadrant_core, expected_core)
    @test encloses(normalization.quadrant_wing, expected_wing)
    @test encloses(normalization.Z, expected_Z)
    @test_throws MethodError verified_normalization(
        p;
        order=2,
        cells_core=(1, 1),
        cells_wing=(1, 1),
    )

    one_cell = verified_normalization(
        p;
        cells_core=(1, 1),
        cells_wing=(1, 1),
    )
    @test isguaranteed(one_cell.Z)
    @test inf(one_cell.quadrant_core) <= exp(-2.0)
    @test sup(one_cell.quadrant_core) >= 1.0
    @test inf(one_cell.quadrant_wing) <= exp(-3.0)
    @test sup(one_cell.quadrant_wing) >= exp(-1.0)
    @test_throws ArgumentError LSEPotentials._cell_density(
        p,
        (-0.1) .. 0.1,
        0.0 .. 1.0,
    )

    # Normalization is stored as explicit portable Float64 bounds.
    p_normalized = LSEPotential(
        p.model;
        domain=p.domain,
        normalization=(lo=inf(normalization.Z), hi=sup(normalization.Z)),
    )
    @test p_normalized.normalization == (lo=inf(normalization.Z), hi=sup(normalization.Z))
end

@testset "Stored normalization" begin
    model = LSEModel(reshape([1.0, 1.0], 2, 1), [0.0], 0.5)
    p = LSEPotential(model; normalization=(lo=2.0, hi=2.0))
    @test p.normalization == (lo=2.0, hi=2.0)
end

@testset "Save/load round-trips the full product" begin
    model = LSEModel([1.0 0.0; 0.0 1.0], [0.0, 0.2], 0.5)
    domain = default_domain(Lx=1.0, Ly=1.0, x_max=2.0)
    p = LSEPotential(model; domain=domain, normalization=(lo=1.5, hi=1.5))

    mktempdir() do dir
        path = joinpath(dir, "global.chk")
        save_lse_potential(path, p)
        loaded = load_lse_potential(path)

        @test loaded.domain == domain
        @test loaded.normalization == (lo=1.5, hi=1.5)
        @test potential_value(loaded, 0.3, -0.2) ≈ potential_value(p, 0.3, -0.2)
    end
end
