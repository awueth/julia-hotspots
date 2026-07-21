include("../src/solver/eigenfunction_linf_norm.jl")

using IntervalArithmetic
using Test
using .MPSFunction
using .EigenfunctionLinfNorm

contains_interval(x, value) = inf(x) <= value <= sup(x)

@testset "Eigenfunction slice L-infinity norm" begin
    fit = FittedEigenfunction([1.0], 1.0, (1, 1), Inf, 2.0, 2.0)
    expected_linf = inv(sqrt(2.0))

    sampled = sampled_linf_norm(fit, 1.0, (17, 11))
    @test sampled.linf ≈ expected_linf
    @test sampled.location == (x=1.0, y=0.0, r=1.0)
    @test sampled.value ≈ expected_linf

    explicit = sampled_linf_norm(fit, (x=[0.0, 1.0], y=[0.0], r=[1.0]))
    @test explicit.linf ≈ expected_linf
    @test explicit.location == (x=1.0, y=0.0, r=1.0)

    tied = sampled_linf_norm(fit, (x=[0.0], y=[0.25, 0.75], r=[0.2, 0.8]))
    @test tied.linf == 0.0
    @test tied.location == (x=0.0, y=0.25, r=0.2)

    atol = 1e-2
    validated = interval_linf_norm(intervalize(fit), atol)
    @test isguaranteed(validated.linf)
    @test contains_interval(validated.linf, expected_linf)
    @test diam(validated.linf) < atol
    @test !isempty(validated.maximizing_cells)
    @test all(cell -> contains_interval(cell.x, 1.0), validated.maximizing_cells)
    @test all(cell -> contains_interval(cell.r, 1.0), validated.maximizing_cells)

    inner_max_fit = FittedEigenfunction([1.0], 4.0, (1, 1), Inf, 2.0, 2.0)
    λr = inner_max_fit.λ - (pi / inner_max_fit.diam_x)^2
    expected_inner_linf = expected_linf * exp(λr / 8)
    inner_sampled = sampled_linf_norm(inner_max_fit, 0.0, (17, 11))
    outer_sampled = sampled_linf_norm(inner_max_fit, 1.0, (17, 11))
    @test inner_sampled.linf ≈ expected_inner_linf
    @test inner_sampled.linf > outer_sampled.linf

    inner_validated = interval_linf_norm(intervalize(inner_max_fit), atol)
    @test contains_interval(inner_validated.linf, expected_inner_linf)
    @test diam(inner_validated.linf) < atol
    @test all(cell -> contains_interval(cell.r, 0.0), inner_validated.maximizing_cells)
end

@testset "Sampled and interval multimode norms" begin
    fit = FittedEigenfunction(
        [1.0, 0.02, 0.02, 0.005],
        3.0,
        (2, 2),
        Inf,
        4.0 * pi,
        2.0,
    )

    sampled = sampled_linf_norm(fit, 0.0, (33, 25))
    validated = interval_linf_norm(intervalize(fit), 5e-2)
    @test sampled.linf <= sup(validated.linf)
    @test diam(validated.linf) < 5e-2

    finite_fit = FittedEigenfunction([1.0], 1.0, (1, 1), 4.0, 2.0, 2.0)
    finite_sampled = sampled_linf_norm(finite_fit, 0.25, (9, 7))
    @test isfinite(finite_sampled.linf)
    @test finite_sampled.location.r == 0.25
end
