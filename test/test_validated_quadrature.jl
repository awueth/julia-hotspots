include("../src/quadrature/validated_quadrature.jl")

using .ValidatedQuadrature
using TaylorModels
using Test

contains(result, expected) = inf(result) <= expected <= sup(result)

@testset "Interval-box quadrature" begin
    rectangular_domain = (0.0 .. 2.0, -1.0 .. 3.0)

    evaluations = Ref(0)
    constant_result = integrate_box_cells(
        (_x, _y) -> begin
            evaluations[] += 1
            return 3.0
        end,
        rectangular_domain;
        cells=(2, 3),
    )
    @test evaluations[] == 6
    @test contains(constant_result, 24.0)
    @test !isguaranteed(constant_result)

    interval_constant_result = integrate_box_cells(
        (_x, _y) -> 2.0 .. 3.0,
        rectangular_domain;
        cells=(3, 3),
    )
    @test inf(interval_constant_result) <= 16.0
    @test sup(interval_constant_result) >= 24.0
    @test isguaranteed(interval_constant_result)

    affine_result = integrate_box_cells(
        (x, y) -> x + 2y,
        rectangular_domain;
        cells=(3, 3),
    )
    @test contains(affine_result, 24.0)

    polynomial_result = integrate_box_cells(
        (x, y) -> x^2 * y,
        [0.0 .. 1.0, 0.0 .. 2.0];
        cells=(4, 4),
    )
    @test contains(polynomial_result, 2 / 3)
end

@testset "Taylor-model quadrature" begin
    rectangular_domain = (0.0 .. 2.0, -1.0 .. 3.0)

    affine_result = integrate_taylor_cells(
        (x, y) -> x + 2y,
        rectangular_domain;
        order=4,
        cells=(3, 3),
    )
    @test contains(affine_result, 24.0)

    polynomial_result = integrate_taylor_cells(
        (x, y) -> x^2 * y,
        [0.0 .. 1.0, 0.0 .. 2.0];
        order=4,
        cells=(4, 4),
    )
    @test contains(polynomial_result, 2 / 3)

    exponential_result = integrate_taylor_cells(
        (x, y) -> exp(x + y),
        (0.0 .. 1.0, 0.0 .. 1.0);
        order=8,
        cells=(4, 4),
    )
    @test contains(exponential_result, (exp(1) - 1)^2)

    reciprocal_exponential_result = integrate_taylor_cells(
        (x, _y) -> inv(exp(2x) + exp(-2x)),
        [0.0 .. (0.5pi), 0.0 .. 1.0];
        order=8,
        cells=(8, 8),
    )
    reciprocal_exponential_exact = atan(sinh(pi)) / 4
    @test contains(reciprocal_exponential_result, reciprocal_exponential_exact)
    @test isfinite(inf(reciprocal_exponential_result))
    @test isfinite(sup(reciprocal_exponential_result))

    lower_order_result = integrate_taylor_cells(
        (x, y) -> exp(x - y),
        rectangular_domain;
        order=3,
        cells=(2, 2),
    )
    higher_order_result = integrate_taylor_cells(
        (x, y) -> exp(x - y),
        rectangular_domain;
        order=6,
        cells=(2, 2),
    )
    exact = (exp(2) - 1) * (exp(1) - exp(-3))
    @test contains(lower_order_result, exact)
    @test contains(higher_order_result, exact)
end