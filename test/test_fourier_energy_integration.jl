include("../src/functions/mps_function.jl")
using .MPSFunction
include("../src/solver/fourier_energy_integration.jl")

using IntervalArithmetic
using Test

contains_interval(x, value) = inf(x) <= value <= sup(x)

@testset "Analytic Fourier energy integration" begin
    @testset "single mode closed form" begin
        fit = FittedEigenfunction([1.0], 1.0, (1, 1), Inf, 4pi, 2.0)
        interval_fit = intervalize(fit)
        expansion = prepare_fourier_energy(interval_fit)
        linf = fourier_linf_bounds(interval_fit)
        domain = [interval(0.0, 2pi), interval(0.0, 1.0)]

        energies = fourier_cell_energies(expansion, domain; cells=(4, 3))
        @test size(expansion.denominator_coefficients) == (2, 1)
        @test contains_interval(sum(energies.denominator), 1 / 4)
        @test contains_interval(sum(energies.numerator), 1 / 64)
        @test all(isguaranteed, energies.denominator)
        @test all(isguaranteed, energies.numerator)
        expected_u_linf = inv(sqrt(4pi))
        @test contains_interval(linf.u, expected_u_linf)
        @test contains_interval(linf.ux, expected_u_linf / 4)
        @test contains_interval(linf.uy, 0.0)
        @test contains_interval(linf.gradient, expected_u_linf / 4)
        @test all(isguaranteed, linf)

        calls = Ref(0)
        weighted = integrate_weighted_fourier_energy(
            expansion,
            (_x, _y) -> begin
                calls[] += 1
                interval(2.0)
            end,
            domain;
            cells=(4, 3),
        )
        @test calls[] == 12
        @test contains_interval(weighted.denominator, 1 / 2)
        @test contains_interval(weighted.numerator, 1 / 32)
        @test isguaranteed(weighted.denominator)
        @test isguaranteed(weighted.numerator)
    end

    @testset "multi-mode point values and Parseval identities" begin
        coefficients = [0.7, -0.2, 0.4, 0.1, -0.3, 0.25]
        fit = FittedEigenfunction(coefficients, 3.0, (3, 2), Inf, 4pi, 2.0)
        interval_fit = intervalize(fit)
        expansion = prepare_fourier_energy(interval_fit)
        linf = fourier_linf_bounds(interval_fit)

        @test size(expansion.denominator_coefficients) == (6, 3)
        @test all(isguaranteed, expansion.denominator_coefficients)
        @test all(isguaranteed, expansion.numerator_coefficients)

        for (x, y) in ((0.3, 0.2), (1.1, 0.7), (2.4, 0.9))
            u_value, ux, uy = value_gradient(fit, x, y)
            values = _fourier_energy_values(expansion, interval(x), interval(y))
            @test contains_interval(values.denominator, u_value^2)
            @test contains_interval(values.numerator, ux^2 + uy^2)
            @test abs(u_value) <= sup(linf.u)
            @test abs(ux) <= sup(linf.ux)
            @test abs(uy) <= sup(linf.uy)
            @test hypot(ux, uy) <= sup(linf.gradient)
        end

        domain = [interval(0.0, fit.diam_x / 2), interval(0.0, fit.diam_y / 2)]
        energies = fourier_cell_energies(expansion, domain; cells=(7, 5))
        expected_denominator = sum(abs2, coefficients) / 4

        lambda_x, lambda_y, _ = fitted_eigenvalues(fit)
        expected_numerator = sum(
            abs2(coefficients[i]) * (lambda_x[i] + lambda_y[i]) / 4
            for i in eachindex(coefficients)
        )
        @test contains_interval(sum(energies.denominator), expected_denominator)
        @test contains_interval(sum(energies.numerator), expected_numerator)
    end

    @testset "representation assumptions" begin
        scalar_fit = FittedEigenfunction([1.0], 1.0, (1, 1), Inf, 4pi, 2.0)
        @test_throws MethodError prepare_fourier_energy(scalar_fit)

        finite_fit = intervalize(FittedEigenfunction([1.0], 1.0, (1, 1), 4.0, 4pi, 2.0))
        @test_throws ArgumentError prepare_fourier_energy(finite_fit)

        expansion = prepare_fourier_energy(intervalize(scalar_fit))
        domain = [interval(0.0, 2pi), interval(0.0, 1.0)]
        @test_throws ArgumentError integrate_weighted_fourier_energy(
            expansion,
            (_x, _y) -> interval(-1.0),
            domain;
            cells=(1, 1),
        )
    end
end
