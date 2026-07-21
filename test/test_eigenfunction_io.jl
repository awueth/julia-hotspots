include("../src/functions/mps_function.jl")

using .MPSFunction
using Serialization
using TaylorModels
using Test

contains(interval_value, value) = inf(interval_value) <= value <= sup(interval_value)

function reference_value_gradient(fit, x::Float64, y::Float64)
    lambda_x, lambda_y, _ = fitted_eigenvalues(fit)
    value = zero(x)
    grad_x = zero(x)
    grad_y = zero(x)

    for i in eachindex(fit.coefficients)
        axial_value, (axial_grad_x, axial_grad_y) =
            axial_basis(lambda_x[i], lambda_y[i], fit.diam_x, fit.diam_y, x, y)
        value += fit.coefficients[i] * axial_value
        grad_x += fit.coefficients[i] * axial_grad_x
        grad_y += fit.coefficients[i] * axial_grad_y
    end

    return value, grad_x, grad_y
end

@testset "FittedEigenfunction IO and evaluation" begin
    fit = FittedEigenfunction(
        [0.7, -0.2, 0.4, 0.1],
        3.0,
        (2, 2),
        Inf,
        4.0 * pi,
        2.0;
        metadata=Dict(:source => "test"),
    )

    @test fit isa FittedEigenfunction{Float64}
    @test fit.coefficients isa Vector{Float64}
    @test typeof(fit.d) === Float64
    @test fit.metadata == Dict{String,Any}("source" => "test")

    interval_fit = intervalize(fit)
    @test interval_fit isa FittedEigenfunction{<:Interval}
    @test all(isguaranteed, interval_fit.coefficients)
    @test isguaranteed(interval_fit.λ)
    @test isguaranteed(interval_fit.diam_x)
    @test isguaranteed(interval_fit.diam_y)
    @test interval_fit.d == Inf
    @test !(interval_fit.d isa Interval)

    for (x, y) in ((0.3, 0.2), (1.1, -0.4), (0.7, 0.9))
        actual = value_gradient(fit, x, y)
        expected = reference_value_gradient(fit, x, y)
        @test all(isapprox.(actual, expected; rtol=1e-10, atol=1e-12))
    end

    x = 0.3
    y = 0.2
    scalar_value, scalar_grad_x, scalar_grad_y = value_gradient(fit, x, y)
    interval_value, interval_grad_x, interval_grad_y =
        value_gradient(interval_fit, interval(x), interval(y))

    @test isguaranteed(interval_value)
    @test isguaranteed(interval_grad_x)
    @test isguaranteed(interval_grad_y)
    @test contains(interval_value, scalar_value)
    @test contains(interval_grad_x, scalar_grad_x)
    @test contains(interval_grad_y, scalar_grad_y)

    finite_fit = FittedEigenfunction([1.0], 1.0, (1, 1), 4.0, 2.0, 2.0)
    @test_throws ArgumentError value_gradient(finite_fit, 0.1, 0.2)

    @test u(fit, x, y, 1.0) == scalar_value
    evaluator = prepare_u(fit)
    @test evaluator isa InfiniteEvaluator{Float64}
    @test u(evaluator, x, y, 1.0) == scalar_value

    radial_fit = FittedEigenfunction([1.0], 1.0, (1, 1), Inf, 4pi, 2.0)
    radial_outer = u(radial_fit, x, y, 1.0)
    radial_eigenvalue = radial_fit.λ - (pi / radial_fit.diam_x)^2
    @test u(radial_fit, x, y, 0.0) ≈ radial_outer * exp(radial_eigenvalue / 8)

    radial_interval_fit = intervalize(radial_fit)
    radial_interval_value = u(
        radial_interval_fit,
        interval(x),
        interval(y),
        interval(0.0),
    )
    @test isguaranteed(radial_interval_value)
    @test contains(radial_interval_value, u(radial_fit, x, y, 0.0))

    radial_interval_evaluator = prepare_u(radial_interval_fit)
    prepared_interval_value = u(
        radial_interval_evaluator,
        interval(x),
        interval(y),
        interval(0.0),
    )
    @test isguaranteed(prepared_interval_value)
    @test contains(
        prepared_interval_value,
        u(radial_fit, x, y, 0.0),
    )
    @test_throws ArgumentError prepare_u(finite_fit)

    mktempdir() do dir
        path = joinpath(dir, "fit.chk")
        save_fitted_eigenfunction(path, fit)

        loaded = load_fitted_eigenfunction(path)
        @test loaded isa FittedEigenfunction{Float64}
        @test loaded.coefficients == fit.coefficients
        @test loaded.λ == fit.λ
        @test loaded.n_modes == fit.n_modes
        @test loaded.d == fit.d
        @test loaded.diam_x == fit.diam_x
        @test loaded.diam_y == fit.diam_y
        @test loaded.metadata == fit.metadata

        loaded_interval = load_fitted_eigenfunction(path; intervalized=true)
        @test loaded_interval isa FittedEigenfunction{<:Interval}
        @test all(isguaranteed, loaded_interval.coefficients)

        struct_path = joinpath(dir, "struct-fit.chk")
        open(struct_path, "w") do io
            serialize(io, fit)
        end
        loaded_struct = load_fitted_eigenfunction(struct_path)
        @test loaded_struct.coefficients == fit.coefficients
        @test loaded_struct.metadata == fit.metadata
    end
end
