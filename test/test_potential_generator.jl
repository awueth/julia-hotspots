include("../src/potential_generator.jl")

using .PotentialGenerator
using ForwardDiff
using LinearAlgebra
using Test

pot = generate_potential()
diam_x = 4.0 * 2.0 * pot.data.Lx
diam_y = 2.0 * pot.data.Ly
x_grid = range(0, 0.5 * diam_x, length=6)
y_grid = range(0, 0.5 * diam_y, length=6)
points = collect(Iterators.product(x_grid, y_grid))

@testset "Fourier-Chebyshev Storage" begin
    coeffs = pot.data.fourier_chebyshev_coeffs
    @test size(coeffs, 2) == length(pot.data.q_coeffs) + 1
    @test pot.data.Mx_convex >= 0
    @test pot.data.My_convex >= 0
    @test pot.data.M_convex == max(pot.data.Mx_convex, pot.data.My_convex)

    odd_rows = 2:2:size(coeffs, 1)
    @test maximum(abs, coeffs[odd_rows, :]) < 1e-8
end

@testset "Primitive Boundary Constraints" begin
    source_polys = pot.source_polys
    @test source_polys !== nothing

    fit_degree = size(pot.data.fourier_chebyshev_coeffs, 1) - 1
    for mode_n in 0:length(pot.data.q_coeffs)
        coeffs, diag = PotentialGenerator.fit_mode_primitive(mode_n, source_polys[mode_n + 1], pot.data.Lx, pot.data.Ly, fit_degree)
        approx = PotentialGenerator.ChebyshevApprox(coeffs, (-pot.data.Lx, pot.data.Lx))
        dapprox = PotentialGenerator.∂(approx)

        @test abs(dapprox(0.0)) < 1e-8
        @test isapprox(approx(pot.data.Lx), diag.boundary_value; atol=1e-8, rtol=1e-8)
        @test isapprox(dapprox(pot.data.Lx), diag.boundary_slope; atol=1e-8, rtol=1e-8)
    end
end

@testset "Integral Regression" begin
    @test pot.source_polys !== nothing

    xs = range(-pot.data.Lx, pot.data.Lx, length=13)
    ys = range(-pot.data.Ly, pot.data.Ly, length=9)

    max_err = 0.0
    max_ref = 0.0
    for x in xs, y in ys
        reference = PotentialGenerator.V₀_integral_reference(pot.source_polys, pot.data.q_coeffs, x, y, pot.data.Ly)
        approx = V₀(pot, x, y)
        max_err = max(max_err, abs(reference - approx))
        max_ref = max(max_ref, abs(reference))
    end

    rel_err = max_ref < 1e-12 ? max_err : max_err / max_ref
    @test rel_err < 5e-3
end

@testset "Transverse Ly Scaling" begin
    Ly_custom = 2.0
    coeffs = zeros(Float64, 1, 2)
    coeffs[1, 2] = 1.0
    data = PotentialGenerator.PotentialData(
        pot.data.Lx,
        Ly_custom,
        [1.0],
        pot.data.J_HARMONICS,
        0.0,
        coeffs,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    scaled_pot = PotentialGenerator.build_potential_functions(data)

    @test isapprox(V₀(scaled_pot, 0.0, 0.0), 1.0; atol=1e-12)
    @test isapprox(V₀(scaled_pot, 0.0, Ly_custom / 2), 0.0; atol=1e-12)
    @test isapprox(V₀(scaled_pot, 0.0, Ly_custom), -1.0; atol=1e-12)
end

@testset "Symmetry" begin
    @test all(points) do (x, y)
        V_extended(pot, x, y) == V_extended(pot, -x, y) &&
        V_extended(pot, x, y) == V_extended(pot, x, -y)
    end
end

@testset "Convexity Diagnostics" begin
    min_lambda = minimum(points) do (x, y)
        H = ForwardDiff.hessian(p -> V_extended(pot, p[1], p[2]), [x, y])
        eigmin(Symmetric(H))
    end

    @test isfinite(min_lambda)
    println("Minimum sampled Hessian eigenvalue: ", min_lambda)
end
