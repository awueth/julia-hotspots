using Test
using LinearAlgebra

include("../src/limit_solvers/spectral_galerkin.jl")

using .SpectralGalerkin

function assemble_advection_reference(basis, domain, grad_V, Nquad)
    bx, by = basis.basis_x, basis.basis_y
    Nx, Ny = SpectralGalerkin.n_modes(bx), SpectralGalerkin.n_modes(by)
    N_basis = Nx * Ny

    x_pts, x_wts = SpectralGalerkin._gl_points_weights(Nquad, domain.x_min, domain.x_max)
    y_pts, y_wts = SpectralGalerkin._gl_points_weights(Nquad, domain.y_min, domain.y_max)
    N_pts = Nquad * Nquad

    Xvals = Matrix{Float64}(undef, Nquad, Nx)
    dXvals = Matrix{Float64}(undef, Nquad, Nx)
    Yvals = Matrix{Float64}(undef, Nquad, Ny)
    dYvals = Matrix{Float64}(undef, Nquad, Ny)

    for (i, x) in enumerate(x_pts), (mi, m) in enumerate(SpectralGalerkin.mode_indices(bx))
        Xvals[i, mi] = SpectralGalerkin.evaluate(bx, m, x)
        dXvals[i, mi] = SpectralGalerkin.evaluate_deriv(bx, m, x)
    end
    for (j, y) in enumerate(y_pts), (ni, n) in enumerate(SpectralGalerkin.mode_indices(by))
        Yvals[j, ni] = SpectralGalerkin.evaluate(by, n, y)
        dYvals[j, ni] = SpectralGalerkin.evaluate_deriv(by, n, y)
    end

    Phi = Matrix{Float64}(undef, N_pts, N_basis)
    dPhi_x = Matrix{Float64}(undef, N_pts, N_basis)
    dPhi_y = Matrix{Float64}(undef, N_pts, N_basis)
    W = Vector{Float64}(undef, N_pts)
    Vx_vec = Vector{Float64}(undef, N_pts)
    Vy_vec = Vector{Float64}(undef, N_pts)

    pt = 1
    for j in 1:Nquad
        for i in 1:Nquad
            W[pt] = x_wts[i] * y_wts[j]
            gx, gy = grad_V(x_pts[i], y_pts[j])
            Vx_vec[pt] = gx
            Vy_vec[pt] = gy

            for mi in 1:Nx
                for ni in 1:Ny
                    b_idx = (mi - 1) * Ny + ni
                    Phi[pt, b_idx] = Xvals[i, mi] * Yvals[j, ni]
                    dPhi_x[pt, b_idx] = dXvals[i, mi] * Yvals[j, ni]
                    dPhi_y[pt, b_idx] = Xvals[i, mi] * dYvals[j, ni]
                end
            end
            pt += 1
        end
    end

    Drift = Vx_vec .* dPhi_x .+ Vy_vec .* dPhi_y
    return Phi' * Diagonal(W) * Drift
end

@testset "Mixed/Half zero-potential spectrum" begin
    bx = MixedSineBasis1D(8, 3.0)
    by = HalfCosineBasis1D(6, 2.0)
    basis = TensorProductBasis(bx, by)
    domain = RectangularDomain(0.0, 3.0, 0.0, 2.0)
    gradV_zero(_x, _y) = (0.0, 0.0)

    prob = SpectralGalerkinProblem(basis, domain, gradV_zero, 24)
    λs_dense, _ = solve_galerkin(prob; nev=6, solver=:dense)
    λs_krylov, _ = solve_galerkin(prob; nev=6, solver=:krylov)

    analytical = sort(vec([
        ((2 * m - 1) * π / (2 * bx.width))^2 + (n * π / by.width)^2
        for m in 1:bx.n_modes, n in 0:(by.n_modes - 1)
    ]))[1:6]

    @test isapprox(λs_dense, analytical; rtol=1e-12, atol=1e-12)
    @test isapprox(λs_krylov, analytical; rtol=1e-8, atol=1e-8)
end

@testset "Mixed/Half advection assembly matches reference" begin
    bx = MixedSineBasis1D(5, 3.0)
    by = HalfCosineBasis1D(4, 2.0)
    basis = TensorProductBasis(bx, by)
    domain = RectangularDomain(0.0, 3.0, 0.0, 2.0)
    gradV(x, y) = (1.0 + 0.25 * x - 0.1 * y, -0.2 + 0.3 * x + 0.15 * y)

    B_reference = assemble_advection_reference(basis, domain, gradV, 12)
    B_fast = SpectralGalerkin._assemble_advection(basis, domain, gradV, 12)

    @test isapprox(B_fast, B_reference; rtol=1e-12, atol=1e-12)
end
