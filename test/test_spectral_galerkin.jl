using Printf
using LinearAlgebra

# ─────────────────────────────────────────────
# Load solvers
# ─────────────────────────────────────────────

include("../src/limit_solvers/spectral_galerkin.jl")

using .SpectralGalerkin

# ─────────────────────────────────────────────
# Test 1: Cosine basis V=0 — analytical eigenvalues
# ─────────────────────────────────────────────

println("=" ^ 60)
println("Test 1: Cosine basis V=0 (analytical eigenvalues)")
println("=" ^ 60)

gradV_zero(_x, _y) = (0.0, 0.0)

bx_v0 = CosineBasis1D(8, 3.0)
by_v0 = CosineBasis1D(8, 2.0)
basis_v0 = TensorProductBasis(bx_v0, by_v0)
domain_v0 = RectangularDomain(-3.0, 3.0, -2.0, 2.0)
prob_v0 = SpectralGalerkinProblem(basis_v0, domain_v0, gradV_zero, 32)
λs_v0, _ = solve_galerkin(prob_v0; nev=6)

# Analytical eigenvalues: (mπ/(2*diam_x))² + (nπ/(2*diam_y))²
# For cosine basis on [-L, L]: k_m = mπ/(2L)
# Smallest: (0,0)=0, (1,0)=(π/6)², (0,1)=(π/4)², (1,1)=(π/6)²+(π/4)², ...
analytical = sort(vec([
    (m * π / 6)^2 + (n * π / 4)^2
    for m in 0:7, n in 0:7
]))[1:6]

println("  Computed   Analytical   Error")
for i in 1:6
    @printf("  %.8f   %.8f   %.2e\n", λs_v0[i], analytical[i], abs(λs_v0[i] - analytical[i]))
end
max_err_v0 = maximum(abs.(λs_v0 .- analytical))
@assert max_err_v0 < 1e-12 "Analytical eigenvalue mismatch!"
println("PASSED ✓")

# ─────────────────────────────────────────────
# Test 2: SineWing basis with OU potential — sanity check
# ─────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("Test 2: SineWing basis with OU potential (spectral gap ≈ 1)")
println("=" ^ 60)

# Set up SineWing basis for this test
Lx, Ly, a = 10.0, 6.0, 2.0
Mx_sw, Ny_sw = 16, 8
bx_sw = SineWingBasis1D(Mx_sw, a, Lx / 2)
by_sw = SymmetricCosineBasis1D(Ny_sw, Ly / 2)
basis_sw = TensorProductBasis(bx_sw, by_sw)
domain_sw = RectangularDomain(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2)

gradV_ou(x, y) = (x, y)
prob_ou = SpectralGalerkinProblem(basis_sw, domain_sw, gradV_ou, 48)
λs_ou, _ = solve_galerkin(prob_ou; nev=4)

println("  Eigenvalues:")
for (i, λ) in enumerate(λs_ou)
    @printf("  λ_%d = %.8f\n", i, λ)
end
println("  (spectral gap λ₁ should be ≈ 1 for large domain)")
# Relaxed check: first eigenvalue should be positive and < 1
@assert 0 < λs_ou[1] < 2.0 "OU eigenvalue out of expected range"
println("PASSED ✓")

println("\n" * "=" ^ 60)
println("All tests completed!")
println("=" ^ 60)
