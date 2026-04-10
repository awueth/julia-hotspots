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
points = Iterators.product(x_grid, y_grid)

@testset "Symmetry" begin
    @test all(points) do (x, y)
        V_extended(pot, x, y) == V_extended(pot, -x, y) &&
        V_extended(pot, x, -y) == V_extended(pot, x, -y)
    end
end


@testset verbose = true "Convexity" begin
    @test all(points) do (x, y)
        H = ForwardDiff.hessian((p) -> V_extended(pot, p[1], p[2]), [x, y])
        eigmin(Symmetric(H)) >= 0
    end
end

for (x, y) in points
    H = ForwardDiff.hessian((p) -> V_extended(pot, p[1], p[2]), [x, y])
    λ = eigmin(Symmetric(H))

    if λ < 0
        print("Negative eigenvalue at (x, y) = ($x, $y): $λ\n")
    end
end