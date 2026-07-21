using Test

include("../src/solver/solver.jl")

@testset "Fibonacci boundary sampling" begin
    V(_x, _y) = 0.0
    gradV(x, y) = (x, y)

    d = Inf
    diam_x = 4.0
    diam_y = 2.0
    n_points = 15
    geometry = make_geometry(d, diam_x, diam_y, V, gradV, FibonacciSampler(n_points))

    @test length(geometry.points.x) == n_points
    @test length(geometry.points.y) == n_points
    @test geometry.points.r === nothing   # d = Inf has no radial samples
    @test length(geometry.normals.x) == n_points
    @test length(geometry.normals.y) == n_points
    @test length(geometry.normals.r) == n_points

    @test all(0.0 .<= geometry.points.x .<= 0.5 * diam_x)
    @test all(0.0 .<= geometry.points.y .<= 0.5 * diam_y)
    @test geometry.points.x == [((k + 0.5) / n_points) * (0.5 * diam_x) for k in 0:(n_points - 1)]

    n_modes = (2, 2)
    A = get_matrix(geometry, n_modes, 5.0)
    @test size(A) == (n_points, prod(n_modes))

    weights = ones(n_points)
    @test get_matrix(geometry, n_modes, 5.0; weights=weights) == A
    @test get_matrix(geometry, n_modes, 5.0; weights=reshape(weights, 3, :)) == A
    @test_throws DimensionMismatch get_matrix(geometry, n_modes, 5.0; weights=ones(n_points - 1))
end

@testset "Invalid boundary sampling count" begin
    V(_x, _y) = 0.0
    gradV(_x, _y) = (0.0, 0.0)

    @test_throws ArgumentError make_geometry(Inf, 4.0, 2.0, V, gradV, FibonacciSampler(0))
end
