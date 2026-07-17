include("../src/solver/eigenfunction_hot_spot.jl")

using Test
using .EigenfunctionHotSpot
using .EigenfunctionLinfNorm

@testset "Sampled hot-spot difference" begin
    fit = FittedEigenfunction([1.0], 4.0, (1, 1), Inf, 2.0, 2.0)
    grid_size = (9, 7, 5)
    result = sampled_hot_spot_difference(fit, grid_size)

    xs = range(0.0, fit.diam_x / 2; length=grid_size[1])
    ys = range(0.0, fit.diam_y / 2; length=grid_size[2])
    rs = range(0.0, 1.0; length=grid_size[3])
    expected_interior = sampled_linf_norm(fit, (x=xs[1:(end - 1)], y=[0.0], r=[0.0]))
    expected_faces = (
        sampled_linf_norm(fit, (x=xs, y=ys, r=[1.0])),
        sampled_linf_norm(fit, (x=[last(xs)], y=ys, r=rs)),
        sampled_linf_norm(fit, (x=xs, y=[last(ys)], r=rs)),
    )

    @test result.interior == expected_interior
    @test getproperty.(result.boundary_faces, :name) == ("outer_shell", "x_side", "y_side")
    @test getproperty.(result.boundary_faces, :value) == getproperty.(expected_faces, :value)
    @test result.boundary.value == maximum(face.value for face in expected_faces)
    @test result.effect == result.interior.value - result.boundary.value
    @test result.interior.location.x < fit.diam_x / 2

    finite_fit = FittedEigenfunction([1.0], 4.0, (1, 1), 4.0, 2.0, 2.0)
    negative_fit = FittedEigenfunction([-1.0], 4.0, (1, 1), Inf, 2.0, 2.0)
    @test_throws ArgumentError sampled_hot_spot_difference(finite_fit, grid_size)
    @test_throws ArgumentError sampled_hot_spot_difference(negative_fit, grid_size)
end
