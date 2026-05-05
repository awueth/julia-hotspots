include("../src/lse_regression.jl")

using Test
using Statistics

@testset "LSE model" begin
    f(x, y) = x^2 + 0.5y^2 + 0.2x - 0.1y
    grad_f(x, y) = (2x + 0.2, y - 0.1)

    model = fit_lse_model(f; x_domain=(-1.0, 1.0), y_domain=(-0.8, 0.8), nx=9, ny=9, temperature=1e-3)

    @test model isa LSEModel
    @test size(model.A) == (2, length(model.b))
    @test isfinite(predict(model, 0.2, -0.3))
    @test predict(model, 0.0, 0.0) ≈ f(0.0, 0.0) atol=2e-2
    @test predict(model, 0.5, -0.4) ≈ f(0.5, -0.4) atol=2e-2

    gx, gy = gradient(model, 0.5, -0.4)
    expected_gx, expected_gy = grad_f(0.5, -0.4)
    @test isfinite(gx)
    @test isfinite(gy)
    @test gx ≈ expected_gx atol=2e-2
    @test gy ≈ expected_gy atol=2e-2

    for (p, q, t) in (
        ((-0.8, -0.3), (0.7, 0.4), 0.25),
        ((-0.2, 0.6), (0.9, -0.5), 0.6),
    )
        x = t * p[1] + (1 - t) * q[1]
        y = t * p[2] + (1 - t) * q[2]
        lhs = predict(model, x, y)
        rhs = t * predict(model, p[1], p[2]) + (1 - t) * predict(model, q[1], q[2])
        @test lhs <= rhs + 1e-10
    end

    theta = pack_parameters(model)
    unpacked = unpack_parameters(theta, length(model.b), model.temperature)
    @test predict(unpacked, 0.25, -0.1) ≈ predict(model, 0.25, -0.1)

    xs = (-0.5, 0.0, 0.5)
    ys = (-0.4, 0.2)
    loss(candidate) = mean((predict(candidate, x, y) - 0.5f(x, y))^2 for x in xs, y in ys)
    initial_loss = loss(model)
    fit = optimize_lse_model(model, loss; maxiters=20)

    final_loss = loss(fit.model)
    @test final_loss < initial_loss

    mktempdir() do dir
        path = joinpath(dir, "lse_model.chk")
        @test save_lse_model(path, model) == path

        loaded = load_lse_model(path)
        @test loaded isa LSEModel
        @test loaded.A isa Matrix
        @test loaded.b isa Vector
        @test predict(loaded, 0.25, -0.1) ≈ predict(model, 0.25, -0.1)
        @test gradient(loaded, 0.25, -0.1) == gradient(model, 0.25, -0.1)
    end
end
