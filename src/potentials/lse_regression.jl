module LSERegression

using ForwardDiff
using Optim
using Serialization

export LSEModel
export predict, gradient
export fit_lse_model, optimize_lse_model
export pack_parameters, unpack_parameters
export save_lse_model, load_lse_model

struct LSEModel{A,B}
    A::A
    b::B
    temperature::Float64
end

function _plane_value(model::LSEModel, i::Integer, x, y)
    return model.A[1, i] * x + model.A[2, i] * y + model.b[i]
end

function _raw_predict(model::LSEModel, x, y)
    inv_temperature = inv(model.temperature)
    best = maximum(_plane_value(model, i, x, y) * inv_temperature for i in eachindex(model.b))
    total = sum(exp(_plane_value(model, i, x, y) * inv_temperature - best) for i in eachindex(model.b))
    return model.temperature * (best + log(total))
end

function predict(model::LSEModel, x, y)
    return _raw_predict(model, x, y)
end

function gradient(model::LSEModel, x, y)
    inv_temperature = inv(model.temperature)
    best = maximum(_plane_value(model, i, x, y) * inv_temperature for i in eachindex(model.b))

    total = zero(best)
    gx = zero(best)
    gy = zero(best)
    for i in eachindex(model.b)
        weight = exp(_plane_value(model, i, x, y) * inv_temperature - best)
        total += weight
        gx += weight * model.A[1, i]
        gy += weight * model.A[2, i]
    end

    return (gx / total, gy / total)
end

function fit_lse_model(
    f;
    x_domain::Tuple{<:Real,<:Real}=(-1.0, 1.0),
    y_domain::Tuple{<:Real,<:Real}=(-1.0, 1.0),
    nx::Integer=25,
    ny::Integer=25,
    temperature::Real=1e-4,
)
    nx > 0 || throw(ArgumentError("nx must be positive."))
    ny > 0 || throw(ArgumentError("ny must be positive."))
    temperature > 0 || throw(ArgumentError("temperature must be positive."))

    xs = range(Float64(x_domain[1]), Float64(x_domain[2]), length=nx)
    ys = range(Float64(y_domain[1]), Float64(y_domain[2]), length=ny)
    n_planes = length(xs) * length(ys)

    A = Matrix{Float64}(undef, 2, n_planes)
    b = Vector{Float64}(undef, n_planes)

    idx = 1
    for y in ys, x in xs
        value = Float64(f(x, y))
        slope = ForwardDiff.gradient(z -> f(z[1], z[2]), [x, y])
        A[:, idx] .= slope
        b[idx] = value - slope[1] * x - slope[2] * y
        idx += 1
    end

    uncalibrated = LSEModel(A, b, Float64(temperature))
    offset = sum(_raw_predict(uncalibrated, x, y) - f(x, y) for y in ys, x in xs) / n_planes
    b .-= offset

    return LSEModel(A, b, Float64(temperature))
end

function pack_parameters(model::LSEModel)
    return vcat(vec(model.A), model.b)
end

function unpack_parameters(theta::AbstractVector, n_planes::Integer, temperature::Real)
    expected_length = 3 * n_planes
    length(theta) == expected_length ||
        throw(DimensionMismatch("expected $expected_length parameters, got $(length(theta))."))

    A = reshape(@view(theta[1:(2 * n_planes)]), 2, n_planes)
    b = @view(theta[(2 * n_planes + 1):expected_length])
    return LSEModel(A, b, Float64(temperature))
end

function optimize_lse_model(model::LSEModel, loss; maxiters::Integer=100, optimizer=LBFGS())
    n_planes = length(model.b)
    temperature = model.temperature
    theta0 = pack_parameters(model)
    objective(theta) = loss(unpack_parameters(theta, n_planes, temperature))
    gradient!(G, theta) = ForwardDiff.gradient!(G, objective, theta)

    result = Optim.optimize(
        objective,
        gradient!,
        theta0,
        optimizer,
        Optim.Options(iterations=maxiters),
    )

    theta = Optim.minimizer(result)
    return (
        model=unpack_parameters(theta, n_planes, temperature),
        result=result,
    )
end

function save_lse_model(path::AbstractString, model::LSEModel)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, (A=Matrix(model.A), b=Vector(model.b), temperature=model.temperature))
    end
    return path
end

function load_lse_model(path::AbstractString)
    payload = open(deserialize, path)
    return LSEModel(payload.A, payload.b, payload.temperature)
end

end # module
