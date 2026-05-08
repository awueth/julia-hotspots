include("../src/solver.jl")
include("../src/eigenfunction_io.jl")
include("../src/lse_regression.jl")
include("../src/potential_interface.jl")

using LinearAlgebra

using .PotentialInterface

const DEFAULT_EPSILON = 0.1
const DEFAULT_WING_LENGTH = 5.0
const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_LSE_WING_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_FITTED_EIGENFUNCTION_PATH = joinpath(@__DIR__, "..", "checkpoints", "fitted_eigenfunction.chk")
const DEFAULT_WING_SCALE = 5e6
const DEFAULT_OPTIMIZED_LSE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "optimized_joined_lse_potential.chk")

function build_joined_lse_potential()
    core = load_lse_core_potential(checkpoint_path=DEFAULT_LSE_CORE_CHECKPOINT_PATH)
    domain = potential_domain(core)
    wing = load_lse_wing_potential(
        checkpoint_path=DEFAULT_LSE_WING_CHECKPOINT_PATH;
        Lx=domain.Lx,
        Ly=domain.Ly,
        scale=DEFAULT_WING_SCALE,
    )
    return join_lse_potentials(core, wing)
end

function collocation_points(diam_x::Real, diam_y::Real, n_points::Tuple{Int,Int})
    n_x, n_y = n_points
    x_grid = collect(range(0.0, 0.5 * Float64(diam_x), length=n_x))
    push!(x_grid, 0.5 * pi)
    y_grid = collect(range(0.0, 0.5 * Float64(diam_y), length=n_y))
    return vec(collect(Iterators.product(x_grid, y_grid)))
end

function sampled_collocation_points(
    diam_x::Real,
    diam_y::Real,
    x_center::Real;
    interface_x_radius::Real=0.10,
    interface_nx::Integer=24,
    interface_ny::Integer=32,
    stability_nx::Integer=8,
    stability_ny::Integer=6,
)
    x_min = 0.0
    x_max = 0.5 * Float64(diam_x)
    y_min = 0.0
    y_max = 0.5 * Float64(diam_y)

    interface_xs = range(
        max(x_min, Float64(x_center) - Float64(interface_x_radius)),
        min(x_max, Float64(x_center) + Float64(interface_x_radius));
        length=interface_nx,
    )
    interface_ys = range(y_min, y_max; length=interface_ny)
    stability_xs = range(x_min, x_max; length=stability_nx)
    stability_ys = range(y_min, y_max; length=stability_ny)

    points = Tuple{Float64,Float64}[]
    append!(points, vec(collect(Iterators.product(interface_xs, interface_ys))))
    append!(points, vec(collect(Iterators.product(stability_xs, stability_ys))))

    return sort(unique(points))
end

function residual_vector_for_model(
    model,
    points,
    d::Real,
    epsilon::Real,
    λx::AbstractVector,
    λy::AbstractVector,
    λr::AbstractVector,
    coefficients::AbstractVector,
)
    n_points = length(points)
    n_modes = length(λx)
    T = promote_type(eltype(model.A), eltype(model.b), Float64)
    residuals = Vector{T}(undef, n_points)

    for point_index in 1:n_points
        x, y = points[point_index]
        gx, gy = gradient(model, x, y)
        gx *= epsilon
        gy *= epsilon

        nr = 4.0
        inv_len = inv(sqrt(gx^2 + gy^2 + nr^2))
        nx = gx * inv_len
        ny = gy * inv_len
        nr_scaled = nr * inv_len

        residual = zero(T)
        for mode_index in 1:n_modes
            lx = λx[mode_index]
            ly = λy[mode_index]
            lr = λr[mode_index]
            av, (agx, agy) = axial_basis(lx, ly, x, y)
            rv, rgrad = ϕ(Float64(d), lr, 1.0)

            residual += coefficients[mode_index] * (nx * agx * rv + ny * agy * rv + nr_scaled * av * rgrad)
        end

        residuals[point_index] = residual
    end

    return residuals
end

function residual_stats(residual::AbstractVector)
    return (
        l2=norm(residual),
        rms=norm(residual) / sqrt(length(residual)),
        max=maximum(abs, residual),
    )
end

function active_plane_indices_near_x(
    model,
    x_center::Real,
    y_max::Real;
    x_radius::Real=0.05,
    nx::Integer=3,
    ny::Integer=64,
    top_k::Integer=1,
)
    xs = range(Float64(x_center) - Float64(x_radius), Float64(x_center) + Float64(x_radius), length=nx)
    ys = range(0.0, Float64(y_max), length=ny)
    indices = Int[]

    for x in xs, y in ys
        values = vec(model.A[1, :] .* x .+ model.A[2, :] .* y .+ model.b)
        if top_k == 1
            push!(indices, last(findmax(values)))
        else
            append!(indices, partialsortperm(values, 1:top_k; rev=true))
        end
    end

    return sort(unique(indices))
end

function pack_selected_lse_parameters(model, plane_indices::AbstractVector{<:Integer})
    return vcat(vec(model.A[:, plane_indices]), model.b[plane_indices])
end

function model_with_selected_lse_parameters(model, plane_indices::AbstractVector{<:Integer}, theta::AbstractVector)
    n_planes = length(plane_indices)
    expected_length = 3 * n_planes
    length(theta) == expected_length ||
        throw(DimensionMismatch("expected $expected_length selected parameters, got $(length(theta))."))

    A = Matrix{eltype(theta)}(model.A)
    b = Vector{eltype(theta)}(model.b)
    A[:, plane_indices] .= reshape(@view(theta[1:(2 * n_planes)]), 2, n_planes)
    b[plane_indices] .= @view(theta[(2 * n_planes + 1):expected_length])

    return LSEModel(A, b, model.temperature)
end

function optimize_selected_lse_planes(model, plane_indices, loss; maxiters::Integer=3)
    theta0 = pack_selected_lse_parameters(model, plane_indices)
    objective(theta) = loss(model_with_selected_lse_parameters(model, plane_indices, theta))
    gradient!(G, theta) = ForwardDiff.gradient!(G, objective, theta)

    result = Optim.optimize(
        objective,
        gradient!,
        theta0,
        LBFGS(),
        Optim.Options(iterations=maxiters),
    )

    return (
        model=model_with_selected_lse_parameters(model, plane_indices, Optim.minimizer(result)),
        result=result,
    )
end

function main()
    epsilon = DEFAULT_EPSILON
    fit = load_fitted_eigenfunction(DEFAULT_FITTED_EIGENFUNCTION_PATH)
    pot = build_joined_lse_potential()
    domain = potential_domain(pot)

    diam_x = 2.0 * (domain.Lx + DEFAULT_WING_LENGTH)
    diam_y = 2.0 * domain.Ly
    interface_x_radius = parse(Float64, get(ENV, "LSE_SAMPLE_INTERFACE_X_RADIUS", "0.10"))
    interface_nx = parse(Int, get(ENV, "LSE_SAMPLE_INTERFACE_NX", "24"))
    interface_ny = parse(Int, get(ENV, "LSE_SAMPLE_INTERFACE_NY", "32"))
    stability_nx = parse(Int, get(ENV, "LSE_SAMPLE_STABILITY_NX", "8"))
    stability_ny = parse(Int, get(ENV, "LSE_SAMPLE_STABILITY_NY", "6"))
    points = sampled_collocation_points(
        diam_x,
        diam_y,
        domain.Lx;
        interface_x_radius=interface_x_radius,
        interface_nx=interface_nx,
        interface_ny=interface_ny,
        stability_nx=stability_nx,
        stability_ny=stability_ny,
    )
    λx, λy, λr = get_eigenvalues(diam_x, diam_y, fit.n_modes, fit.λ)

    active_x_radius = parse(Float64, get(ENV, "LSE_ACTIVE_X_RADIUS", "0.05"))
    active_nx = parse(Int, get(ENV, "LSE_ACTIVE_NX", "3"))
    active_ny = parse(Int, get(ENV, "LSE_ACTIVE_NY", "64"))
    active_top_k = parse(Int, get(ENV, "LSE_ACTIVE_TOPK", "1"))
    plane_indices = active_plane_indices_near_x(
        pot.model,
        domain.Lx,
        domain.Ly;
        x_radius=active_x_radius,
        nx=active_nx,
        ny=active_ny,
        top_k=active_top_k,
    )

    println("Optimizing ", length(plane_indices), " / ", length(pot.model.b), " LSE planes active near x = pi / 2.")
    println("Evaluating residual on ", length(points), " sampled collocation points and ", length(λx), " modes.")
    residual_before = residual_vector_for_model(pot.model, points, fit.d, epsilon, λx, λy, λr, fit.coefficients)
    before = residual_stats(residual_before)

    println("Residual before optimization:")
    println("  l2  = ", before.l2)
    println("  rms = ", before.rms)
    println("  max = ", before.max)

    function residual_loss(model)
        residual = residual_vector_for_model(model, points, fit.d, epsilon, λx, λy, λr, fit.coefficients)
        return sum(abs2, residual) / length(residual)
    end

    maxiters = parse(Int, get(ENV, "LSE_RESIDUAL_MAXITERS", "3"))
    optimized = optimize_selected_lse_planes(pot.model, plane_indices, residual_loss; maxiters=maxiters)
    optimized_pot = JoinedLSEPotential(optimized.model; Lx=domain.Lx, Ly=domain.Ly)

    residual_after = residual_vector_for_model(optimized_pot.model, points, fit.d, epsilon, λx, λy, λr, fit.coefficients)
    after = residual_stats(residual_after)

    println("Optimization converged: ", Optim.converged(optimized.result))
    println("Residual after optimization:")
    println("  l2  = ", after.l2)
    println("  rms = ", after.rms)
    println("  max = ", after.max)

    if get(ENV, "SAVE_OPTIMIZED_LSE_POTENTIAL", "0") == "1"
        save_lse_model(DEFAULT_OPTIMIZED_LSE_CHECKPOINT_PATH, optimized_pot.model)
        println("Saved optimized joined LSE potential to ", DEFAULT_OPTIMIZED_LSE_CHECKPOINT_PATH)
    end

    return (before=before, after=after, result=optimized.result)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
