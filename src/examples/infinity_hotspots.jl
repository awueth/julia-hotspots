using Plots
using Printf
using Revise

includet("../limit_solvers/spectral_galerkin.jl")
includet("../potential_interface.jl")

using .PotentialInterface
using .SpectralGalerkin

begin
    const DEFAULT_EPSILON = 0.1
    const DEFAULT_WING_LENGTH = 1.5*pi
    const DEFAULT_MX = 128
    const DEFAULT_NY = 32
    const DEFAULT_N_GRID = 256
    const DEFAULT_NEV = 1
    const DEFAULT_SOLVER = :krylov
    const DEFAULT_T_FINAL = 1.0
    const DEFAULT_YS_PROFILE_COUNT = 200
    const DEFAULT_TS_TRACE_COUNT = 401
    const DEFAULT_NX_PLOT = 32
    const DEFAULT_NY_PLOT = 32
    const DEFAULT_OUTPUT_FILE = "eigenfunction_mixed_extended.png"
    const DEFAULT_WING_POTENTIAL = :nonconvex
    const DEFAULT_SMOOTH_MAX_STRENGTH = 10.0
    const DEFAULT_WING_SCALE = 5e6
    const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")
end

struct InfinityHotspotsContext{P,D,Pr,Bx,By}
    pot::P
    data::D
    prob::Pr
    basis_x::Bx
    basis_y::By
    epsilon::Float64
    diam_x::Float64
end

struct InfinityHotspotsSolution
    λ::Float64
    coeffs::Vector{Float64}
end

struct InfinityHotspotAnalysis
    x_middle::Float64
    t_final::Float64
    ys_profile::Vector{Float64}
    ts_trace::Vector{Float64}
    u_inner_t0::Vector{Float64}
    u_middle_t0::Vector{Float64}
    u_outer_t0::Vector{Float64}
    u_middle_t1::Vector{Float64}
    u_outer_t1::Vector{Float64}
    boundary_trace::Vector{Float64}
    interior_trace::Vector{Float64}
    boundary_idx::Int
    interior_idx::Int
    boundary_max::Float64
    interior_max::Float64
end

"""
    select_potential(; wing=:handmade, smooth_max_strength=10.0)

Build the potential used by this example from the saved LSE core model.
`wing` can be `:handmade`, `:non_convex`, or an `AbstractWingPotential`.
"""
function select_potential(;
    wing=DEFAULT_WING_POTENTIAL,
    smooth_max_strength::Real=DEFAULT_SMOOTH_MAX_STRENGTH,
    wing_scale::Real=DEFAULT_WING_SCALE,
    lse_core_checkpoint_path::AbstractString=DEFAULT_LSE_CORE_CHECKPOINT_PATH,
)
    @printf("Loaded LSE core checkpoint: %s\n", lse_core_checkpoint_path)
    core_potential = load_lse_core_potential(checkpoint_path=lse_core_checkpoint_path)

    wing_potential = if wing == :handmade
        domain = potential_domain(core_potential)
        HandmadeWingPotential(domain.Lx; scale=wing_scale)
    elseif wing == :non_convex || wing == :nonconvex
        domain = potential_domain(core_potential)
        NonConvexWingPotential(domain.Lx; anchor=core_value(core_potential, domain.Lx, domain.Ly), scale=wing_scale)
    elseif wing isa AbstractWingPotential
        wing
    else
        throw(ArgumentError("Unknown wing potential selector `$wing`. Use `:handmade`, `:non_convex`, or pass an AbstractWingPotential."))
    end

    return SmoothMaxPotential(core_potential, wing_potential; smooth_max_strength=smooth_max_strength)
end

"""
    build_context(pot; epsilon=0.1, wing_length=5.0, Mx=128, Ny=32, N_grid=256)

Generate the current potential, build the mixed quarter-domain Galerkin problem,
and return reusable solve context for interactive experiments.
"""
function build_context(pot::AbstractPotential;
    epsilon::Float64=DEFAULT_EPSILON,
    wing_length::Float64=DEFAULT_WING_LENGTH,
    Mx::Int=DEFAULT_MX,
    Ny::Int=DEFAULT_NY,
    N_grid::Int=DEFAULT_N_GRID)

    data = potential_domain(pot)

    diam_x = data.Lx + wing_length
    basis_x = MixedSineBasis1D(Mx, diam_x)
    basis_y = HalfCosineBasis1D(Ny, data.Ly)
    basis = TensorProductBasis(basis_x, basis_y)
    domain = RectangularDomain(0.0, diam_x, 0.0, data.Ly)

    _, gradV_scaled = potential_functions(pot; scale=epsilon)

    prob = SpectralGalerkinProblem(basis, domain, gradV_scaled, N_grid)

    println("Built context: epsilon = ", epsilon,
        ", wing_length = ", wing_length,
        ", diam_x = ", diam_x)

    return InfinityHotspotsContext(pot, data, prob, basis_x, basis_y, epsilon, diam_x)
end

"""
    solve_problem(ctx; nev=1, solver=:krylov)

Solve the mixed-basis eigenvalue problem for a previously built context.
"""
function solve_problem(ctx::InfinityHotspotsContext;
    nev::Int=DEFAULT_NEV,
    solver::Symbol=DEFAULT_SOLVER)

    λs, vecs = solve_galerkin(ctx.prob; nev=nev, solver=solver)
    λ = Float64(λs[1])
    coeffs = vec(Float64.(vecs[:, 1]))
    coeffs .*= sign(coeffs[1])

    @printf("Eigenvalue λ₁ = %.6f\n", λ)

    # _, gradV_scaled = potential_functions(ctx.pot; scale=ctx.epsilon)

    # x_grid, y_grid, residual = compute_residual(ctx.prob, gradV_scaled, coeffs, λ)

    # display(heatmap(x_grid, y_grid, abs.(residual)))

    # @printf("Maximal error: %.2e\n", maximum(abs, residual))

    return InfinityHotspotsSolution(λ, coeffs)
end

"""
    evaluate_hotspot(ctx, sol; x_middle=ctx.data.Lx + 2.5, t_final=1.0, ys_profile_count=200, ts_trace_count=401)

Evaluate the same profile and hotspot diagnostics used in the higher-order
mixed example,  reusing the already solved eigenpair.
"""
function evaluate_hotspot(ctx::InfinityHotspotsContext, sol::InfinityHotspotsSolution;
    x_search_count::Int=100,
    t_final::Float64=DEFAULT_T_FINAL,
    ys_profile_count::Int=DEFAULT_YS_PROFILE_COUNT,
    ts_trace_count::Int=DEFAULT_TS_TRACE_COUNT)

    ys_profile = collect(range(-ctx.data.Ly, ctx.data.Ly, length=ys_profile_count))
    ts_trace = collect(range(0.0, t_final, length=ts_trace_count))
    xs_scan = collect(range(ctx.data.Lx, ctx.diam_x, length=x_search_count))

    # 1. Local closure to save repetition
    eval_u(x, y, t) = SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, x, y, t)

    # 2. Directly use the final variable names
    x_middle = ctx.data.Lx
    interior_max = -Inf
    interior_trace = Float64[]
    interior_idx = 1

    for x in xs_scan
        # 3. Use broadcasting
        trace = eval_u.(x, 0.0, ts_trace)
        
        # 4. Use `findmax` to get both value and index at once
        val, idx = findmax(trace)
        if val > interior_max
            interior_max = val
            interior_trace = trace
            interior_idx = idx
            x_middle = x
        end
    end

    # Replace list comprehensions with broadcasting
    u_inner_t0  = eval_u.(ctx.data.Lx, ys_profile, 0.0)
    u_middle_t0 = eval_u.(x_middle, ys_profile, 0.0)
    u_outer_t0  = eval_u.(ctx.diam_x, ys_profile, 0.0)

    u_middle_t1 = eval_u.(x_middle, ys_profile, t_final)
    u_outer_t1  = eval_u.(ctx.diam_x, ys_profile, t_final)

    boundary_trace = eval_u.(ctx.diam_x, 0.0, ts_trace)
    boundary_max, boundary_idx = findmax(boundary_trace)

    println("\nTrace maxima on t ∈ [0, $(t_final)]:")
    @printf("  boundary point  (x = %.6f, y = 0.0): max u = %.8f at t = %.6f\n",
        ctx.diam_x, boundary_max, ts_trace[boundary_idx])
    @printf("  interior point  (x = %.6f, y = 0.0): max u = %.8f at t = %.6f\n",
        x_middle, interior_max, ts_trace[interior_idx])
        
    if boundary_max >= interior_max
        @printf("  overall maximum occurs at the boundary point (x = %.6f, y = 0.0, t = %.6f)\n",
            ctx.diam_x, ts_trace[boundary_idx])
    else
        @printf("  overall maximum occurs at the interior point (x = %.6f, y = 0.0, t = %.6f)\n",
            x_middle, ts_trace[interior_idx])
    end
    @printf("Hotspot ratio (interior max / boundary max) = %.9f\n", interior_max / boundary_max)

    return InfinityHotspotAnalysis(
        x_middle, t_final, ys_profile, ts_trace,
        u_inner_t0, u_middle_t0, u_outer_t0,
        u_middle_t1, u_outer_t1,
        boundary_trace, interior_trace,
        boundary_idx, interior_idx,
        boundary_max, interior_max
    )
end

"""
    plot_results(ctx, sol; hotspot=nothing, nx_plot=100, ny_plot=50, output_file="eigenfunction_mixed_extended.png")

Plot the full-domain eigenfunction, the wing potential, and the hotspot profile
panels. If a precomputed hotspot analysis is supplied, it is reused directly.
"""
function plot_results(ctx::InfinityHotspotsContext, sol::InfinityHotspotsSolution;
    hotspot::Union{Nothing,InfinityHotspotAnalysis}=nothing,
    nx_plot::Int=DEFAULT_NX_PLOT,
    ny_plot::Int=DEFAULT_NY_PLOT,
    output_file::String=DEFAULT_OUTPUT_FILE)

    hotspot === nothing && (hotspot = evaluate_hotspot(ctx, sol))
    inner_label = @sprintf("x = %.3f", ctx.data.Lx)
    middle_label = @sprintf("x = %.3f", hotspot.x_middle)
    outer_label = @sprintf("x = %.3f", ctx.diam_x)

    x_grid, y_grid, u_grid = SpectralGalerkin.reconstruct_full_field(
        ctx.prob,
        sol.coeffs;
        nx=nx_plot,
        ny=ny_plot,
    )

    x_grid_pot = range(ctx.data.Lx-0.5, ctx.diam_x, length=nx_plot)
    y_grid_pot = range(-ctx.data.Ly, ctx.data.Ly, length=ny_plot)
    V_grid = [potential_value(ctx.pot, x, y) for x in x_grid_pot, y in y_grid_pot]

    p1 = surface(x_grid, y_grid, u_grid';
        xlabel="x", ylabel="y", zlabel="u(x,y)",
        title="First Eigenfunction of L",
        colorbar=true, camera=(35, 30))

    z_vals = [
        SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, hotspot.x_middle, y, 0.0)
        for y in hotspot.ys_profile
    ]
    plot!(p1,
        fill(hotspot.x_middle, length(hotspot.ys_profile)),
        hotspot.ys_profile,
        z_vals,
        color=:red,
        linewidth=2,
        label=middle_label)

    p_pot = wireframe(x_grid_pot, y_grid_pot, V_grid';
        xlabel="x", ylabel="y", zlabel="V(x,y)",
        title="Potential in the wing region",
        camera=(-20, 30))

    p2 = plot(hotspot.ys_profile, hotspot.u_inner_t0;
        xlabel="y", ylabel="u",
        title="Profiles at t = 0",
        linewidth=2, color=:blue, label=inner_label,
        legend=:topright)
    plot!(p2, hotspot.ys_profile, hotspot.u_middle_t0;
        linewidth=2, linestyle=:dot, color=:red, label=middle_label)
    plot!(p2, hotspot.ys_profile, hotspot.u_outer_t0;
        linewidth=2, linestyle=:dash, color=:black, label=outer_label)

    p3 = plot(hotspot.ys_profile, hotspot.u_middle_t1;
        xlabel="y", ylabel="u",
        title="Profiles at t = $(hotspot.t_final)",
        linewidth=2, linestyle=:dot, color=:red, label=middle_label,
        legend=:topright)
    plot!(p3, hotspot.ys_profile, hotspot.u_outer_t1;
        linewidth=2, linestyle=:dash, color=:black, label=outer_label)

    fig = plot(p1, p_pot, p2, p3;
        layout=@layout([a b; c d]),
        size=(1200, 800))

    display(fig)
    savefig(fig, output_file)
    println("Plot saved to ", output_file)

    return fig
end

pot = select_potential(
    wing=DEFAULT_WING_POTENTIAL,
    smooth_max_strength=DEFAULT_SMOOTH_MAX_STRENGTH,
    wing_scale=DEFAULT_WING_SCALE,
);

ctx = build_context(pot;
    epsilon=DEFAULT_EPSILON,
    wing_length=DEFAULT_WING_LENGTH,
    Mx=DEFAULT_MX,
    Ny=DEFAULT_NY,
    N_grid=DEFAULT_N_GRID
);

sol = solve_problem(ctx; nev=DEFAULT_NEV, solver=DEFAULT_SOLVER);

hotspot = evaluate_hotspot(ctx, sol;
    x_search_count=100,
    t_final=DEFAULT_T_FINAL,
    ys_profile_count=DEFAULT_YS_PROFILE_COUNT,
    ts_trace_count=DEFAULT_TS_TRACE_COUNT,
);
fig = plot_results(ctx, sol;
    hotspot=hotspot,
    nx_plot=DEFAULT_NX_PLOT,
    ny_plot=DEFAULT_NY_PLOT,
    output_file=DEFAULT_OUTPUT_FILE,
);
