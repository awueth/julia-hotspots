using Plots
using Printf
using Revise

includet("../limit_solvers/spectral_galerkin.jl")
includet("../potential_generator.jl")

using .PotentialGenerator
using .SpectralGalerkin

begin
    const DEFAULT_EPSILON = 0.1
    const DEFAULT_EXT_FACTOR = 4.0
    const DEFAULT_MX = 128
    const DEFAULT_NY = 32
    const DEFAULT_N_GRID = 256
    const DEFAULT_NEV = 1
    const DEFAULT_SOLVER = :krylov
    const DEFAULT_X_MIDDLE_OFFSET = 2.5
    const DEFAULT_T_FINAL = 1.0
    const DEFAULT_YS_PROFILE_COUNT = 200
    const DEFAULT_TS_TRACE_COUNT = 401
    const DEFAULT_NX_PLOT = 100
    const DEFAULT_NY_PLOT = 50
    const DEFAULT_OUTPUT_FILE = "eigenfunction_mixed_extended.png"
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
    build_context(; epsilon=0.1, ext_factor=4.0, Mx=128, Ny=32, N_grid=256, potential_kwargs...)

Generate the current potential, build the mixed quarter-domain Galerkin problem,
and return reusable solve context for interactive experiments.
"""
function build_context(pot;
    epsilon::Float64=DEFAULT_EPSILON,
    ext_factor::Float64=DEFAULT_EXT_FACTOR,
    Mx::Int=DEFAULT_MX,
    Ny::Int=DEFAULT_NY,
    N_grid::Int=DEFAULT_N_GRID)

    data = pot.data

    diam_x = ext_factor * data.Lx
    basis_x = MixedSineBasis1D(Mx, diam_x)
    basis_y = HalfCosineBasis1D(Ny, data.Ly)
    basis = TensorProductBasis(basis_x, basis_y)
    domain = RectangularDomain(0.0, diam_x, 0.0, data.Ly)

    gradV_scaled = (x, y) -> begin
        g = epsilon .* ∇V_extended(pot, x, y)
        return (g[1], g[2])
    end

    prob = SpectralGalerkinProblem(basis, domain, gradV_scaled, N_grid)

    println("Built context: epsilon = ", epsilon,
        ", ext_factor = ", ext_factor,
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
    return InfinityHotspotsSolution(λ, coeffs)
end

"""
    evaluate_hotspot(ctx, sol; x_middle=ctx.data.Lx + 2.5, t_final=1.0, ys_profile_count=200, ts_trace_count=401)

Evaluate the same profile and hotspot diagnostics used in the higher-order
mixed example,  reusing the already solved eigenpair.
"""
function evaluate_hotspot(ctx::InfinityHotspotsContext, sol::InfinityHotspotsSolution;
    x_middle::Float64=ctx.data.Lx + DEFAULT_X_MIDDLE_OFFSET,
    t_final::Float64=DEFAULT_T_FINAL,
    ys_profile_count::Int=DEFAULT_YS_PROFILE_COUNT,
    ts_trace_count::Int=DEFAULT_TS_TRACE_COUNT)

    ys_profile = collect(range(-ctx.data.Ly, ctx.data.Ly, length=ys_profile_count))
    ts_trace = collect(range(0.0, t_final, length=ts_trace_count))

    u_inner_t0 = [
        SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, ctx.data.Lx, y, 0.0)
        for y in ys_profile
    ]
    u_middle_t0 = [
        SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, x_middle, y, 0.0)
        for y in ys_profile
    ]
    u_outer_t0 = [
        SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, ctx.diam_x, y, 0.0)
        for y in ys_profile
    ]

    u_middle_t1 = [
        SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, x_middle, y, t_final)
        for y in ys_profile
    ]
    u_outer_t1 = [
        SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, ctx.diam_x, y, t_final)
        for y in ys_profile
    ]

    boundary_trace = [
        SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, ctx.diam_x, 0.0, t)
        for t in ts_trace
    ]
    interior_trace = [
        SpectralGalerkin.evaluate_solution(ctx.prob, sol.coeffs, sol.λ, x_middle, 0.0, t)
        for t in ts_trace
    ]

    boundary_idx = argmax(boundary_trace)
    interior_idx = argmax(interior_trace)
    boundary_max = boundary_trace[boundary_idx]
    interior_max = interior_trace[interior_idx]

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

    return InfinityHotspotAnalysis(
        x_middle,
        t_final,
        ys_profile,
        ts_trace,
        u_inner_t0,
        u_middle_t0,
        u_outer_t0,
        u_middle_t1,
        u_outer_t1,
        boundary_trace,
        interior_trace,
        boundary_idx,
        interior_idx,
        boundary_max,
        interior_max,
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

    x_grid_pot = range(ctx.data.Lx, ctx.diam_x, length=nx_plot)
    y_grid_pot = range(-ctx.data.Ly, ctx.data.Ly, length=ny_plot)
    V_grid = [V_extended(ctx.pot, x, y) for x in x_grid_pot, y in y_grid_pot]

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

    p_pot = surface(x_grid_pot, y_grid_pot, V_grid';
        xlabel="x", ylabel="y", zlabel="V(x,y)",
        title="Potential in the wing region",
        colorbar=true)

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

pot = generate_potential();

ctx = build_context(pot;
    epsilon=DEFAULT_EPSILON,
    ext_factor=DEFAULT_EXT_FACTOR,
    Mx=DEFAULT_MX,
    Ny=DEFAULT_NY,
    N_grid=DEFAULT_N_GRID
);

sol = solve_problem(ctx; nev=DEFAULT_NEV, solver=DEFAULT_SOLVER);

hotspot = evaluate_hotspot(ctx, sol;
    x_middle=ctx.data.Lx + 2.0,
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
