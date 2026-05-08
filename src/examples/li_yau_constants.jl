using Revise
using Integrals
using Printf
using Random
import StochasticDiffEq as SDE

includet("../potentials/potential_interface.jl")

using .PotentialInterface

t1 = 1.0
t2 = 1.0
epsilon = 10.0
c2_mode = :reflected_core # :reflected_core or :soft_wall

wing_length = 1.5 * pi
reltol = 1e-6
abstol = 1e-9
sde_reltol = 1e-3
sde_abstol = 1e-6
sde_dt = 1e-2
nx_start = 11
ny_start = 11
ntrajectories = 128
seed = 1234

smooth_max_strength = 10.0
wing_scale = 5e6
checkpoint_path = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")

function build_potential()
    core = load_lse_core_potential(checkpoint_path=checkpoint_path)
    core_domain = potential_domain(core)
    wing = NonConvexWingPotential(
        core_domain.Lx;
        anchor=core_value(core, core_domain.Lx, core_domain.Ly),
        scale=wing_scale,
    )
    pot = SmoothMaxPotential(core, wing; smooth_max_strength=smooth_max_strength)
    return core, pot
end

function compute_C1(pot)
    domain = potential_domain(pot)
    diam_x = domain.Lx + wing_length

    integrand(u, _) = begin
        x, y = u
        # Factor of 4 because the numerical integral is over the first quadrant.
        return 4.0 * exp(-(x^2 + y^2) / t1 - epsilon * potential_value(pot, x, y))
    end

    prob = IntegralProblem(integrand, [0.0, 0.0], [diam_x, domain.Ly])
    sol = solve(prob, HCubatureJL(); reltol=reltol, abstol=abstol)

    return (
        C1=inv(sol.u),
        integral=sol.u,
        domain=(diam_x, domain.Ly),
    )
end

function compute_core_weighted_initial_integral(core)
    domain = potential_domain(core)

    integrand(u, _) = begin
        x, y = u
        return 4.0 * exp((x^2 + y^2) / (4.0 * t1) - epsilon * core_value(core, x, y))
    end

    prob = IntegralProblem(integrand, [0.0, 0.0], [domain.Lx, domain.Ly])
    sol = solve(prob, HCubatureJL(); reltol=reltol, abstol=abstol)

    return (
        integral=sol.u,
        domain=(domain.Lx, domain.Ly),
    )
end

function reflect_coordinate(z, upper)
    period = 2.0 * upper
    r = mod(z, period)
    return r <= upper ? r : period - r
end

function reflect_y!(u, Ly)
    u[2] = reflect_coordinate(u[2], Ly)
    return u
end

function reflect_core!(u, core_domain)
    u[1] = reflect_coordinate(u[1], core_domain.Lx)
    u[2] = reflect_coordinate(u[2], core_domain.Ly)
    return u
end

function smooth_max_gradient(pot::SmoothMaxPotential, x, y)
    core_val = core_value(pot.core, x, y)
    wing_val = wing_value(pot.wing, x, y)
    strength = pot.smooth_max_strength

    z_core = strength * core_val
    z_wing = strength * wing_val
    z_max = max(z_core, z_wing)
    core_weight = exp(z_core - z_max)
    wing_weight = exp(z_wing - z_max)
    total_weight = core_weight + wing_weight

    core_weight /= total_weight
    wing_weight /= total_weight

    core_gx, core_gy = core_gradient(pot.core, x, y)
    wing_gx, wing_gy = wing_gradient(pot.wing, x, y)

    return (
        core_weight * core_gx + wing_weight * wing_gx,
        core_weight * core_gy + wing_weight * wing_gy,
    )
end

function compute_C2(core, pot; mode::Symbol=c2_mode)
    core_domain = potential_domain(core)
    start_points = [
        (x0, y0)
        for x0 in range(0.0, core_domain.Lx; length=nx_start)
        for y0 in range(0.0, core_domain.Ly; length=ny_start)
    ]

    if mode == :reflected_core
        gradient = (x, y) -> core_gradient(core, x, y)
        reflect! = u -> reflect_core!(u, core_domain)
    elseif mode == :soft_wall
        gradient = (x, y) -> smooth_max_gradient(pot, x, reflect_coordinate(y, core_domain.Ly))
        reflect! = u -> reflect_y!(u, core_domain.Ly)
    else
        throw(ArgumentError("Unknown C2 mode `$mode`. Use :reflected_core or :soft_wall."))
    end

    sqrt2 = sqrt(2.0)

    function drift!(du, u, _, _)
        gx, gy = gradient(u[1], u[2])
        du[1] = -epsilon * gx
        du[2] = -epsilon * gy
        return nothing
    end

    function diffusion!(du, _, _, _)
        du[1] = sqrt2
        du[2] = sqrt2
        return nothing
    end

    reflect_callback = SDE.DiscreteCallback(
        (_, _, _) -> true,
        integrator -> reflect!(integrator.u);
        save_positions=(false, false),
    )

    terminal_weight(x) = begin
        reflect!(x)
        return exp((x[1]^2 + x[2]^2) / (4.0 * t1))
    end

    base_prob = SDE.SDEProblem(drift!, diffusion!, collect(start_points[1]), (0.0, t2))
    ensemble_prob = SDE.EnsembleProblem(
        base_prob;
        prob_func=(prob, i, _) -> begin
            point_idx = fld(i - 1, ntrajectories) + 1
            return remake(prob; u0=collect(start_points[point_idx]))
        end,
        output_func=(sol, i) -> begin
            point_idx = fld(i - 1, ntrajectories) + 1
            return ((point_idx, terminal_weight(sol.u[end])), false)
        end,
    )

    Random.seed!(seed)
    ensemble_sol = SDE.solve(
        ensemble_prob,
        SDE.SOSRA(),
        SDE.EnsembleThreads();
        trajectories=length(start_points) * ntrajectories,
        adaptive=false,
        dt=sde_dt,
        abstol=sde_abstol,
        reltol=sde_reltol,
        callback=reflect_callback,
        save_everystep=false,
        verbose=false,
    )

    sums = zeros(length(start_points))
    sumsq = zeros(length(start_points))
    counts = zeros(Int, length(start_points))
    failed_samples = 0

    for (point_idx, value) in ensemble_sol.u
        if !isfinite(value)
            failed_samples += 1
            continue
        end

        sums[point_idx] += value
        sumsq[point_idx] += value^2
        counts[point_idx] += 1
    end

    estimates = fill(-Inf, length(start_points))
    stderrs = fill(NaN, length(start_points))
    for i in eachindex(start_points)
        if counts[i] > 0
            estimates[i] = sums[i] / counts[i]
            variance = max(sumsq[i] / counts[i] - estimates[i]^2, 0.0)
            stderrs[i] = sqrt(variance / counts[i])
        end
    end

    C2_idx = argmax(estimates)
    return (
        C2=estimates[C2_idx],
        stderr=stderrs[C2_idx],
        maximizer=start_points[C2_idx],
        bound=exp((core_domain.Lx^2 + core_domain.Ly^2) / (4.0 * t1)),
        finite_samples=sum(counts),
        total_samples=length(start_points) * ntrajectories,
        failed_samples=failed_samples,
        core_domain=(core_domain.Lx, core_domain.Ly),
        mode=mode,
    )
end

core, pot = build_potential()
C1_result = compute_C1(pot)
core_initial_integral = compute_core_weighted_initial_integral(core)
C2_result = compute_C2(core, pot; mode=c2_mode)

@printf("t1 = %.12g\n", t1)
@printf("t2 = %.12g\n", t2)
@printf("epsilon = %.12g\n", epsilon)
@printf("C2 mode = %s\n", string(C2_result.mode))
@printf("C1 domain = [0, %.12g] x [0, %.12g]\n", C1_result.domain...)
@printf("C2 core domain = [0, %.12g] x [0, %.12g]\n", C2_result.core_domain...)
@printf("SDE grid = %d x %d, trajectories per point = %d, dt = %.12g, seed = %d\n", nx_start, ny_start, ntrajectories, sde_dt, seed)
@printf("integral = %.16e\n", C1_result.integral)
@printf("C1 = %.16e\n", C1_result.C1)
@printf("core weighted initial integral = %.16e\n", core_initial_integral.integral)
@printf("C2 ≈ %.16e\n", C2_result.C2)
@printf("C2 maximum-principle bound = %.16e\n", C2_result.bound)
@printf("C2 maximizer ≈ (%.12g, %.12g)\n", C2_result.maximizer...)
@printf("C2 Monte Carlo stderr ≈ %.16e\n", C2_result.stderr)
@printf("C2 finite samples = %d / %d\n", C2_result.finite_samples, C2_result.total_samples)
@printf("C2 failed samples = %d\n", C2_result.failed_samples)
