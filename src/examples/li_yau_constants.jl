using Revise
using Integrals
using Optim
using Printf

includet("../potentials/potential_lab.jl")

using .PotentialLab

t1_initial = inv(4.0)
t2_initial = inv(4.0)
t = t1_initial + t2_initial

epsilon = 10.0
wing_length = 1.5 * pi
reltol = 1e-6
abstol = 1e-9

smooth_max_strength = 10.0
wing_scale = 5e6
Mx = 0.6872455106751706
checkpoint_path = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")

function build_potential(;
    checkpoint_path::AbstractString,
    wing_scale::Real,
    smooth_max_strength::Real,
)
    core = load_lse_core_potential(checkpoint_path=checkpoint_path)
    domain = potential_domain(core)
    wing = NonConvexWingPotential(
        domain.Lx;
        anchor=core_value(core, domain.Lx, domain.Ly),
        scale=wing_scale,
    )
    pot = SmoothMaxPotential(core, wing; smooth_max_strength=smooth_max_strength)
    return core, pot
end

function compute_normalization(
    pot;
    epsilon::Real,
    wing_length::Real,
    reltol::Real,
    abstol::Real,
)
    domain = potential_domain(pot)
    diam_x = domain.Lx + wing_length
    lower = [0.0, 0.0]
    upper = [diam_x, domain.Ly]

    integrand(u, _) = begin
        x, y = u
        # Factor of 4 because the numerical integral is over the first quadrant.
        return 4.0 * exp(-epsilon * potential_value(pot, x, y))
    end

    prob = IntegralProblem(integrand, (lower, upper))
    sol = solve(prob, HCubatureJL(); reltol=reltol, abstol=abstol)

    return (
        Z=sol.u,
        domain=(diam_x, domain.Ly),
    )
end

function compute_tail_mass(
    pot,
    normalization,
    epsilon::Real,
    wing_length::Real,
    x_tail::Real
)
    domain = potential_domain(pot)
    diam_x = domain.Lx + wing_length
    lower = [x_tail, 0.0]
    upper = [diam_x, domain.Ly]

    integrand(u, _) = begin
        x, y = u
        return 4.0 * exp(-epsilon * potential_value(pot, x, y)) / normalization.Z
    end

    prob = IntegralProblem(integrand, (lower, upper))
    sol = solve(prob, HCubatureJL(); reltol=1e-12, abstol=1e-12)

    return sol.u
end

function compute_C1(
    pot;
    t1::Real,
    epsilon::Real,
    wing_length::Real,
    Z::Real,
    reltol::Real,
    abstol::Real,
)
    t1 > 0 || return (C1=Inf, normalized_integral=NaN, domain=(NaN, NaN))
    Z > 0 || return (C1=Inf, normalized_integral=NaN, domain=(NaN, NaN))

    domain = potential_domain(pot)
    diam_x = domain.Lx + wing_length
    lower = [0.0, 0.0]
    upper = [diam_x, domain.Ly]

    integrand(u, _) = begin
        x, y = u
        # Factor of 4 because the numerical integral is over the first quadrant.
        return 4.0 * exp(-(x^2 + y^2) / t1 - epsilon * potential_value(pot, x, y)) / Z
    end

    prob = IntegralProblem(integrand, (lower, upper))
    sol = solve(prob, HCubatureJL(); reltol=reltol, abstol=abstol)

    return (
        C1=1/sol.u,
        normalized_integral=sol.u,
        domain=(diam_x, domain.Ly),
    )
end

# Weighted convexification: Mx = 0.6872455106751706, My = 2.4678490408651026, cost = 3.155094551540273
function compute_C2(
    pot;
    t1::Real,
    t2::Real,
    epsilon::Real,
    wing_length::Real,
    Mx::Real,
)
    if t1 <= 0 || t2 < 0 || epsilon <= 0 || Mx <= 0
        return Inf
    end

    domain = potential_domain(pot)
    diam_x = domain.Lx + wing_length

    alpha = 1.0 / (4.0 * t1)
    epsilon_Mx = epsilon * Mx
    denominator = 2.0 + (epsilon_Mx / alpha - 2.0) * exp(2.0 * epsilon_Mx * t2)

    if !isfinite(denominator) || denominator <= 0
        return Inf
    end

    a = epsilon_Mx / denominator
    if !isfinite(a) || a <= 0
        return Inf
    end

    b = epsilon_Mx * t2 + 0.5 * log(a / alpha)
    exponent = a * (0.5 * diam_x)^2 + b
    return isfinite(exponent) ? exp(exponent) : Inf
end

function compute_C(
    pot;
    t1::Real,
    t2::Real,
    epsilon::Real,
    wing_length::Real,
    Mx::Real,
    Z::Real,
    reltol::Real,
    abstol::Real,
)
    C1_result = compute_C1(
        pot;
        t1=t1,
        epsilon=epsilon,
        wing_length=wing_length,
        Z=Z,
        reltol=reltol,
        abstol=abstol,
    )
    C2 = compute_C2(
        pot;
        t1=t1,
        t2=t2,
        epsilon=epsilon,
        wing_length=wing_length,
        Mx=Mx,
    )

    C = C1_result.C1 * C2
    return (
        C=C,
        C1=C1_result.C1,
        C2=C2,
        C1_result=C1_result,
        t1=Float64(t1),
        t2=Float64(t2),
    )
end

function optimize_C_over_split(
    pot;
    t::Real,
    epsilon::Real,
    wing_length::Real,
    Mx::Real,
    Z::Real,
    reltol::Real,
    abstol::Real,
)
    t > 0 || throw(ArgumentError("t must be positive."))

    lower = 1e-6 * t
    upper = (1.0 - 1e-6) * t

    objective(t1) = begin
        result = compute_C(
            pot;
            t1=t1,
            t2=t - t1,
            epsilon=epsilon,
            wing_length=wing_length,
            Mx=Mx,
            Z=Z,
            reltol=reltol,
            abstol=abstol,
        )
        return result.C
    end

    opt = optimize(objective, lower, upper, Brent())
    best_t1 = Optim.minimizer(opt)
    best = compute_C(
        pot;
        t1=best_t1,
        t2=t - best_t1,
        epsilon=epsilon,
        wing_length=wing_length,
        Mx=Mx,
        Z=Z,
        reltol=reltol,
        abstol=abstol,
    )

    return (
        result=opt,
        t1=best.t1,
        t2=best.t2,
        C=best.C,
        C1=best.C1,
        C2=best.C2,
        C1_result=best.C1_result,
    )
end

_, pot = build_potential(
    checkpoint_path=checkpoint_path,
    wing_scale=wing_scale,
    smooth_max_strength=smooth_max_strength,
)

normalization = compute_normalization(
    pot;
    epsilon=epsilon,
    wing_length=wing_length,
    reltol=reltol,
    abstol=abstol,
)

reference = compute_C(
    pot;
    t1=t1_initial,
    t2=t2_initial,
    epsilon=epsilon,
    wing_length=wing_length,
    Mx=Mx,
    Z=normalization.Z,
    reltol=reltol,
    abstol=abstol,
)

optimized = optimize_C_over_split(
    pot;
    t=t,
    epsilon=epsilon,
    wing_length=wing_length,
    Mx=Mx,
    Z=normalization.Z,
    reltol=reltol,
    abstol=abstol,
)

@printf("epsilon = %.12g\n", epsilon)
@printf("total t = %.12g\n", t)
@printf("Mx = %.16g\n", Mx)
@printf("integration domain = [0, %.12g] x [0, %.12g]\n", normalization.domain...)

@printf("\nReference split:\n")
@printf("  t1 = %.16e\n", reference.t1)
@printf("  t2 = %.16e\n", reference.t2)
@printf("  C1 = %.16e\n", reference.C1)
@printf("  C2 = %.16e\n", reference.C2)
@printf("  C = %.16e\n", reference.C)

@printf("\nOptimized split:\n")
@printf("  t1 = %.16e\n", optimized.t1)
@printf("  t2 = %.16e\n", optimized.t2)
@printf("  t1 + t2 = %.16e\n", optimized.t1 + optimized.t2)
@printf("  C1 = %.16e\n", optimized.C1)
@printf("  C2 = %.16e\n", optimized.C2)
@printf("  C = %.16e\n", optimized.C)
@printf("  converged = %s\n", string(Optim.converged(optimized.result)))
@printf("  iterations = %d\n", Optim.iterations(optimized.result))
