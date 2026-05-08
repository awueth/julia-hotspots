using Revise
using Integrals
using Printf

includet("../potentials/potential_interface.jl")

using .PotentialInterface

t1 = 1.0
epsilon = 10.0
wing_length = 1.5 * pi
reltol = 1e-6
abstol = 1e-9

smooth_max_strength = 10.0
wing_scale = 5e6
checkpoint_path = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")

core = load_lse_core_potential(checkpoint_path=checkpoint_path)
domain = potential_domain(core)
wing = NonConvexWingPotential(
    domain.Lx;
    anchor=core_value(core, domain.Lx, domain.Ly),
    scale=wing_scale,
)
pot = SmoothMaxPotential(core, wing; smooth_max_strength=smooth_max_strength)

domain = potential_domain(pot)
diam_x = domain.Lx + wing_length

integrand(u, _) = begin
    x, y = u
    return 4.0 * exp(-(x^2 + y^2) / t1 - epsilon * potential_value(pot, x, y)) # factor of 4 since we only integrate over the first quadrant
end

prob = IntegralProblem(integrand, [0.0, 0.0], [diam_x, domain.Ly])
sol = solve(prob, HCubatureJL(); reltol=reltol, abstol=abstol)

C1 = inv(sol.u)

@printf("t1 = %.12g\n", t1)
@printf("epsilon = %.12g\n", epsilon)
@printf("domain = [0, %.12g] x [0, %.12g]\n", diam_x, domain.Ly)
@printf("integral = %.16e\n", sol.u)
@printf("C1 = %.16e\n", C1)
