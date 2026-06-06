using Revise

includet("../solver/solver.jl")
includet("../solver/eigenfunction_visualization.jl")
includet("../potentials/potential_interface.jl")

using .PotentialInterface


const DEFAULT_EPSILON = 10.0
const DEFAULT_WING_LENGTH = 1.5 * pi
const DEFAULT_LSE_CORE_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_core_potential.chk")
const DEFAULT_SMOOTH_MAX_STRENGTH = 1.0
# const DEFAULT_LSE_WING_CHECKPOINT_PATH = joinpath(@__DIR__, "..", "..", "checkpoints", "lse_wing_potential.chk")
const DEFAULT_WING_SCALE = 1e6

ε = DEFAULT_EPSILON
core = load_lse_core_potential(checkpoint_path=DEFAULT_LSE_CORE_CHECKPOINT_PATH; Ly=1.0)
domain = potential_domain(core)
#wing = HandmadeWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE)
wing = NonConvexWingPotential(domain.Lx; scale=DEFAULT_WING_SCALE, anchor=core_value(core, domain.Lx, domain.Ly))
pot = SmoothMaxPotential(core, wing; smooth_max_strength=DEFAULT_SMOOTH_MAX_STRENGTH)
# wing = load_lse_wing_potential(
#     checkpoint_path=DEFAULT_LSE_WING_CHECKPOINT_PATH;
#     Lx=domain.Lx,
#     Ly=domain.Ly,
#     scale=DEFAULT_WING_SCALE,
# )
# pot = join_lse_potentials(core, wing)
domain = potential_domain(pot)
d = Inf
diam_x = 2.0 * (domain.Lx + DEFAULT_WING_LENGTH)
diam_y = 2.0 * domain.Ly
n_boundary_points = 256 * 64
n_modes = (128, 32)
λ = 3.9297514935298103

V, gradV = potential_functions(pot; scale=ε)

geometry = make_geometry(d, diam_x, diam_y, V, gradV, n_boundary_points)

# λ, _ = optimize_eigenvalue(geometry, n_modes, (3.85, 4.0))
coefficients, residual = solve_iterative(geometry, n_modes, λ)

#plot_u_boundary(geometry, coefficients, n_modes, λ)
plot_u_edge_profile(geometry, coefficients, n_modes, λ)

println("Infinity norm of boundary residual: ", maximum(abs.(residual)))

residual_fine, xs, ys = boundary_residual(geometry, coefficients, λ, n_modes, (1024, 128))
println("Infinity norm of fine residual: ", maximum(abs.(residual_fine)))
residual_plot = Plots.heatmap(
    xs,
    ys,
    abs.(residual_fine)';
    title="Boundary Residual",
    xlabel="x",
    ylabel="y",
    right_margin=5Plots.mm,
)
Plots.scatter!(
    residual_plot,
    geometry.points.x,
    geometry.points.y;
    label="Collocation points",
    markercolor=:white,
    markeralpha=0.35,
    markersize=1.0,
    markerstrokewidth=0,
)
display(residual_plot)

lx, ly, lr = get_eigenvalues(geometry.diam_x, geometry.diam_y, n_modes, λ)
u_star(x, y, r) = u(geometry.d, coefficients, lx, ly, lr, geometry.diam_x, geometry.diam_y, x, y, r)

xp = range(0.5 * pi + 0.75, 0.5 * diam_x, 128)
yp = range(-1.0, 1.0, 128)
tp = [ones(20)..., (1.0 .- logrange(0.02, 1.0, 20))..., zeros(20)...]
#tp = 1.0 .- logrange(0.02, 1.0, 20)

surface(range(-0.5 * diam_x, 0.5 * diam_x, 64), yp, (x, y) -> u_star(x, y, 1.0))
heatmap(range(-0.5 * diam_x, 0.5 * diam_x, 128), yp, (x, y) -> u_star(x, y, 1.0))

heatmap_anim = @animate for t in tp
    i_max = argmax(u_star.(xp, 0.0, t))
    heatmap(xp, yp, (x, y) -> u_star(x, y, t), color=:jet, colorbar=false)
    annotate!(xp[i_max], 0.0, text("×", :white, 12))
end
gif(heatmap_anim, "eigenfunction_heatmap.mp4", fps=10)


using Printf

begin
    x_minimap = range(-0.5 * diam_x, 0.5 * diam_x, 60)
    cam = (30, 30)

    surface_anim = @animate for t in tp
        title_str = @sprintf("Zoomed Detail (r = %.3f)", t)
        
        # 1. MINIMAP / CONTEXT
        p_left = surface(
            x_minimap, yp, (x, y) -> u_star(x, y, t),
            alpha = 0.3, 
            color = :lightgray,
            camera = cam,
            colorbar = false, 
            title = "Full Domain",
            zlims = (-0.4, 0.4),
            margin = 5Plots.mm,
        )
        surface!(p_left, xp, yp, (x, y) -> u_star(x, y, t), 
                 color = :lightgreen,  # Set to a single color
                 alpha = 0.8,           # Slightly transparent to see the gray mesh
                 label = "")

        # 2. MAIN DETAIL PLOT
        p_right = surface(
            xp, yp, (x, y) -> u_star(x, y, t), 
            zaxis=false, grid=false, colorbar=false, color=:jet,
            title = title_str,
            camera = cam,
            margin = 5Plots.mm,
            zticks = false,
        )
        
        i_max = argmax(u_star.(xp, 0.0, t))
        scatter!(p_right, [xp[i_max]], [0.0], [u_star(xp[i_max], 0.0, t)], 
                label="", markercolor=:white)

        plot(p_left, p_right, layout=grid(1, 2, widths=[0.4, 0.6]), size=(1000, 500))
    end
end
gif(surface_anim, "eigenfunction_surface.mp4", fps=10)