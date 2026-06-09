using Revise
using Printf

includet("../solver/solver.jl")
includet("../potentials/potential_interface.jl")
includet("../solver/eigenfunction_io.jl")
includet("../solver/eigenfunction_visualization.jl")

fit = load_fitted_eigenfunction(joinpath(@__DIR__, "..", "..", "checkpoints", "fitted_eigenfunction.chk"))

coefficients = fit.coefficients
diam_x = fit.diam_x
diam_y = fit.diam_y

lx, ly, lr = get_eigenvalues(fit.diam_x, diam_y, fit.n_modes, fit.λ)
u_star(x, y, r) = u(fit.d, coefficients, lx, ly, lr, diam_x, diam_y, x, y, r)

xp = range(0.5 * pi, 0.5 * diam_x, 64)
yp = range(-1.0, 1.0, 64)

begin
    x_minimap = range(-0.5 * diam_x, 0.5 * diam_x, 60)
    cam = (-30, 30)

    z_minimap = [u_star(x, y, 1.0) for x in x_minimap, y in yp]
    color_vals_minimap = zeros(size(z_minimap))
        
    # 1. MINIMAP / CONTEXT
    p_left = surface(
        x_minimap, yp, z_minimap,
        zcolor = color_vals_minimap,
        # alpha = 0.3, 
        # zaxis=false,
        # grid=false,
        # camera = cam,
        # colorbar = false, 
        # zlims = (-0.4, 0.4),
        # margin = 5Plots.mm,
    )
    surface!(p_left, xp, yp, (x, y) -> u_star(x, y, 1.0), 
                color = :jet,  # Set to a single color
                alpha = 0.8,  
                label = "")

    # 2. MAIN DETAIL PLOT
    p_right = heatmap(
        xp, yp, (x, y) -> u_star(x, y, 1.0), 
        zaxis=false, grid=false, colorbar=false, color=:jet,
        camera = cam,
        margin = 5Plots.mm,
        zticks = false,
    )

    plot(p_left, p_right, layout=grid(1, 2, widths=[0.4, 0.6]), size=(1000, 500))
end

savefig(p_left, "eigenfunction_minimap.png")
savefig(p_right, "eigenfunction_detail.png")