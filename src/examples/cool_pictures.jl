using Revise
using Printf

includet("../solver/solver.jl")
includet("../solver/eigenfunction_io.jl")
includet("../solver/eigenfunction_visualization.jl")

fit = load_fitted_eigenfunction(joinpath(@__DIR__, "..", "..", "checkpoints", "fitted_eigenfunction.chk"))

coefficients = fit.coefficients
diam_x = fit.diam_x
diam_y = fit.diam_y

lx, ly, lr = get_eigenvalues(fit.diam_x, diam_y, fit.n_modes, fit.λ)
u_star(x, y, r) = u(fit.d, coefficients, lx, ly, lr, diam_x, diam_y, x, y, r)

xp = range(0.5 * pi + 0.8, 0.5 * diam_x, 64)
yp = range(-1.0, 1.0, 64)

# macroscopic plot of the eigenfunction
begin
    x_minimap = range(-0.5 * diam_x, 0.5 * diam_x, 60)
    cam = (-30, 30)
        
    # 1. MINIMAP / CONTEXT
    s1 = surface(
        x_minimap, yp, (x, y) -> u_star(x, y, 1.0),
        color = :gray,
        alpha = 0.5, 
        zaxis=false,
        grid=false,
        camera = cam,
        colorbar = false, 
        zlims = (-0.5, 0.5),
    )
    s1 = surface!(xp, yp, (x, y) -> u_star(x, y, 1.0), 
                color = :lightgreen,  # Set to a single color
                alpha = 0.8,  
                label = "")

    display(s1)
    savefig(s1, "surface_plot_eigenfunction_boundary.png")

    s2 = surface(
        x_minimap, yp, (x, y) -> u_star(x, y, 0.0),
        color = :gray,
        alpha = 0.5, 
        zaxis=false,
        grid=false,
        camera = cam,
        colorbar = false,
        zlims = (-0.5, 0.5),
    )
    s2 = surface!(xp, yp, (x, y) -> u_star(x, y, 0.0), 
                color = :lightgreen,  # Set to a single color
                alpha = 0.8,  
                label = "")

    display(s2)
    savefig(s2, "surface_plot_eigenfunction_interior.png")
end

# heatmap of the wing region
begin
    h1 = heatmap(
        xp, yp, (x, y) -> u_star(x, y, 1.0), 
        zaxis=false, grid=false, colorbar=false, color=:jet,
        camera = cam,
        margin = 5Plots.mm,
        zticks = false,
    )

    display(h1)
    savefig(h1, "heatmap_eigenfunction_boundary.png")

    h2 = heatmap(
        xp, yp, (x, y) -> u_star(x, y, 0.0), 
        zaxis=false, grid=false, colorbar=false, color=:jet,
        camera = cam,
        margin = 5Plots.mm,
        zticks = false,
    )

    display(h2)
    savefig(h2, "heatmap_eigenfunction_interior.png")
end

#savefig(p_left, "eigenfunction_minimap.png")
#savefig(p_right, "eigenfunction_detail.png")