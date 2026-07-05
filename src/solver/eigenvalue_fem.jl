if !isdefined(@__MODULE__, :LSEPotentials)
    include(joinpath(@__DIR__, "..", "potentials", "lse_potential.jl"))
end

module EigenvalueFEM

using Gridap
using Gridap.Geometry: add_tag!, get_cell_coordinates
using Gridap.ReferenceFEs: crouzeix_raviart
using Arpack
using LinearAlgebra
using SparseArrays

using ..LSEPotentials: LSEPotential, potential_value

function simplex_oscillation(V, vertices)
    values = V.(vertices)
    return maximum(values) - minimum(values)
end

function is_second_quadrant_simplex(vertices; atol=1e-12)
    all(v -> v[1] <= atol && v[2] >= -atol, vertices)
end

function max_second_quadrant_simplex_oscillation(V, trian)
    cell_vertices = get_cell_coordinates(trian)
    cells = filter(cell -> is_second_quadrant_simplex(cell_vertices[cell]), eachindex(cell_vertices))
    isempty(cells) && error("No simplices found in the second quadrant.")

    oscillations = map(cell -> simplex_oscillation(V, cell_vertices[cell]), cells)
    i = argmax(oscillations)
    cell = cells[i]
    return (
        value=oscillations[i],
        cell=cell,
        vertices=cell_vertices[cell],
        ncells=length(cells),
    )
end

function compute_fem_eigenvalues(pot::LSEPotential; partition=(256, 64), nev=4)
    domain = (-pi/2, pi/2, -1.0, 1.0)
    model = simplexify(CartesianDiscreteModel(domain, partition))

    labels = get_face_labeling(model)
    add_tag!(labels, "empty_dirichlet", Int[])
    refe = ReferenceFE(crouzeix_raviart, Float64, 1)
    V = FESpace(model, refe; labels=labels, dirichlet_tags="empty_dirichlet")
    U = TrialFESpace(V)
    
    V_pot(x) = potential_value(pot, x[1], x[2])
    weight(x) = exp(-V_pot(x))

    trian = Triangulation(model)
    degree = 4
    dΩ = Measure(trian, degree)

    a(u, v) = ∫( weight * (∇(v) ⋅ ∇(u)) )dΩ
    b(u, v) = ∫( weight * (v * u) )dΩ

    A_op = AffineFEOperator(a, (v)->0.0, U, V)
    A = get_matrix(A_op)

    B_op = AffineFEOperator(b, (v)->0.0, U, V)
    B = get_matrix(B_op)

    v_init = ones(size(A, 1)) # Initial guess for eigenvector
    values, vectors = eigs(A, B, nev=nev, which=:SR, v0=v_init, maxiter=10000, ncv=40)

    values = real(values)
    vectors = real(vectors)
    perm = sortperm(values)
    values = values[perm]
    vectors = vectors[:, perm]

    oscillation = max_second_quadrant_simplex_oscillation(V_pot, trian)

    return values, vectors, oscillation
end

end # module
