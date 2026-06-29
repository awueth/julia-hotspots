using Gridap
using Gridap.Geometry: add_tag!
using Gridap.ReferenceFEs: crouzeix_raviart
using Arpack
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "potentials", "lse_potential.jl"))
using .LSEPotentials

# 1. Setup Domain and Mesh
domain = (-pi/2, pi/2, -1.0, 1.0)
partition = (64, 64)
model = simplexify(CartesianDiscreteModel(domain, partition))

# 2. Crouzeix-Raviart FE Space for Neumann
# Gridap treats CR as L2-conforming. With no Dirichlet tags, the high-level
# constructor takes the discontinuous branch. An empty tag forces the generic
# facet-glued construction while preserving pure Neumann boundary conditions.
labels = get_face_labeling(model)
add_tag!(labels, "empty_dirichlet", Int[])
refe = ReferenceFE(crouzeix_raviart, Float64, 1)
V = FESpace(model, refe; labels=labels, dirichlet_tags="empty_dirichlet")
U = TrialFESpace(V)

# 3. Load the LSE core potential and define the weight exp(-V)
pot = load_lse_potential(joinpath(@__DIR__, "..", "..", "checkpoints", "lse_global_potential.chk"))
V_pot(x) = potential_value(pot, x[1], x[2])
weight(x) = exp(-V_pot(x))

# 4. Define Integration Quadrature
trian = Triangulation(model)
degree = 4 # Slightly higher degree recommended due to the exponential weight
dΩ = Measure(trian, degree)

# 5. Define Symmetric Bilinear Forms with exp(-V)
# Both u and v are now weighted by exp(-V)
a(u, v) = ∫( weight * (∇(v) ⋅ ∇(u)) )dΩ
b(u, v) = ∫( weight * (v * u) )dΩ

# 6. Assemble into Global Sparse Matrices
A_op = AffineFEOperator(a, (v)->0.0, U, V)
A = get_matrix(A_op)

B_op = AffineFEOperator(b, (v)->0.0, U, V)
B = get_matrix(B_op)

# 7. Solve using Arpack
# Since A and B are symmetric and B is positive definite, ask directly for the
# smallest-real generalized eigenpairs. A shifted solve near zero returns the
# high end of the spectrum for this Arpack generalized problem.
num_eigenvalues = 4
v_init = ones(size(A, 1)) # Initial guess for eigenvector
values, vectors = eigs(A, B, nev=num_eigenvalues, which=:SR, v0=v_init, maxiter=10000, ncv=40)

# Because the system is symmetric, outputs are strictly real numbers 
# (any tiny complex part is just floating-point roundoff)
values = real(values)
vectors = real(vectors)
perm = sortperm(values)
values = values[perm]
vectors = vectors[:, perm]

println("Found Symmetric Neumann eigenvalues: ", values)

# 8. Save the first non-constant eigenmode
nonconstant_mode = findfirst(λ -> λ > 1e-8, values)
if isnothing(nonconstant_mode)
    error("No positive Neumann eigenvalue found among the computed eigenpairs.")
end
u2_dofs = vectors[:, nonconstant_mode]
uh2 = FEFunction(U, u2_dofs)
writevtk(trian, "symmetric_neumann_mode", cellfields=["u2"=>uh2])
