#import "template.typ": *

= Solving for the eigenfunction Numerically <numerics>


In this section we explain how we compute approximations of the Neumann eigenfunctions of the Barrel sets. We use the Method of Particular Solutions (MPS) as described by @betcke_reviving_2005. That is, we write our approximate eigenfunction $phi.alt_*$ as a finite linear combination $sum_j c_j phi.alt_j$ of the basis functions derived in @mps_basis. The basis functions are such that $-∆ phi.alt_j = lambda_*$ for all $j$ any chosen $lambda_*$ which is a parameter of $phi.alt_j$. As a result, the approximation $phi.alt_*$ satisfies $-∆ phi.alt_* = lambda_*$ exactly. We aim to find coefficients $c_j$ and a value $lambda_*$, which minimize the error in the Neumann boundary condition. We sample $m_B$ points $(x_i, r_i)$ from the boundary $Gamma_2 subset ∂Ω$. Using these collocation points we define the matrix

$
A_B (lambda) &= [ ∂_arrow(n) phi.alt_j (x_i, r_i) ]_(i,j).
$

Finding coefficients $c in RR^N$ that minimize $norm(A_B c)^2 = sum_i abs(∂_arrow(n) phi.alt_* (x_i ,r_i))^2$ under the constraint $norm(c)=1$ is equivalent to finding the smallest singular value of $A_B$ and its corresponding right singular vector. The problem is, that the condition number of $A_B$ grows exponentially with the number of basis modes $N$, and, for large $N$, there exist linear combinations of $phi.alt_j$ that approximate the zero function in the interior @betcke_reviving_2005. A solution for this issue was proposed by @betcke_reviving_2005, and also used by @dahne_counterexample_2021: We sample $m_I$ points from the interior of $Ω$ to construct the matrix $A_I (lambda) &= [phi.alt_j (x_i, r_i) ]_(i,j)$. We factorize

$
A = mat(A_B; A_I) = mat(Q_B; Q_I) R = Q R,
$

to obtain an orthonormal basis $Q_B$ for the boundary. We then solve the problem

$
c = arg min_(norm(c)=1) norm(Q_B c) 
$

by computing the right singular vector of $Q_B$.


== Results

#inline-note-a[
  Here goes the theroem that we have an approximation $phi_*$ at dimension $d$ with a certain boundary error. 
]