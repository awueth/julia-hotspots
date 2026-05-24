#import "template.typ": *

= Solving for the eigenfunction Numerically <numerics>


#inline-note-a[
  Explain the spectral galerkin method for the infninite dimensional case and the collocation method.
]

We sample $m_B$ points $(x_i, r_i)$ from the boundary $Gamma_2 subset ∂Ω$

$
A_B (lambda) &= [ ∂_arrow(n) phi.alt_j (x_i, r_i) ]_(i,j).
$

We aim to find coefficients $c in RR^N$ such that $A_B c = 0$ under the constraint $norm(c)=1$. This is equivalent to finding the smallest singular value of $A_B$ and its corresponding right singular vector. The problem is, that the condition number of $A_B$ grows exponentially with the number of basis modes $N$, and, for large $N$, there exist linear combinations of $phi.alt_j$ that approximate the zero function in the interior @betcke_reviving_2005. A solution for this issue was proposed by @betcke_reviving_2005, and also used by @dahne_counterexample_2021: We sample $m_I$ points from the interior of $Ω$ to construct the matrix $A_I (lambda) &= [phi.alt_j (x_i, r_i) ]_(i,j)$. We factorize

$
A = mat(A_B; A_I) = mat(Q_B; Q_I) R = Q R,
$

to obtain an orthonormal basis $Q_B$ for the boundary. We then solve the problem

$
c = arg min_(norm(c)=1) norm(Q_B c) 
$

by computing the right singular vector of $Q_B$.


== Guaranteed error bounds using interval arithmetic

#inline-note-a[
  Here goes the theroem that we have an approximation $phi_*$ at dimension $d$ with a certain boundary error. 
]