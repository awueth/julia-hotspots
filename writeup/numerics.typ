#import "template.typ": *

= Solving for the eigenfunction Numerically <numerics>


In this section we explain how we compute approximations of the Neumann eigenfunctions of the Barrel sets. We use the Method of Particular Solutions (MPS) as described by @betcke_reviving_2005. That is, we write our approximate eigenfunction $phi.alt_*$ as a finite linear combination $sum_j c_j b_j$ of the basis functions derived in @mps_basis. The basis functions are such that $-∆ b_j = lambda_*$ for all $j$ any chosen $lambda_*$ which is a parameter of $b_j$. As a result, the approximation $phi.alt_*$ satisfies $-∆ phi.alt_* = lambda_*$ exactly. We aim to find coefficients $c_j$ and a value $lambda_*$, which minimize the error in the Neumann boundary condition. We sample $m_B$ points $(x_i, r_i)$ from the boundary $Gamma_2 subset ∂Ω$. Using these collocation points we define the matrix

$
A_B (lambda) &= [ ∂_arrow(n) b_j (x_i, r_i) ]_(i,j).
$

Finding coefficients $c in RR^N$ that minimize $norm(A_B c)^2 = sum_i abs(∂_arrow(n) phi.alt_* (x_i ,r_i))^2$ under the constraint $norm(c)=1$ is equivalent to finding the smallest singular value of $A_B$ and its corresponding right singular vector. The problem is, that the condition number of $A_B$ grows exponentially with the number of basis modes $N$, and, for large $N$, there exist linear combinations of $b_j$ that approximate the zero function in the interior, called _spurious solutions_  @betcke_reviving_2005. A solution for this issue was proposed by @betcke_reviving_2005, and also used by @dahne_counterexample_2021: Sample $m_I$ points from the interior of $Ω$ to construct the matrix $A_I (lambda) &= [b_j (x_i, r_i) ]_(i,j)$. We factorize

$
A = mat(A_B; A_I) = mat(Q_B; Q_I) R = Q R.
$

The columns of $Q_B$ and $Q_I$ form a basis of the space of trial functions sampled at the boundary and the interior respectively. Let $sigma(lambda)$ be the least singular value of $Q_B$ and $v_*$ the corresponding right singular vector. Then,

$
min_(norm(v)=1) norm(Q_B v) = sigma(lambda)
$

and $v_*$ is the minimizer. Observe that,

$
1 = norm(Q v_*)^2 = norm(Q_B v_*)^2 +  norm(Q_I v_*)^2 = sigma(lambda)^2 + norm(Q_I v_*)^2,
$

i.e. any coefficient vector that produces a small boundary error, automatically corresponds to a function that has approximate unit norm. Therefore, spurious solutions are automatically ruled out.

In $d=oo$ dimensions we target an approximation $phi.alt_*$ that is of unit norm with respect to the measure with density $dif mu = 1/Z e^(-V) dif x$. This could be achieved by multiplying each entry of $A$ by $Z^(-1) e^(-V(x_i))$. However, since $V$ is large in the wings, this weight would simply underflow. We resort to only weighting the points of the interior matrix. This is not a problem, since our true objective is not minimizing the $L^2$-norm of the boundary error, but rather the $L^(oo)$ norm.


== Results

#theorion-restate(filter: <thm:main-finite-dim-result>)