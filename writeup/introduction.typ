#import "template.typ": *

= Introduction

Let $Ω subset RR^d$ be a connected, bounded domain with a sufficiently regular boundary $∂Ω$. Let $0 = lambda_0 < lambda_1 ≤ lambda_2 ≤ ...$ be the eigenvalues of the Neumann Laplacian on $Ω$ such that

$
-∆ phi_j &= lambda_j phi_j &&"in" Ω, \
∂_arrow(n) phi_j &= 0 &&"in" ∂Ω.
$ <eq:ev-problem>

where $phi_0, phi_1, phi_2, ...$ denote the corresponding eigenfunctions. Rauch's hot spots conjecture is commonly stated as:

#conjecture[
  For "nice" enough domains $Ω$, the first eigenfunction $phi_1$ attains its global extrema at the boundary $∂Ω$.
]<hs-conjecture>


In a 1975 lecture, J. Rauch @rauch_lecture_1975 introduced the conjecture as a problem in heat conduction: Suppose $Ω$ is some uniformly heat-conducting medium, that is insulated on the boundary. Let $u(t, x)$ be the temperature at $x in Ω$ at time $t$, and suppose the initial temperature distribution is given by $u_0$. Rauch posed the question of where the function $u(t, dot)$ achieves its maximum as $t -> oo$. The answer is given by @hs-conjecture, which can be stated as follows: _For generic initial conditions $u_0$, the point at which $u(t, dot)$ achieves its maximum converges to a point on the boundary._


To explain why this statement is equivalent to @hs-conjecture and define generic initial conditions, we note the temperature distribution can be modeled as

$
∂_t u &= ∆ u &&"in" (0, oo) times Ω, \
∂_arrow(n) u &= 0 &&"in" (0,oo) times ∂Ω.
$

By separating the variables $t$ and $x$, we can rewrite $u$ as

$
u(t, x) = ∑_(j≥0) chevron.l u_0, phi_j chevron.r phi_j (x) e^(-lambda_j t).
$

Since $phi_0$ is constant, this shows that $u$ converges to the mean $overline(u) = ∫_Ω  u_0$ as $t -> oo$ in the $L^2(Ω)$ sense. Convergence in the $L^oo$ sense follows by showing that, under mild regularity conditions on $Ω$, the functions ${u(t, dot)}_t$ are uniformly bounded and uniformly equicontinuous. Therefore, assuming for now that $lambda_2 > lambda_1$, for large $t$, the "shape" of $u(t, dot)$ will be dominated by the first eigenfunction in the sequence, $phi_j_0$ for which $chevron.l u_0, phi_j_0 chevron.r ≠ 0$. We say the initial condition $u_0$ is _generic_ if $chevron.l u_0, phi_1 chevron.r ≠ 0$. Hence, under generic initial conditions, $tilde(u)(t, x) = e^(lambda_1 t)(u(t,x) - overline(u))$ achieves its extrema at the same points as $u$, and

$
tilde(u)(t, x) = chevron.l u_0, phi_1 chevron.r phi_1 + ∑_(j≥2) chevron.l u_0, phi_(j) chevron.r phi_(j)(x) e^(-(lambda_j - lambda_1)t).
$

As $t -> oo$, the function $tilde(u)$ must converge exponentially fast to the function $chevron.l u_0, phi_1 chevron.r phi_1$, and in particular, the extremal points of $u$ converge subsequentially to the extremal points of $phi_1$.



*Non-generic initial conditions.* The requirement that the initial condition be generic is essential. If we naively formulate the hot spots conjecture as stating that the point $x_0 in Ω$ at which $u(t, dot)$ achieves its maximum tends to the boundary as $t -> oo$ for any initial condition, it is straightforward to construct a counterexample. Suppose $Ω subset RR^2$ is the unit disk and $u_0$ is a radially symmetric heat distribution that achieves a strict maximum at the origin. By symmetry, $u(t, dot)$ will also achieve its maximum at the origin at any time $t$. Such an initial condition, however, is never generic. The first non-constant eigenfunction $phi_1$ is antisymmetric, and therefore $chevron.l phi_1, u_0 chevron.r = 0$ (see @fig:disk-spectrum).

#figure(
  image("my_plot.png"),
  caption: [The first eigenfunctions of $-∆$ on a disk. Due to rotational symmetry, each eigenspace is two-dimensional.]
) <fig:disk-spectrum>


*Multiple ground eigenfunctions* The principal eigenspace can be two- or more dimensional; this is often the case if the domain $Ω$ is highly symmetric, such as a square or a disk. In such cases, there are several variations of the conjecture, depending on whether it should hold for all or only some eigenfunctions in the eigenspace. Following Bañuelos and Burdzy @banuelos_hot_1999, we distinguish three formulations:

(HS1) For _every_ eigenfunction $phi_1$ corresponding to $lambda_1$ which is not identically 0, and all $y in Ω$, we have $inf_(x in ∂Ω) phi_1(x) < phi_1(y) < sup_(x in ∂Ω) phi_1(x)$.

(HS2) For _every_ eigenfunction $phi_1$ corresponding to $lambda_1$ and all $y in Ω$, we have $inf_(x in ∂Ω) phi_1(x) ≤ phi_1(y) ≤ sup_(x in ∂Ω) phi_1(x)$.

(HS3) There _exists_ an eigenfunction $phi_1$ corresponding to $lambda_1$ which is not identically 0, and such that for all $y in Ω$, we have $inf_(x in ∂Ω) phi_1(x) < phi_1(y) < sup_(x in ∂Ω) phi_1(x)$.

In this work, we will build counterexamples with one-dimensional eigenspaces and eigenfunctions which attain a strict maximum in the interior, therefore disproving all three versions simultaneously. 

*What are nice enough domains?* The open question is identifying the classes of domains for which the conjecture holds and those for which it fails. The difficulty is that the principal eigenfunction of the Laplacian can be computed in closed form only for very few domains. For example, on a rectangle $phi_1$ is a product of trigonometric functions and the statement of the conjecture is easily verified. 

Kawohl @kawohl_rearrangements_1985[Corollary 2.15] showed that (HS1) holds for cylindrical domains $Ω = Ω_0 times (0,l) subset RR^d$ where $Ω_0$ is bounded with $∂Ω_0$ of class $C^(0,1)$. Furthermore, Kawohl conjectured that the property holds for all convex domains @kawohl_rearrangements_1985[p.56].
Bañuelos and Burdzy @banuelos_hot_1999 later proved that (HS1) holds for certain triangles and for long convex planar domains with sufficient symmetry. Conversely, Burdzy and Werner @burdzy_counterexample_1999 constructed a counterexample consisting of a bounded planar domain with two holes, the number of holes was later reduced to one @burdzy_hot_2004. For numerical counterexamples, see @kleefeld_hot_2021.

The conjecture was believed to be true for convex sets until it was recently disproven @pont_convex_2024 in high dimensions. This recent work is the subject of the next subsection.

== Convex sets can have interior hot spots

The counterexample in @pont_convex_2024 is built using a sequence of convex sets $Ω_d subset RR^(d+2)$ which are of the following nature:

$
Ω_d = {(x,w) in Q times RR^(d+1) : |w| ≤ 1/2 (sqrt(d) - V(x)/sqrt(d))},
$

where $Q subset RR^2$ is a rectangle and $V : Q -> RR$ is a convex potential. We now study the sequence of principal eigenfunctions $phi_(1,d)$ of $Ω_d$. First of all, notice that for $d$ large enough $phi_(1,d)$ is radial in the seecond argument $w$. We can therefore, overloading the notation, write $phi_(1,d) (x,r) := phi_(1,d) (x, 2d^(-1/2) abs(w))$. Under the change of variables $r = 2d^(-1/2) abs(w)$, the set $Ω_d$ becomes effectively three dimensional

$
Ω_d = {(x,r) in Q times RR_(≥0) : r ≤ 1 - V(x)/d},
$

and the eigenvalue equation $-∆ phi_(1,d) = lambda_(1,d) phi_(1,d)$ transforms into

$
∂_r phi_(1,d) &= r/4 (∆_x - lambda_(1,d)) phi_(1,d) + O(d^(-1)).
$

The sequence $phi_(1,d)$ converge uniformly to a function $phi_(1,oo) : Q times [0,1]$ which satisfies

$
∂_r phi_(1,d) &= r/4 (∆_x - lambda) phi_(1,d) \
L phi_(1,d)(x, 1) &= lambda_(1, oo) phi_(1,d) \
∂_r phi_(1,d)(x, r) &= 0 "at" r = 0 \
∂_arrow(n) phi_(1,d) &= 0,
$

where $L:= -∆ + nabla V dot nabla$, see @pont_convex_2024[Theorem 2.6]. The counterexample is now effectively built at $d = oo$, i.e. it is shown that there exists a convex potential $V$ such that $phi_(1,oo)$ does achieve its maximum in $Q^circle times [0,1)$. As a result there must be a critical dimension $d_"HS"$ such that for all $d≥d_"HS"$ the functions $phi_(1,d)$ achieve there maximum in $Q^circle times [0,1)$ as well. 

The dimension $d_"HS"$, at which the hot spot property starts to fail, is not explicit and the naive tracking of constants does not yield a satisfactory bound. In this work we aim to give an explicit lower bound of the dimension for which (HS1-HS3) starts to fail. We do this by solving for the principal eigenfunction $phi_1$ numerically and then certifying the counterexample using interval arithmetic.

== The method of particular solutions and certified numerics

We draw inspiration from @dahne_counterexample_2021, where the authors construct a numerical counterexample to Payne's nodal line conjecture using the method of particular solutions and rigorously certify the counterexample. The idea behind the method is simple. We explain it in context of our eigenvalue problem, but it transfers to any boundary value problem: Choose a set of functions ${f_j}_j$ which satisfy the eigenvalue equation $-∆ f_j = lambda_* f_j$ in the interior, disregarding the boundary conditions. Any linear combination $phi_* = ∑_j c_j f_j$ of functions in ${f_j}_j$ also satisfies the eigenvalue equation $-∆ phi_* = lambda_*$. In order to obtain an approximation of the true eigenfunction $phi_1$ we optimize the coefficients ${c_j}$ and the approximate eigenvalue $lambda^*$ with the objective of minimizing the error on the boundary.

Once we obtained a candidate approximation $phi_*$, we prove that if the error on the boundary is small $norm(partial_arrow(n) phi_*)_(L^oo (∂Ω))$, then $phi_*$ must be close to the true eigenfunction $phi_1$ point wise. One limitation of the method of particular solutions is that it does not guarantee the position in the spectrum of the approximation $phi_*$, that is, $phi_*$ could have small boundary error but be an approximation of the second eigenfunction $phi_2$ instead of $phi_1$. This can be particularly problematic in settings as in @dahne_counterexample_2021, where the eigenvalue of interest lies in a cluster. Luckily, in our case we have a large enough spectral gap $lambda_2 - lambda_1 > ?$. Furthermore we can show that $phi_1$ is the only eigenfunction satisfying certain geometric properties which are easy to check for the approximation $phi_*$.

Since $phi_*$ and $norm(partial_arrow(n) phi_*)_(L^oo (∂Ω))$ are evaluated numerically, they are subject to floating point arithmetic errors. To circumvent this issue, once we have constructed a counterexample $phi_*$, we compute rigorous enclosures of $phi_*$ and $norm(partial_arrow(n) phi_*)_(L^oo (∂Ω))$ using inverval arithmetic.



== Outline of the thesis

This thesis is structured as follows: In @barrels we introduce the class of sets used in @pont_convex_2024 to construct a counterexample. We show that on this class of sets the principal eigenfunction $phi_1$ is highly symmetrical along some dimensions allowing us to reduce the effective dimension of the set giving the counterexample to three. On this effectively three dimensional set we construct a basis for the method of particular solutions. \
In @construction we explain the counterexample in @pont_convex_2024 and derive an adaptation which can be explicitly computed. In @numerics we present our numerical implementation of our method of particular solutions and compute the rigorous error bounds. Finally, in @certificate we prove that with the error bound found in @numerics, there must be a counterexample in finite dimensions.
