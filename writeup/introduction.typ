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

The counterexample in @pont_convex_2024 pursues the following strategy: The authors establish the log-concave analog of the hot spots conjecture. That is, instead of a Neumann eigenfunction minimizing $(integral_Omega abs(nabla phi)^2 dif x) / (integral_Omega abs(phi)^2 dif x)$, with respect to the uniform measure $bb(1)_Omega$, they generalize to log-concave measures $mu$

$
lambda_1 (mu) = inf_(phi) (integral_Omega abs(nabla phi)^2 dif mu) / (integral_Omega abs(phi)^2 dif mu).
$

They then show that there exists a log-concave extension of the conjecture is false. They transfer to the uniform measure, by simulating the log-concave measure $mu$ by a uniform measure on so called Barrel sets in high dimensions. These are sets of the form

$ F_d (Q, V) := {(x,w) in Q times RR^(d+1) : |w| ≤ sqrt(d)/2 (1 - V(x)/d)}. $


The map $Phi : (x,w) |-> (x, (1-V(x)/d)^(-1) w)$ is a bijection between $F_d (Q, V)$ and $Q times B_(sqrt(d)/2)$. The pushforward of the uniform measure on $F_d (Q, V)$ under $Phi$ approximates the log-concave measure with density $e^(-V) dif x dif w$ on $Q times B_(sqrt(d)/2)$. As a result, we expect $norm(phi_(i, d) compose Phi - phi_(mu, i))_(L^2(mu)) -> 0$, as $d -> oo$. The authors then show that the principal eigenfunction $phi_(mu, 1)$ of the log-concave measure $mu$ has a strict maximum in the interior of $Q times B_(sqrt(d)/2)$, and therefore, for sufficiently large $d$, the principal eigenfunction $phi_(1, d)$ of the uniform measure on $F_d (Q, V)$ must also have a strict maximum in the interior. The dimension $d_"HS"$, at which the hot spot property starts to fail, is not explicit and the naive tracking of constants does not yield a satisfactory bound. In this work we aim to give an explicit lower bound of the dimension for which (HS1-HS3) starts to fail.

== The method of particular solutions and certified numerics

We draw inspiration from @dahne_counterexample_2021, where the authors construct a numerical counterexample to Payne's nodal line conjecture using the method of particular solutions and rigorously certify the counterexample. The idea behind the method is simple. We explain it in context of our eigenvalue problem, but it transfers to any boundary value problem: Choose a set of functions ${f_j}_j$ which satisfy the eigenvalue equation $-∆ f_j = lambda_* f_j$ in the interior, disregarding the boundary conditions. Any linear combination $phi_* = ∑_j c_j f_j$ of functions in ${f_j}_j$ also satisfies the eigenvalue equation $-∆ phi_* = lambda_*$. In order to obtain an approximation of the true eigenfunction $phi_1$ we optimize the coefficients ${c_j}$ and the approximate eigenvalue $lambda^*$ with the objective of minimizing the error on the boundary.

Once we obtained a candidate approximation $phi_*$, we prove that if the error on the boundary is small $norm(partial_arrow(n) phi_*)_(L^oo (∂Ω))$, then $phi_*$ must be close to the true eigenfunction $phi_1$ point wise. One limitation of the method of particular solutions is that it does not guarantee the position in the spectrum of the approximation $phi_*$, that is, $phi_*$ could have small boundary error but be an approximation of the second eigenfunction $phi_2$ instead of $phi_1$. This can be particularly problematic in settings as in @dahne_counterexample_2021, where the eigenvalue of interest lies in a cluster. Luckily, in our case we have a large enough spectral gap $lambda_2 - lambda_1 > ?$. Furthermore we can show that $phi_1$ is the only eigenfunction satisfying certain geometric properties which are easy to check for the approximation $phi_*$.

Since $phi_*$ and $norm(partial_arrow(n) phi_*)_(L^oo (∂Ω))$ are evaluated numerically, they are subject to floating point arithmetic errors. To circumvent this issue, once we have constructed a counterexample $phi_*$, we compute rigorous enclosures of $phi_*$ and $norm(partial_arrow(n) phi_*)_(L^oo (∂Ω))$ using inverval arithmetic.



== Outline of the thesis

This thesis is structured as follows: In @barrels we introduce the class of sets used in @pont_convex_2024 to construct a counterexample. We show that on this class of sets the principal eigenfunction $phi_1$ is highly symmetrical along some dimensions allowing us to reduce the effective dimension of the set giving the counterexample to three. On this effectively three dimensional set we construct a basis for the method of particular solutions. \
In @construction we explain the counterexample in @pont_convex_2024 and derive an adaptation which can be explicitly computed. In @numerics we present our numerical implementation of our method of particular solutions and compute the rigorous error bounds. Finally, in @certificate we prove that with the error bound found in @numerics, there must be a counterexample in finite dimensions.
