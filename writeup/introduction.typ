#import "template.typ": *
#import "numerical_values.typ": *

= Introduction

Let $Ω subset RR^d$ be a connected, bounded domain with a sufficiently regular boundary $∂Ω$. Let $0 = lambda_0 < lambda_1 ≤ lambda_2 ≤ ...$ be the eigenvalues of the Neumann Laplacian on $Ω$ such that

$
cases(
-∆ phi.alt_j &= lambda_j phi.alt_j &&"in" Ω, 
∂_arrow(n) phi.alt_j &= 0 &&"on" ∂Ω,
)
$ <eq:ev-problem>

where $phi.alt_0, phi.alt_1, phi.alt_2, ...$ denote the corresponding eigenfunctions, normalized such that $(phi.alt_j)$ is an orthonormal basis of $L^2 (Omega)$. Rauch's hot spots conjecture is commonly stated as:

#conjecture[
  For "nice" enough domains $Ω$, the first eigenfunction $phi.alt_1$ attains its global extrema at the boundary $∂Ω$.
]<hs-conjecture>


In a 1975 lecture, J. Rauch @rauch_lecture_1975 introduced the conjecture as a problem in heat conduction: Suppose $Ω$ is some uniformly heat-conducting medium that is insulated on the boundary. Let $u(t, x)$ be the temperature at $x in Ω$ at time $t$, and suppose the initial temperature distribution is given by $u_0$. Rauch posed the question of where the function $u(t, dot)$ achieves its maximum as $t -> oo$. A standard separation of variables argument shows that  @hs-conjecture implies that _For generic initial conditions $u_0$, the points at which $u(t, dot)$ achieves its extrema accumulate on the boundary._


Indeed, the evolution of the temperature distribution in a uniform, insulated material can be modeled as

$
cases(
  ∂_t u &= ∆ u &&"in" (0, oo) times Ω,
  ∂_arrow(n) u &= 0 &&"on" (0,oo) times ∂Ω,
  u(0, x) &= u_0 (x) &&"for all" x in Omega.
)
$

The Neumann Laplacian is a self-adjoint non-positive operator in $L^2(Ω)$. The Neumann eigenfunctions of the Laplace operator in $Ω$, are a sequence of functions ${phi.alt_j}_(j >= 0)$ with eigenvalues $0 = lambda_0 <= lambda_1 <= dots <= lambda_j -> oo$, which satisfy
$
cases(
∆ phi.alt_j &= - lambda_j phi.alt_j &&"in"   Ω,
∂_arrow(n) phi.alt_j &= 0 &&"in"  ∂Ω.
)
$
The functions $phi.alt_j$ are an orthonormal basis of $L^2(Omega)$, and in particular one can write
$
u(0, x) = ∑_(j≥0) inner(u_0, phi.alt_j) phi.alt_j (x).
$

By linearity of the heat equation,  we can rewrite $u(x,t)$ as

$
u(t, x) = ∑_(j≥0) inner(u_0, phi.alt_j) phi.alt_j (x) e^(-lambda_j t),
$
where each of the functions $phi.alt_j (x) e^(-lambda_j t)$ is a solution of the heat equation. It satisfies the equation  
$(∂_t u - ∆) phi.alt_j (x) e^(-lambda_j t) = (lambda_j - lambda_j) phi.alt_j (x) e^(-lambda_j t) = 0$.

The first eigenfunction $phi.alt_0$ is the constant function, and in a connected domain, for all $i>= 1$ all the $lambda_i$ are positive. This shows that $u$ converges to the mean $overline(u) = 1/abs(Omega) ∫_Ω  u_0 dif x$ as $t -> oo$ in the $L^2(Ω)$ sense. Convergence in the $L^oo$ sense follows by showing that, under mild regularity conditions on $Ω$, the families ${u(t, dot)}_(t≥t_0)$ and ${e^(lambda_1 t)(u(t, dot) - overline(u))}_(t≥t_0)$ are uniformly bounded and equicontinuous, for some fixed $t_0 > 0$. Arzelà--Ascoli then upgrades each of the $L^2$ limits below to a uniform one. Therefore, assuming for now that $lambda_2 > lambda_1$, for large $t$, the "shape" of $u(t, dot)-overline(u)$ will be dominated by the first non-constant eigenfunction $phi.alt_(j_0)$ in the sequence $(phi.alt_j)$ for which $inner(u_0, phi.alt_(j_0)) ≠ 0$. We say the initial condition $u_0$ is _generic_ if $inner(u_0, phi.alt_1) ≠ 0$. Hence, under generic initial conditions, $tilde(u)(t, x) = e^(lambda_1 t)(u(t,x) - overline(u))$ achieves its extrema at the same points as $u$, and

$
tilde(u)(t, x) = inner(u_0, phi.alt_1) phi.alt_1 + ∑_(j≥2) inner(u_0, phi.alt_(j)) phi.alt_(j)(x) e^(-(lambda_j - lambda_1)t).
$

As $t -> oo$, the function $tilde(u)$ must converge uniformly and exponentially fast to the function $inner(u_0, phi.alt_1) phi.alt_1$, and in particular, the extremal points of $u$ converge subsequentially to the extremal points of $phi.alt_1$.



*Non-generic initial conditions.* The requirement that the initial condition be generic is essential. If we naively formulate the hot spots conjecture as stating that the point in $Ω$ at which $u(t, dot)$ achieves its maximum tends to the boundary as $t -> oo$ for any initial condition, it is straightforward to construct a counterexample. Suppose $Ω subset RR^2$ is the unit disk and $u_0$ is a radially symmetric heat distribution that achieves a strict maximum at the origin and is radially decreasing. By symmetry and monotonicity, $u(t, dot)$ will also achieve its maximum at the origin at any time $t$. Such an initial condition, however, is never generic. The first non-constant eigenfunction $phi.alt_1$ is antisymmetric, and therefore $inner(u_0, phi.alt_1) = 0$ (see @fig:disk-spectrum).

#figure(
  image("my_plot.png"),
  caption: [The first eigenfunctions of $-∆$ on a disk. Due to rotational symmetry, the eigenspaces of $phi.alt_1$ and $phi.alt_2$ are two-dimensional.]
) <fig:disk-spectrum>


*Multiple principal eigenfunctions.* The principal eigenspace can be two- or higher-dimensional; this is often the case if the domain $Ω$ is highly symmetric, such as a square or a disk. In such cases, there are several variations of the conjecture, depending on whether it should hold for all or only some eigenfunctions in the eigenspace. Following Bañuelos and Burdzy @banuelos_hot_1999, one can distinguish three formulations:

#enum(
  numbering: i => [(HS#i)],
  [For _every_ eigenfunction $phi.alt_1$ corresponding to $lambda_1$ which is not identically 0, and all $y in Ω$, we have $inf_(x in ∂Ω) phi.alt_1(x) < phi.alt_1(y) < sup_(x in ∂Ω) phi.alt_1(x)$.],

  [For _every_ eigenfunction $phi.alt_1$ corresponding to $lambda_1$ and all $y in Ω$, we have $inf_(x in ∂Ω) phi.alt_1(x) ≤ phi.alt_1(y) ≤ sup_(x in ∂Ω) phi.alt_1(x)$.],
  
  [There _exists_ an eigenfunction $phi.alt_1$ corresponding to $lambda_1$ which is not identically 0, and such that for all $y in Ω$, we have $inf_(x in ∂Ω) phi.alt_1(x) < phi.alt_1(y) < sup_(x in ∂Ω) phi.alt_1(x)$.]
)

In this work, we will build counterexamples with one-dimensional eigenspaces and eigenfunctions which exceed the boundary values in the interior, therefore disproving all three versions simultaneously. 

*What are nice enough domains?* The open question is identifying the classes of domains for which the conjecture holds and those for which it fails. A significant difficulty is that the principal eigenfunction of the Laplacian can be computed in closed form only for very few domains, such as balls, some special triangles or products of those. For example, on a rectangle $phi.alt_1$ is a cosine and the statement of the conjecture is easily verified. 

The original conjecture of Rauch understands _nice enough_ to mean domains with enough regularity to make sense of the spectrum of the Laplace operator and Neumann boundary condition #footnote[Smooth, or Lipschitz would suffice, and be equivalent for most scenarios: A positive resolution of the conjecture for smooth domains where $lambda_1 < lambda_2$ would imply the conjecture for Lipschitz domains with the same property.]

Kawohl @kawohl_rearrangements_1985[Corollary 2.15] showed that (HS1) holds for cylindrical domains $Ω = Ω_0 times (0,l) subset RR^d$ where $Ω_0$ is bounded with $∂Ω_0$ of class $C^(0,1)$. Furthermore, Kawohl conjectured that the property holds for all convex domains @kawohl_rearrangements_1985[p.56].
Bañuelos and Burdzy @banuelos_hot_1999 later proved that (HS1) holds for certain triangles and for long convex planar domains with sufficient symmetry. Conversely, Burdzy and Werner @burdzy_counterexample_1999 constructed a counterexample consisting of a bounded planar domain with two holes; This was later reduced to a single hole @burdzy_hot_2004. For numerical counterexamples, see @kleefeld_hot_2021.

The conjecture was believed to be true for convex sets until it was recently disproven @de_dios_convex_2024 in high dimensions. This recent work is the subject of the next subsection.


// #inline-note-j[Maybe introducing the reader to the Dirichlet energy _before_ this section may be good? I.e. explain why the first eigenfunction is special.]

// #inline-note-a[
//   How big of a priority is it? I think that it is the minimizer of the Rayleigh quotient is well known.
// ]

// #inline-note-j[I mean it's not bad to remind the reader. You don't have to prove it just state it.]

== Convex sets can have interior hot spots

The key to the counterexample constructed in @de_dios_convex_2024 is to first pose the log-concave extension of the hot spots conjecture. The eigenfunctions of the Neumann Laplacian on $Omega$ are critical points of the Rayleigh quotient $integral_Omega abs(nabla phi.alt)^2 dif x \/ integral_Omega abs(phi.alt)^2 dif x$. The log-concave analogue is obtained by generalizing the Lebesgue measure on $Omega$ to log-concave measures. That is, to measures $mu$ with density $dif mu(x) = e^(-V (x)) dif x$ for some convex function $V$ on $Omega$. The principal eigenfunction $phi.alt_(1, mu)$ is now the minimizer of 
$
(integral_Omega abs(nabla phi.alt)^2 dif mu) / (integral_Omega abs(phi.alt)^2 dif mu),
$

over the all $mu$-mean-zero functions. The log-concave extension of the hot spots conjecture states that $phi.alt_(1, mu)$ attains its maximum on $∂ Omega$, for any log-concave measure $mu$. The argument of of @de_dios_convex_2024 is to first this version of the conjecture and then transfer the counterexample by approximating $mu$ by a uniform measure on a high-dimensional _barrel set_. Given a rectangle $Q subset RR^2$ and a potential $V : Q -> RR$, the _barrel set_ is defined as

$ F_d (Q, V) := {(x,w) in Q times RR^(d+1) : abs(w) <= sqrt(d)/2 (1 - V(x)/d)}, $

which is convex in $RR^(2+d+1)$. The diffeomorphism

$ Phi : F_d (Q, V) -> Q times B_(sqrt(d)/2), quad (x,w) |-> (x, (1 - V(x)/d)^(-1) w) $

pushes the uniform measure on $F_d (Q, V)$ forward to a measure that approximates, as $d -> oo$, the log-concave measure $dif mu = e^(-V(x)) dif x dif w$ on the product domain $Omega = Q times B_(sqrt(d)/2)$, where $V$ is chosen so that $mu$ is a counterexample to the log-concave conjecture. Consequently, the principal eigenfunctions satisfy $norm(phi.alt_(1, d) compose Phi^(-1) - phi.alt_(1, mu))_(L^2(mu)) -> 0$ as $d -> oo$. This can be upgraded to uniform convergence using Wang--Li--Yau ultracontractivity estimates @rockner_supercontractivity_2003. Since $phi.alt_(1, mu)$ attains a strict interior maximum, it follows that for all sufficiently large $d$, the principal eigenfunction $phi.alt_(1, d)$ of the Neumann Laplacian on the convex domain $F_d (Q, V)$ must also attain a strict interior maximum. Since the principal eigenvalue of $F_d (Q, V)$ is simple, $phi.alt_(1, d)$ is, up to sign, the unique principal eigenfunction, so this violates (HS1)--(HS3) simultaneously. The threshold dimension $d_"HS"$ above which the hot spots property fails is not made explicit in @de_dios_convex_2024, and a naive tracking of constants does not yield a satisfactory quantitative bound. The primary goal of this work is to provide an explicit upper bound on $d_"HS"$.

== The method of particular solutions and certified numerics

We draw inspiration from @dahne_counterexample_2021, where the authors construct a numerical counterexample to Payne's nodal line conjecture using the method of particular solutions and rigorously certify it. The idea behind the method is simple. We explain it in the context of our eigenvalue problem, but it transfers to any boundary value problem: choose a set of functions ${f_j}_j$ that satisfy the eigenvalue equation $-∆ f_j = lambda_* f_j$ in the interior, disregarding the boundary conditions. Any linear combination $phi.alt_* = ∑_j c_j f_j$ then also satisfies $-∆ phi.alt_* = lambda_* phi.alt_*$. To obtain an approximation of the true eigenfunction $phi.alt_1$, we optimize the coefficients ${c_j}$ and the approximate eigenvalue $lambda_*$ so as to minimize the error on the boundary.

Once we have obtained a candidate approximation $phi.alt_*$, we prove that if the boundary error $norm(partial_arrow(n) phi.alt_*)_(L^oo (∂Ω))$ is small, then $phi.alt_*$ must be pointwise close to the true eigenfunction $phi.alt_1$. One limitation of the method of particular solutions is that it does not control where in the spectrum the approximation $phi.alt_*$ lies; that is, $phi.alt_*$ could have a small boundary error yet approximate the second eigenfunction $phi.alt_2$ rather than $phi.alt_1$. This is particularly problematic in settings such as that of @dahne_counterexample_2021, where the eigenvalue of interest lies in a cluster. Fortunately, in our case the spectral gap $lambda_2 - lambda_1 gt.tilde #lower_bound_spectral_gap$ is large enough that we can separate $lambda_1$ from $lambda_2$ using a finite element method @liu_guaranteed_2024. The main difficulty compared to @dahne_counterexample_2021 -- apart from the higher ambient dimension -- is that we work with Neumann rather than Dirichlet boundary conditions. For the Dirichlet case, @moler_bounds_1968 describes how to bound the approximation error of the eigenfunction in terms of the boundary error: one introduces a harmonic correction term $w$ that matches $phi.alt_*$ on the boundary. The function $phi.alt_* - w$ then satisfies the Dirichlet boundary condition exactly, at the cost of an error in the interior equation, $-Delta (phi.alt_* - w) = lambda_* (phi.alt_* - w) + lambda_* w$. The argument then relies on $w$ being small. In the Dirichlet case, $norm(w)_(L^oo)$ is bounded immediately by the boundary error via the maximum principle. In the Neumann case, $w$ also attains its maximum on the boundary, but we have no direct control over its boundary values, since we only control the normal derivative of $phi.alt_*$. Bounding $norm(w)_(L^oo)$ therefore requires significantly more effort. We propose a novel approach that constructs a supersolution for $w$ computationally. So far, however, we have not been able to find such a supersolution to sufficient precision.

Since $phi.alt_*$ and $norm(partial_arrow(n) phi.alt_*)_(L^oo (∂Ω))$ are evaluated numerically, they are subject to floating-point arithmetic errors. To obtain a truly rigorous counterexample, @dahne_counterexample_2021 employ interval arithmetic to compute rigorous enclosures of the boundary error and the eigenfunction. The counterexamples in this work are implemented in floating-point arithmetic only, for two reasons: the approximation error must first be brought down further, and computing error enclosures over the effectively two-dimensional boundary is computationally expensive.

== Results and contributions

Our first contribution is a modification of the potential $V$ of @de_dios_convex_2024 that is explicitly computable and implemented. Building on it, we implement the method of particular solutions on the barrel set $F_d (Q, V)$ and obtain an approximate principal eigenfunction that attains its maximum in the interior, thereby visualizing the mechanism behind the counterexample of @de_dios_convex_2024. Our approximation scheme is parametrized by the dimension $d$, and in the limit $d -> oo$ it reduces to an approximation of the log-concave extension. We derive pointwise bounds on the eigenfunction: for the log-concave extension, that is, in the limit $d -> oo$, these bounds can be computed explicitly, while in the finite-dimensional case they rely on a numerical supersolution. Finally, we rigorously separate the first two eigenvalues $lambda_1$ and $lambda_2$ by means of a finite element method @liu_guaranteed_2024. Our main numerical finding is made precise in the theorem below: for $d = #optimistic_finite_d$ it gives a candidate convex counterexample in dimension $d + 3$, and hence gives the numerical---not yet certified---bound $d_"HS" <= d + 3$.

#theorem[
  Let $Q = [-2pi, 2pi] times [-1, 1] subset RR^2$ and let $d = #optimistic_finite_d$, so that the barrel $F_d (Q, V)$ is a convex set in $RR^(d + 3)$. There exist an explicit smooth convex potential $V$ on $Q$ and a function $phi.alt_*$ on $F_d (Q, V)$ that satisfies the interior eigenvalue equation exactly, $-Delta phi.alt_* = lambda_* phi.alt_*$, with small boundary error
  $
    norm(partial_arrow(n) phi.alt_*)_(L^oo (∂ F_d (Q, V))) lt.tilde #boundary_error_finite_d,
  $
  and which attains its maximum in the interior:
  $
    max_(F_d (Q, V)) phi.alt_* - max_(∂ F_d (Q, V)) phi.alt_* gt.tilde #hot_spot_effect_finite_d .
  $
  The approximate relations $lt.tilde$ and $gt.tilde$ account for floating-point arithmetic and for the boundary error being estimated by sampling on a uniform $1024 times 1024$ grid in $Q$.
]<thm:main-finite-dim-result>

#remark[
  For the log-concave extension, that is, the formal limit $d -> oo$, which we denote $F_oo (Q, V)$, the same construction yields the boundary error
  $
    norm(partial_arrow(n) phi.alt_*)_(L^oo (∂ F_oo (Q, V))) lt.tilde #boundary_error_limit
  $
  and the interior maximum
  $
    max_(F_oo (Q, V)) phi.alt_* - max_(∂ F_oo (Q, V)) phi.alt_* gt.tilde #hot_spot_effect_limit .
  $
  In this case we can moreover bound the pointwise distance to the true principal eigenfunction $phi.alt_1$,
  $
    norm(phi.alt_* - phi.alt_1)_(L^oo (F_oo (Q, V))) lt.tilde #error_pointwise_limit ,
  $
  which still exceeds the hot-spot effect and is therefore insufficient to certify the interior maximum.
]

We do not yet obtain a certified counterexample. Even for the log-concave extension, the $L^oo$ boundary error of our approximation remains too large relative to the small hot-spot effect to certify the latter. We expect this can be improved by enlarging the approximation basis and the computational budget, by better sampling strategies, or by fine-tuning the potential to increase the hot-spot effect. In the finite-dimensional case, our pointwise bounds depend on a supersolution that we do not currently know how to compute. It is conceivable that this is not the right route, and that one should instead invoke the existing $L^2$ eigenfunction bounds for the Neumann problem and upgrade them directly to $L^oo$ via ultracontractivity. A further obstruction is that for the approximate eigenfunction to exhibit the desired interior maximum, the potential $V$ must be very large at the ends of the barrel, of order $10^6$, and the ultracontractivity estimates that pass from $L^2$ to pointwise bounds then require $d$ of order $norm(V)_(L^oo)^2 tilde 10^12$. We nonetheless observe the effect at far smaller $d$, which suggests that the limitation lies in these estimates rather than in the dynamics itself. Finally, all of our computations are carried out in floating-point arithmetic; we do not yet employ interval arithmetic.

== Outline of the thesis

This thesis is structured as follows: In @barrels we study barrel sets in detail. We explain how barrel sets approximate the log-concave problem in high dimensions. We show that on barrel sets the principal eigenfunction $phi.alt_1$ is highly symmetric, allowing us to reduce the effective dimension of the set giving the counterexample to three. On this effectively three dimensional set we construct a basis for the method of particular solutions. \
In @construction we explain the construction of the potential $V$ inducing a counterexample log-concave measure in @de_dios_convex_2024 and derive an adaptation which can be explicitly computed. \
In @numerics we present our numerical implementation of our method of particular solutions and the resulting approximation. Since the method of particular solutions cannot guarantee the position in the spectrum of the approximate eigenpairs, we derive a priori bounds in @sec:eigenvalues. \
In @sec:pointwise we derive pointwise bounds for the eigenfunction based on the eigenvalue bounds and the MPS-residual. \
Finally, 
//in @certificate 
we present the resulting numerical bounds and discuss improvements needed for a certificate for the failure of the hot spots conjecture. 
