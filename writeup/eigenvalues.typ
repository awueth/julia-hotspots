#import "template.typ": *

= Bounds for the eigenvalues <sec:eigenvalues>

To locate the eigenvalues within the spectrum we need rigorous enclosures. We compute such enclosures only for the limiting problem at $d = oo$, that is, for the Neumann eigenvalues of the operator $L = -∆ + nabla V dot nabla$ on the base $Q$, which is self-adjoint with respect to the measure $dif mu = Z^(-1) e^(-V) dif x$. The corresponding bounds at finite dimension then follow from the convergence rate of @thm:eigenvalue-convergence. We write
$
a(u, v) = integral_Q nabla u dot nabla v dif mu, quad R(u) = a(u, u) / norm(u)_(L^2(mu))^2
$
for the Dirichlet form of $L$ and its Rayleigh quotient, and denote by $0 = lambda_0 < lambda_1 ≤ lambda_2 ≤ ...$ the Neumann eigenvalues of $L$, with $L^2(mu)$-orthonormal eigenfunctions $phi.alt_0 ≡ 1, phi.alt_1, phi.alt_2, ...$. An upper bound is only needed for the first eigenvalue $lambda_0$, it will come directly from the Rayleigh quotient of our candidate eigenfunction, lower bounds of $lambda_1$ and $lambda_2$ rely on the guaranteed finite element bounds of @liu_guaranteed_2024.

== Upper bounds

By the min-max principle, testing against the candidate antisymmetric eigenfunction $phi.alt_1^*$ computed in @numerics gives
$
lambda_1 ≤ R(phi.alt.alt_1).
$

#proposition[
  Evaluating the Rayleigh quotient
  $
  lambda_1 ≤ 4.0.
  $
]<prop:upper-bounds>

== Lower bounds

Lower bounds require more effort. We use the guaranteed lower bounds of the nonconforming Crouzeix–Raviart finite element method.

#inline-note-a[
  We should note that we assume that the eigenvalues of the discretized problem are computed exactly. Bounding this error as well is out of scope.
]

#theorem([@liu_guaranteed_2024, Theorem 3.2])[
  Suppose the interpolation operator $Pi_h$ satisfies $integral abs(u - Pi_h u)^2 dif x ≤ C_h^2 integral abs(nabla(u - Pi_h u))^2 dif x$ for all $u in H^1$. Then the discrete eigenvalues $lambda_(k, h)$ yield the lower bound
  $
  lambda_(k, h) / (1 + lambda_(k, h) C_h^2) ≤ lambda_k.
  $
]<thm:guaranteed-lower>

The method relies on an explicit interpolation constant $C_h$. For the weighted eigenproblem on $Q$ this constant picks up a factor measuring the oscillation of the potential across each mesh cell.

#lemma[
  Let $cal(T)_h$ be a triangulation of $Q$ and let $op("osc")_K V = max_(x in K) V(x) - min_(x in K) V(x)$ for any simplex $K in cal(T)_h$. For the operator $L$ with weight $e^(-V)$ the interpolation constant satisfies
  $ C_h ≤ max_(K in cal(T)_h) e^(op("osc")_K V slash 2) C^"CR" (K), quad C^"CR" (K) ≤ 0.1893 thin h_K, $
  where $h_K = op("diam") K$.
]<lem:interp-constant>
#proof[
  On a simplex $K$ the Crouzeix–Raviart interpolant $Pi^"CR"$ we have the  the unweighted estimate $norm(u - Pi^"CR" u)_K ≤ C^"CR"(K) norm(nabla(u - Pi^"CR" u))_K$ with $C^"CR"(K) ≤ 0.1893 thin h_K$ @liu_guaranteed_2024. Bounding the weight $e^(-V)$ from above by $e^(-min_(x in K) V)$ in the $L^2$ term and from below by $e^(-max_(x in K) V)$ in the gradient term,
  $
  integral_K abs(u - Pi_h u)^2 e^(-V) dif x
  &≤ e^(-min_(x in K) V) integral_K abs(u - Pi_h u)^2 dif x \
  &≤ e^(-min_(x in K) V) C^"CR" (K)^2 integral_K abs(nabla(u - Pi_h u))^2 dif x \
  &≤ e^(op("osc")_K V) C^"CR" (K)^2 integral_K abs(nabla(u - Pi_h u))^2 e^(-V) dif x.
  $
  Summing over the cells $K in cal(T)_h$ and retaining the worst one gives the claim.
]

The main bottleneck is $e^(op("osc")_K V slash 2)$. In the core $Q_"core"$ the potential is moderate and this factor can be controled by refining the mesh. In the wing $Q_"wing"$, $V$ varies by orders of magnitude over short distances, requiring too fine of a mesh. Furthermore, the weighted measure underflows in $Q_"wing"$.  We therefore never triangulate the wing. We bound the global eigenvalues below by the eigenvalues of $L$ on the core alone, at the cost of an exponentially small correction, since the wing carries a negligible fraction of the mass $mu$.

=== Reduction to the core

Write $lambda_k^"core"$ for the $k$-th nonzero Neumann eigenvalue of $L$ on $Q_"core"$, and recall that $mu(Q) = 1$ and $norm(phi.alt_k)_(L^2(mu)) = 1$. The reduction rests on two ingredients: the smallness of the wing mass $mu(Q_"wing")$, and a uniform bound on the eigenfunctions through the ultracontractivity estimate of @thm:ultracontractivity,
$
norm(phi.alt_k)_oo ≤ e^(lambda_k t) C_t, quad t > 0,
$
where $C_t$ is a constant depending on $t$, derived in @sec:ultracontractivity.

We begin with $lambda_1$. By @lem:parity the principal eigenfunction $phi.alt_1$ is odd in $x_1$, hence has vanishing $mu$-mean on the core. It is therefore admissible in the variational characterization of $lambda_1^"core"$, so $R_(Q_"core")(phi.alt_1) ≥ lambda_1^"core"$. Discarding the nonnegative Dirichlet energy on the wing,
$
lambda_1 = integral_Q abs(nabla phi.alt_1)^2 dif mu
≥ integral_(Q_"core") abs(nabla phi.alt_1)^2 dif mu
= R_(Q_"core")(phi.alt_1) integral_(Q_"core") abs(phi.alt_1)^2 dif mu
≥ lambda_1^"core" integral_(Q_"core") abs(phi.alt_1)^2 dif mu.
$
The core mass differs from one only by the wing contribution,
$
integral_(Q_"core") abs(phi.alt_1)^2 dif mu = 1 - integral_(Q_"wing") abs(phi.alt_1)^2 dif mu ≥ 1 - norm(phi.alt_1)_oo^2 thin mu(Q_"wing"),
$
which, combined with the ultracontractivity bound, gives the following.

#lemma[
  For every $t > 0$ and every upper bound $overline(lambda_1) ≥ lambda_1$,
  $
  lambda_1 ≥ lambda_1^"core" (1 - e^(2 overline(lambda_1) t) C_t^2 thin mu(Q_"wing")).
  $
]<lem:lambda1-core>

For $lambda_2$ we test the second core eigenvalue against the two-dimensional trial space spanned by the first two global eigenfunctions, $S = op("span"){phi.alt_1, phi.alt_2}$. By the min-max principle, for $h in S$ with zero mean on the core,
$
lambda_2^"core"
&≤ max_(1 perp h in S) (integral_(Q_"core") abs(nabla h)^2 dif mu) / (integral_(Q_"core") abs(h)^2 dif mu) \
&≤ max_(1 perp h in S) (integral_Q abs(nabla h)^2 dif mu) / (integral_Q abs(h)^2 dif mu) dot.c (integral_Q abs(h)^2 dif mu) / (integral_(Q_"core") abs(h)^2 dif mu) \
&≤ lambda_2 max_(1 perp h in S) (integral_Q abs(h)^2 dif mu) / (integral_(Q_"core") abs(h)^2 dif mu),
$
since the first factor is at most $lambda_2$ because $h$ lies in the span of the first two eigenfunctions. For the remaining ratio, write $h = a phi.alt_1 + b phi.alt_2$ with $a^2 + b^2 = norm(h)_(L^2(mu))^2$. Cauchy–Schwarz gives the pointwise bound $abs(h)^2 ≤ (a^2 + b^2)(norm(phi.alt_1)_oo^2 + norm(phi.alt_2)_oo^2)$, so $integral_(Q_"wing") abs(h)^2 dif mu ≤ (norm(phi.alt_1)_oo^2 + norm(phi.alt_2)_oo^2) mu(Q_"wing") norm(h)_(L^2(mu))^2$ and hence
$
(integral_Q abs(h)^2 dif mu) / (integral_(Q_"core") abs(h)^2 dif mu) ≤ 1 / (1 - (norm(phi.alt_1)_oo^2 + norm(phi.alt_2)_oo^2) mu(Q_"wing")).
$
Bounding both eigenfunction norms by ultracontractivity yields the analogue of the previous lemma.

#lemma[
  For every $t > 0$ and every upper bound $overline(lambda_2) ≥ lambda_2$,
  $
  lambda_2 ≥ lambda_2^"core" (1 - 2 e^(2 overline(lambda_2) t) C_t^2 thin mu(Q_"wing")).
  $
]<lem:lambda2-core>

=== Certified enclosures

#let lambda1_core_lower = 0
#let lambda2_core_lower = 0

Applying the Crouzeix–Raviart method on the core with Gridap.jl, on a mesh fine enough that @lem:interp-constant keeps the constant $C_h$ small, and inserting the discrete eigenvalues into @thm:guaranteed-lower, we obtain
$
lambda_(1, h)^"core" ≥ #lambda1_core_lower, quad lambda_(2, h)^"core" ≥ #lambda2_core_lower.
$
With $mu(Q_"wing") ≤ 10^(-10)$ from @lem:wing-mass, the upper bounds $overline(lambda_1) = 4.0$ and $overline(lambda_2) = 10.0$ from @prop:upper-bounds, and the smoothing constant $C_t$ at the chosen time $t$, the correction factors of @lem:lambda1-core and @lem:lambda2-core, and combining them with @prop:upper-bounds gives the enclosures
$
lambda_1 in [#lambda1_core_lower, 4.0], quad lambda_2 in [#lambda2_core_lower, 10.0].
$
In particular $lambda_1 < lambda_2$: the principal eigenvalue is simple and is separated from the rest of the spectrum by a gap $lambda_2 - lambda_1 ≥ 0$.
