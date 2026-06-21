#import "template.typ": *

= Constructing a counterexample set <construction>

In this section we explain the counterexample from @pont_convex_2024 and how we adapted it into a version that can be computed explicitly. Since our proof relies on verified numerics, we do not establish any properties of the counterexample set rigorously here; instead we give the intuition for why we expect it to work. Recall from @eq:limit-problem that the spatial variable $x$ ranges over the base $Q$ of the Barrel, while $r in [0, 1]$ is the radial coordinate, and $psi_1(x) = phi.alt_1(x, r=1)$. Following @pont_convex_2024, we perform the change of variables $t = (1-r^2) slash 8$, which sends the shell $r=1$ to $t=0$ and the axis $r=0$ to $t=1\/8$. Since $partial_r = -r\/4 thin partial_t$, this turns the boundary value problem @eq:limit-problem into a reaction-diffusion initial value problem:

$
∂_t h &= ∆_x h+ lambda_1 h "for" t in (0, 1\/8]\
h(x, 0) &= psi_1(x) \
∂_arrow(n) h &= 0 "on the spatial boundary",
$<eq:limit-ivp>

where $h(x, t) = phi.alt_1(x, r)$ is simply the Barrel eigenfunction in the new coordinates. The initial condition at $t=0$ corresponds to the values of $h$ on the shell of the Barrel, and an interior maximum of $h$ yields the interior maximum of $phi.alt_1$ that we are after. In terms of the MPS basis of @mps_basis, separating variables gives

$
h(x, t) = sum_(j,k) c_(j,k) X_(j, k) (x) exp(lambda_(r, j, k) t),
$<eq:limit-expansion>

where the $X_(j,k)$ are the Neumann eigenfunctions of $-∆_x$ on $Q$ with eigenvalues $lambda_(x,j,k)$, and $lambda_(r,j,k) = lambda_1 - lambda_(x,j,k)$; indeed, under $t = (1-r^2)\/8$ the $d=oo$ radial factor $exp(- lambda_(r,j,k)\/8 thin (r^2-1))$ of @mps_basis is exactly $exp(lambda_(r,j,k) t)$. Note that the reaction term shifts every exponent by $lambda_1$: the modes with $lambda_(x,j,k) < lambda_1$ have $lambda_(r,j,k) > 0$ and therefore _grow_ as $t -> 1\/8$, rather than only decaying slowest as they would under pure diffusion, which is what makes an interior maximum at the axis possible.

To show that the global maximum of $h$ is attained in the interior of the Barrel, it suffices to establish two claims: a _temporal_ claim, that $max_(0 ≤ t ≤ 1\/8) max_x h(x,t)$ is attained at $t=1\/8$, and a _spatial_ claim, that $max_(x in Q^circle) h(x, 1\/8) > max_(x in ∂Q) h(x,1\/8)$. The following lemma gives a numerically verifiable sufficient condition for the temporal claim.

#lemma[
  If $max_x h(x, T) ≥ e^(lambda_1 T) max_x h(x, 0)$, then $max_(0 ≤ t ≤ T) max_x h(x,t) = max_x h(x,T)$.
]
#proof[
  Let $u(x,t) := e^(-lambda_1 t) h(x,t)$, so that $∂_t u = Delta u$ with no-flux boundary conditions. By the parabolic maximum principle the spatial maximum is non-increasing in time, $max_x u(x, t) ≤ max_x u(x, 0)$ for $t ≥ 0$. Equivalently, since $u(dot,0)=h(dot, t)$,

  $
  max_x h(x,t) ≤ e^(lambda_1 t) max_x h(x, 0).
  $

  Hence for every $t ≤ T$, using $lambda_1 > 0$ and the assumption,
  
  $
  max_x h(x,t) ≤ e^(lambda_1 T) max_x h(x,0) ≤ max_x h(x, T).
  $
]


We now describe how we engineer the initial datum $psi_1$ so that the flow @eq:limit-ivp pushes its maximum off the boundary and into the interior. This gives the intuition for the spatial claim above.

We split the base $Q$ of the barrel into two parts: the _core_ $Q_"core" = [-pi/2, pi/2] times [-1, 1]$ and the _wing_ $Q_"wing" = ([-2pi, 2 pi] times [-1, 1]) without Q_"core"$. On a macroscopic scale we want $psi_1$ to look like $sin(x_1)$ in $Q_"core"$, that is, like the principal eigenfunction of $-∆$ on $Q_"core"$, and to be a constant extension of this profile in $Q_"wing"$. On a microscopic scale we perturb $psi_1$ so that at the core-wing interface $x_1 = pi/2$ we have $psi_1(x) approx q(x_2) + "const."$, where $q$ is the solid profile of @fig:q. The profile $q$ has a global maximum at $x_2 = 0$ and two secondary maxima at $x_2 = plus.minus 1$. We transport this profile into the wing in three stages along $x_1$. Just past the interface there is a _plateau_ on which $psi_1$ is, approximately, independent of $x_1$ and equal to $q(x_2) + "const."$. Then comes a _transition region_ in which we _trim_ the profile, cutting off the secondary peaks at $x_2 = plus.minus 1$ (the dashed profile of @fig:q). Finally, near the wing end there is a second plateau on which $psi_1$ is again independent of $x_1$ and equal to the trimmed profile $tilde(q)(x_2) + "const."$. As a result the initial datum $psi_1$ attains its maximum at $x = (2pi, 0)$, the end of the wing, which lies on the boundary.

#figure(
  lq.diagram(
    let ys = lq.linspace(-1.0, 1.0),
    //let q(y) = 0.5*calc.cos(calc.pi*y) + 0.3*calc.cos(2.0*calc.pi*y) - 0.2*calc.cos(3.0*calc.pi*y),
    let q(y) = 0.2*calc.cos(calc.pi*y) + 1.0*calc.cos(2.0*calc.pi*y) - 0.0*calc.cos(3.0*calc.pi*y),
    let q_trimmed(y) = if calc.abs(y) < 0.55 {q(y)} else {q(0.55)},
    lq.plot(ys, q, color: blue, mark: none, label: [$q(x_2)$]),
    lq.plot(ys, q_trimmed, color: red, mark: none, label: [$tilde(q)(x_2)$], stroke: (dash: "dashed")),
  ), 
  caption: [Idealized target cross section of $psi_1 - "const."$ in $x_2$. Solid: the target profile $q$ at the core-wing interface $x_1 = pi/2$. Dashed: the trimmed profile $tilde(q)$ at the end of the wing, with the secondary peaks at $x_2 = plus.minus 1$ cut off.]
)<fig:q>

To see why the maximum diffuses inward, ignore $x_1$ and consider the pure heat flow in $x_2$ of the interface profile $q$ and of its trimmed version $tilde(q)$. Both stay maximized at $x_2 = 0$ for all times, since they are even with a central peak. However, as time increases the flow of $q$ strictly dominates that of $tilde(q)$ at $x_2 = 0$: the untrimmed profile carries more mass in the tails near $x_2 = plus.minus 1$, and diffusion transports this mass toward the center. Thus the interface cross section ends up taller at its center than the trimmed wing-end cross section.

The reason this one-dimensional picture approximates the two dimensional dynamics is the plateau structure described above. On each of the two plateaus we have $∂_(x_1) psi_1 approx 0$, so the $x_1$-diffusion $∂_(x_1)^2 h$ is negligible and the flow @eq:limit-ivp decouples there into an independent $x_2$-heat flow, behaving like its one-dimensional counterpart from the previous paragraph. In the transition region between the plateaus, and right at the interface $x_1 = pi/2$, $psi_1$ does vary in $x_1$ and $∂_(x_1)^2 h$ is not negligible. But the comparison only needs that some point in the interior plateau dominates the wing-end plateau.

#subpar.grid(
  figure(image("surface_plot_eigenfunction_boundary.png", width: 100%), caption: [$h(x,t=0)=phi.alt_1 (x,r=1) = psi_1 (x)$]),<a>,
  figure(image("surface_plot_eigenfunction_interior.png", width: 100%), caption: [$h(x,t=1 slash 8)=phi.alt_1 (x,r=0)$]),<b>,
  figure(image("heatmap_eigenfunction_boundary.png", width: 100%), caption: [Heatmap of the green region in (a), at $t=0$. The two bands of nearly constant colour are the plateaus on which $psi_1$ is independent of $x_1$.]),<c>,
  figure(image("heatmap_eigenfunction_interior.png", width: 100%), caption: [Heatmap of the same region, at $t=1 slash 8$. #linebreak() #linebreak() #linebreak()]),<d>,
  columns:(1fr, 1fr),
  v(0%),
  caption: [The computed Barrel eigenfunction $h$ in the reaction-diffusion coordinates of @eq:limit-ivp. Top row: the global eigenfunction at $t=0$ (a), where the maximum sits at the wing end on the boundary, and at $t=1\/8$ (b), where the diffusion has pushed the maximum off the boundary and into the interior. Bottom row: heatmaps of the wing subset marked in green above, at $t=0$ (c) and $t=1\/8$ (d). We show only this subset rather than the full wing because the profile effect driving the maximum inward is small. Including the full wing region, would let the diffusion into the core dominate the colour scale, making the effect invisible.]
)<fig:initial-datum>

== Construction of the core potential

In order to build a potential $V$ such that $psi_1$ has the before-mentioned properties in $Q_"core"$, we consider the operator $L = -∆ + nabla V dot nabla$ to be a perturbation of the Laplacian $-∆$. Let $V_epsilon = epsilon^(-1) V$, then we can write $L$ as

$
L = -∆ + epsilon nabla V_epsilon dot nabla.
$

Expanding both $lambda_1$ and $psi_1$ in powers of $epsilon$,

$
lambda_1 &= tilde(lambda)_1 + epsilon mu + O(epsilon^2) \
psi_1 &= tilde(psi)_1 + epsilon beta + O(epsilon^2).
$

Substituting the expansions into $L psi_1 = lambda_1 psi_1$ and collecting terms of the same order in $epsilon$ we get

$
tilde(psi)_1 &= sqrt(pi/2) sin(x_1), quad tilde(lambda)_1 = 1 \
(-∆-1) beta &= sqrt(pi/2) (mu sin(x_1) - (∂_1 V_epsilon) cos(x_1)).
$ <eq:perturbation-terms>

We already obtain the correct macroscopic behavior of $psi_1$, given that $epsilon$ is small enough. The rest of the desired properties will be enabled by $beta$. Suppose we have a function $beta_0$ defined on $Q_"core"$ that is

+ odd in $x_1$ and even in $x_2,$
+ has Neumann boundary data and additionally $beta_0(pi\/2,x_2) = q(x_2) + "const."$, and 
+ satisfies $(∆+1) beta_0(pi\/2, x_2) = 0$.

Then, the possibly non-convex potential

$
V_0 (x_1,x_2) = -sqrt(2/pi) ∫_0^x_1 ((∆ + 1) beta_0 (s,x_2)) / cos(s) dif s,
$

satisfies

$
(-∆-1) beta_0 &= -sqrt(pi/2) (∂_1 V_0) cos(x_1).
$

To obtain a convex potential, we add a quadratic term $1/2 (M_1 x_1^2 + M_2 x_2^2)$ to $V_0(x_1, x_2)$, that gives us the potential $V(x_1, x_2) = V_0 (x_1, x_2) + 1/2 (M_1 x_1^2 + M_2 x_2^2)$. The Hessian of $V$ is therefore

$
H_V (x) 
= H_(V_0) + H_V
=
mat(
  a(x), c(x);
  c(x), b(x)
) 
+ mat(
  M_1, 0;
  0, M_2
).
$

For $H_V$ to be PSD, it is sufficient that $det H ≥ 0$ and $tr H ≥ 0$, so we choose $M_1, M_2$ such that $M_1 + M_2$ is minimized under the constraints

$
forall x : a + M_1 ≥ 0, b + M_2 ≥ 0, (a + M_1)(b + M_2) - c^2 ≥ 0.
$

In order to obtain the full $beta$ satisfying @eq:perturbation-terms, let $s(x_1), mu$ be a solution of the ODE $-∂_1^2 s(x_1) - s(x_1) = sqrt(pi/2) (mu/M sin(x_1) - x_1 cos(x_1))$ on $[-pi\/2, pi/2]$ with Neumann boundary conditions. Then, $beta(x_1, x_2) = beta_0 (x_1, x_2) + M s(x_1)$ satisfies

$
(-∆-1) beta(x_1,x_2)
&= sqrt(pi/2)(-(∂_1 V_0) cos(x_1) + M(mu/M sin(x_1) - x_1 cos(x_1))) \
&= sqrt(pi/2)(mu sin(x_1) - (∂_1 V) cos(x_1)),
$

as desired. In practice we never compute $s$ since it does not affect the interface profile and we only are interested in obtaining a potential.


*Constructing $beta_0$* We choose a target boundary profile $q(x_2) = sum_(n=1)^N q_n cos(n pi x_2)$. We construct $beta_0$ as a Chebyshev-Fourier expansion:
  
$
beta_0 (x_1,x_2) = f_0(x_1) + sum_(n=1)^N q_n f_n (x_1) cos(n pi x_2)
$

where $f_n (x_1) = sum_(j=0)^(J-1) a_(n,j) T_(2j + 1) (2x_1 \/ pi)$. The symmetry constraints on $beta$ are automatically satisfied. By imposing the constraint $sum_j a_(n,j) = 1$ for all $n≥1$ we obtain

$
beta_0 (pi/2, x_2) - f_0 (pi/2) 
&=  sum_(n=1)^N q_n (sum_j a_(n,j) T_(2j + 1) (1)) cos(n pi x_2) \
&= sum_n q_n cos(n pi x_2)
=  q(x_2).
$

From the Neumann boundary condition $f'_n (pi/2) &= 0$, we get the constraint 

$
sum_j a_(n, j) (2j + 1)^2 = n^2 pi^2 - 1.
$

Finally, $(∆+1) beta_0(pi\/2, x_2) = 0$ is imposed by 

$
f''_n (pi/2) &= (n^2 pi^2 - 1) f_n (pi/2) => sum_j a_(n,j) (2j+1)^2 ((2j+1)^2-1) = (n^2 pi^2 - 1) sum_j a_(n,j).
$

It is possible to satisfy the three constraints with $J=3$. However, we choose $J>3$, in order to have some degrees of freedom to minimize the convexification cost $M$. The zeroth mode $f_0$ has the same constraints except $f_0 (pi/2) = 1$, but instead $inner(f_0,sin(x_1)) = 0$.

== Extending to the wings

In the previous section we built $V_0$ on the core so that the Neumann eigenfunction $psi_1$ has the prescribed interface profile $q$. We now extend $V$ to the wing so that two things hold: gluing the wing leaves the core eigenfunction essentially unchanged, and the interface profile is carried into the wing as the plateau-trim-plateau structure used in our heuristic for the spatial claim. We get there in three steps. First, any sufficiently steep, outward-increasing potential turns the interface into a virtual Neumann boundary, decoupling the core. Second, such a steep potential forces $psi_1$ to be constant along the flow lines of $-nabla V$. Third, we shape those flow lines so that they transport $q$ into the wing and trim its secondary peaks along the way.

The operator $L = -∆ + nabla V dot nabla$ is self-adjoint with respect to the weighted measure $dif mu = e^(-V) dif x$, and $psi_1$ is characterized by

$
integral_Q nabla psi_1 dot nabla v thin e^(-V) dif x = lambda_1 integral_Q psi_1 thin v thin e^(-V) dif x quad "for all" v in H^1(Q).
$ <eq:weak-eig>

Suppose that $V$ in the wing grows steeply outward, $V(x) gt.tilde V(pi/2, x_2) + s (abs(x_1) - pi/2)$ for a large constant $s$. Then the weight $e^(-V)$ decays by a factor $e^(-s (abs(x_1) - pi/2))$ across the wing, so $Q_"wing"$ carries an exponentially small fraction of the total mass $mu(Q)$. In @eq:weak-eig the integrals over $Q_"wing"$ are therefore negligible against those over $Q_"core"$, and the identity reduces, up to exponentially small error, to

$
integral_(Q_"core") nabla psi_1 dot nabla v thin e^(-V) dif x approx lambda_1 integral_(Q_"core") psi_1 thin v thin e^(-V) dif x quad "for all" v in H^1(Q_"core").
$

This is exactly the weak Neumann eigenproblem on the core, with the interface $x_1 = pi/2$ now a free boundary on which the natural condition $∂_arrow(n) psi_1 = 0$ holds. Hence $psi_1|_(Q_"core")$ agrees, to leading order, with the core eigenfunction constructed before: the steep wing acts as a virtual Neumann boundary condition at $∂ Q_"core"$, without one being imposed.

#inline-note-a[
  TODO: A paragraph explaining why $nabla V dot nabla psi_1 approx 0$ in the wing.
]

We have turned the construction of the wing potential into a problem about flow lines: since $psi_1$ is constant along the flow lines of $-nabla V$, we may choose $V$ so that these flow lines carry each interface height to the right place in the wing. It is natural to organize the wing into three: a central channel around the main peak at $x_2 = 0$, and two outer channels from the minima of $q$ out to $x_2 = plus.minus 1$. In the central channel the flow lines should run straight and horizontal, so that the main peak is transported across the wing unchanged and $psi_1 approx q(x_2)$ throughout. In the two outer channels the flow must instead be compressed inward: looking from the end of the wing back toward the core, the flow liens of each outer channel converge toward the center. As a result, the strips near $x_2 = plus.minus 1$ at the wing end trace back to the minima of $q$, producing the trimmed profile $tilde(q)$ at the wing end.


== Parametrizing the potential and guaranteed convexity

So far we have constructed the core potential and the wing potential separately, on $Q_"core"$ and $Q_"wing"$. To use them in the eigenvalue problem we need a single potential on all of $Q$ that is smooth across the interface $x_1 = pi\/2$ and globally convex. In the core construction we enforced convexity by adding a quadratic term. Choosing $M_1, M_2$ so that the Hessian (@eq:perturbation-terms) is positive semidefinite, is a condition we can only check numerically. We now describe a parametrization that gives smoothness, convexity, and the gluing of the two pieces all at once.

#definition[
  Let $x in RR^n$ and $T > 0$. The _log-sum-exp_ function with _temperature_ $T$ is
  $
  LSE_T (x_1, ..., x_n) = T log sum_(i=1)^n e^(x_i slash T).
  $
]

We represent the potential as the $LSE$ of a collection of affine planes $l_i (x) = a_i x_1 + b_i x_2 + c_i$,

$
V_"LSE" (x) = LSE_T (l_1(x), ..., l_n (x)).
$

The function $LSE_T$ is convex and nondecreasing in each argument, composed with the affine $l_i$ it follows immediately that $V_"LSE"$ is convex. Both convexity and smoothness hold _by construction_, for every choice of planes.

$LSE_T$ is a smooth approximation of the maximum, with the standard two-sided bound

$
max_i l_i (x) ≤ V_"LSE" (x) ≤ max_i l_i (x) + T log n,
$

so $T$ controls the trade-off between accuracy of the approximation (small $T$) and smoothness (large $T$).

To approximate a given convex potential $tilde(V)$ on a region we sample points $x_i$ on a grid and take $l_i$ to be the tangent plane of $tilde(V)$ at $x_i$. Because $tilde(V)$ is convex, each tangent plane is a supporting hyperplane, $l_i (x) ≤ tilde(V)(x)$ everywhere with equality at $x_i$, so $max_i l_i ≤ tilde(V)$ and the bound above sandwiches $V_"LSE"$ within $T log n$ of $tilde(V)$.

The key feature of this representation is that gluing is simple. We fit the core potential on $Q_"core"$ to a set of planes ${l_i^"core"}$ and, independently, the wing potential on $Q_"wing"$ to a set of planes ${l_j^"wing"}$. The global potential is simply the $LSE$ over the union of both collections,

$
V (x) = LSE_T ({l_i^"core"} union {l_j^"wing"}).
$

The steep outward growth of $V$ in the wing makes $Q_"wing"$ carry only a negligible fraction of the total mass. With the explicit potential at hand this can be approximated quadrature. In future work, this should be replaced by verified quadrature.

#lemma[
  $mu(Q_"wing") ≤ 10^(-10).$
]<lem:wing-mass>
#proof[
  By definition $mu(Q_"wing") = (integral_(Q_"wing") e^(-V) dif x) slash (integral_Q e^(-V) dif x)$. To evaluate the integrand $e^(-V) = (sum_i e^(l_i (x) slash T))^(-T)$ without overflow we factor out the dominant plane $M(x) = max_i l_i (x)$,
  $
  e^(-V(x)) = (sum_i e^(l_i (x) slash T))^(-T) = e^(-M(x)) (sum_i e^((l_i (x) - M(x)) slash T))^(-T),
  $
  so that every exponential lies in $(0, 1]$. The resulting quadrature gives the stated bound.
]

Since $V_"LSE"$ is fully parametrized by the plane coefficients, it can be optimized against any differentiable objective, and crucially convexity is enforced automatically: every point of the parameter space yields a convex potential, so the optimizer does not need to respect a convexity constraint. This opens up several directions. After solving for the eigenfunction $phi.alt_1$ with the MPS, one could turn the problem around and adjust the potential to reduce the $L^oo$ error in the Neumann boundary condition.  More difficulty, one could optimize $V_"LSE"$ directly to, maximizie the gap $max_(Q^circle) h(dot, 1\/8) - max_(∂Q) h(dot, 1\/8)$, and therefore strengthen the counterexample itself.