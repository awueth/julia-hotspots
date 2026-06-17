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


We split the base $Q$ of the barrel into two parts: The _core_ $Q_"core" = [-pi/2, pi/2] times [-1, 1]$ and the _wing_ $Q_"wing" = ([-2pi, 2 pi] times [-1, 1]) without Q_"core"$. On a macroscopic scale we want $psi_1$ to look like $sin(x_1)$ in $Q_"core"$, that is, like the principal eigenfunction of $-∆$ on $Q_"core"$. In $Q_"wing"$ we want $psi_1$ to be a constant extension of the core function. On a microscopic scale we want $psi_1$ to be perturbed in such a way that at the interface of $Q_"core"$ and $Q_"wing"$ we have $psi_1(x) approx q(x_2) + "const."$, where $q$ is as in @fig:q. This profile should extend into $Q_"wing"$ but lose its high points at $x_2 = plus.minus 1.0$ before the end of the wing, see @fig:q. The initial datum $psi_1$ achieves its maximum at $x = (2pi, 0)$, the end of the wing, which is on the boundary. The heat extensions of the profile $q$ and its trimmed version both achieve their maximum at the origin at any time. However, the heat extension of $q$ will be strictly larger than the one of the trimmed versions, due to the additional energy in the tails. The full solution $h$ will capture the same phenomenon, but  due to the additional reaction term in @eq:limit-ivp, the extension $h$ will be strictly larger near $t=1/8$ (or $r=0$), hence the maximum of $h$ will be in the interior near $x_2=0, r=0$ and $x_1 in (pi/2, 2pi)$.


#figure(
  lq.diagram(
    let ys = lq.linspace(-1.0, 1.0),
    //let q(y) = 0.5*calc.cos(calc.pi*y) + 0.3*calc.cos(2.0*calc.pi*y) - 0.2*calc.cos(3.0*calc.pi*y),
    let q(y) = 0.2*calc.cos(calc.pi*y) + 1.0*calc.cos(2.0*calc.pi*y) - 0.0*calc.cos(3.0*calc.pi*y),
    let q_trimmed(y) = if calc.abs(y) < 0.67 {q(y)} else {q(0.67)},
    lq.plot(ys, q, color: blue, mark: none),
    lq.plot(ys, q_trimmed, color: red, mark: none, stroke: (dash: "dashed")),
  ), 
  caption: [Cross section of $psi_1$ at the core-wing interface and at the end of the wing.]
)<fig:q>

#figure(
  grid(
    columns: 2,
    image("eigenfunction_split.png", width: 50%),
    image("eigenfunction_surface.png", width: 50%)
  ),
  
  caption: [Left: The eigenfunction on the full domain.  Center: The green region magnified. Right: Surface plot of the region.]
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
$

We already obtain the correct macroscopic behavior of $psi_1$, given that $epsilon$ is small enough. The rest of the desired properties will be enabled by $beta$. Let $beta_0$ be

- antisymmetric in $x_1$,
- symmetric in $x_2$,
- have Neumann boundary data with $beta_0(pi\/2,x_2) = q(x_2) + "const."$, and 
- satisfy $(∆+1) beta_0(pi\/2, x_2) = 0$.

The function

$
V_0 (x_1,x_2) = -sqrt(2/pi) ∫_0^x_1 ((∆ + 1) beta_0 (s,x_2)) / cos(s) dif s
$

satisfies

$
(-∆-1) beta_0 &= -sqrt(pi/2) (∂_1 V_0) cos(x_1).
$

From the function $V_0$ we obtain the convex potential $V(x) = V_0(x) + 1/2 M norm(x)^2$, for $M$ large enough. In practice we get a lower bound for $M$ by computing the least eigenvalue of the hessian of $V$. Let $s(x_1), mu$ be a solution of the ODE $-∂_1^2 s(x_1) - s(x_1) = sqrt(pi/2) (mu/M sin(x_1) - x_1 cos(x_1))$ on $[-pi\/2, pi/2]$ with Neumann boundary conditions. Then, $beta(x_1, x_2) = beta_0 (x_1, x_2) + M s(x_1)$ satisfies

$
(-∆-1) beta(x_1,x_2)
&= sqrt(pi/2)(-(∂_1 V_0) cos(x_1) + M(mu/M sin(x_1) - x_1 cos(x_1))) \
&= sqrt(pi/2)(mu sin(x_1) - (∂_1 V) cos(x_1)),
$

as desired.


*Constructing $beta_0$* We use the ansatz  
  
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

In the previous paragraph we built $V_0$ such that the _Neumann_ eigenfunction $psi_1$ has the perscribed boundary profile $q$. In order to mantain this profile at the core-wing interface while extending the domain, we need $V$ to act as a virtual Neumann boundary condition.


*The probabilistic interpretation of the semigroup.* Let $P_t = e^(t L)$ be the semigroup corresponding to $L$. We have that $P_t psi_1 = e^(-lambda_1 t) psi_1$. Let $X_t$ denote the stochastic process described by $dif X_t = - nabla V (X_t) dif t + sqrt(2) dif B_t$, then the action of $P_t$ on a function $f$ can be written as

$
P_t f(x) = EE[f(X_t) | X_0 = x].
$

The Neumann boundary condition is encoded by haveing $dif X_t$ reflect at the boundary. Therefore, by letting $nabla V$ increase sharply at the interface between $Q_"core"$ and $Q_"wing"$, we can enforce a virtual Neumann boundary condition at $partial Q_"core"$. We can achieve this by setting $V(x) = V(pi/2, 1.0) + 10^7 (abs(x_1)-pi/2)$ in $Q_"wing"$, this way particles coming from $Q_"core"$ have a high probability of reflecting near $x_1 = pi/2$, this simulates the reflecting boundary conditions. 

Next, we establish that the eigenfunction $psi_1$ is approximately constant along the flow lines of $nabla V$ in $Q_"wing"$. Indeed, let $h$ be the extension of $psi_1$, then, near $r=1$ we have $nabla_x V dot nabla_x h(dot, r) approx ∆_x h + lambda_1 h = 0$. Therefore, since $h$ is smooth, we have $nabla V dot nabla psi_1 approx 0$. As a result, we can transport the profile at the core-wing interface via the flow lines of $nabla V$.
