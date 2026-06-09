#import "template.typ": *

= Constructing a counterexample set <construction>

In this section we aim to explain the counterexample from @pont_convex_2024 and how we adapted it to a version which can be computed explicitly. Since our proof relies on verified numerics, we do not proof any properties of the counterexample set rigorously, but rather explain the inuition behind the reason why we expect the counterexample to work.

The counterexample is effectively built in the limit $d -> oo$, we first have to understand how to eigenvalue problem behaves in this limit. This is the subject of the next subsection.

== The effective problem in the limit of dimension

The sequence of principal eigenfunctions $phi_(1,d)$ of $F_d$ at dimension $d$ converges to a function $h$ that is an eigenfunction of $L = -∆ + nabla V dot nabla$ at $r=1$ and satisfyes the following initial value problem:

$
∂_r h &= -r/4 (∆_x + lambda) h && "for" x in Q, r in [0, 1) \
h(x, 1) &= psi_1 (x) && "for" x in Q \
// ∂_r h(x, r) &= 0 "at" r = 0 \ 
∂_arrow(n) h &= 0 && "for" x in ∂ Q,
$ <eq:limit-problem>

here $psi_1$ is the first non-trivial eigenfunction of $L$. The first equation follows immediately from the eigenvalue equation $-Delta phi_(1, d) = lambda_(1, d) phi$ since

$
∆ = ∆_x + 4/d ∂_r^2 + 4/r ∂_r -> Delta_x + 4/r ∂_r.
$

For the boundary condition we examine rayleigh quotient

$
// (integral_Omega abs(nabla u)^2 dif z) / (integral_Omega abs(u)^2 dif z)
// &= 
(integral_Q integral_0^(rho_d (V)) abs(nabla u)^2 r^d dif r dif x) / (integral_Q integral_0^(rho_d (V)) abs(u)^2 r^d dif r dif x) 
&= (integral_Q abs(nabla u)^2 (sqrt(d) - V(x)/sqrt(d))^(d+1) dif x) / (integral_Q abs(u)^2 (sqrt(d) - V(x)/sqrt(d))^(d+1) dif x)
-> (integral_Q abs(nabla u)^2 e^(-V(x)) dif x) / (integral_Q abs(u)^2 e^(-V(x)) dif x).
$

This is the Rayleigh quotient for the Neumann problem $L u = lambda u$ on $Q$. 

// Another way to see the same thing is te examine the boundry condition $∂_arrow(n) u = 0$ on the face $r=rho$.
// Maybe add this later if I have time

The boundary value problem in @eq:limit-problem can be transformed into a reaction-diffusion initial value problem by the change of variables $t = (1-r^2)/8$, we obtain

$
∂_t h &= ∆_x h+ lambda h "for" t in (0, 1\/8]\
h(x, 0) &= psi_1(x) \
∂_arrow(n) h &= 0 "on the spatial boundary".
$<eq:limit-ivp>

The goal is now to find a suitable initial value $psi_1$.

== The counterexample in @pont_convex_2024 and adapting it for computation

In this section we explain the main idea behin the counterexample in @pont_convex_2024 and how to build the corresponding potential explicitly in a way suitable for computation. We begin our discussion with the initial condition for @eq:limit-ivp we are aiming for. This is perhaps best explained by a plot of $psi_1$, see @fig:initial-datum. 

#figure(
  grid(
    columns: 2,
    image("eigenfunction_split.png", width: 50%),
    image("eigenfunction_surface.png", width: 50%)
  ),
  
  caption: [Left: The eigenfunction on the full domain.  Center: The green region magnified. Right: Surface plot of the region.]
)<fig:initial-datum>

The axis of the Barrel decomposes into two parts: The "core" $A_"core" = [0, pi\/2] times [0,1]$ and the "wing" $A_"wing" = [pi\/2,pi\/2 + m]$. On a macroscopic scale we want $psi_1$ to look like $sin(x_1)$ in $A_"core"$, that is, like the principal eigenfunction of $-∆$ on $A_"core"$. In $A_"wing"$ we want $psi_1$ to be a constant extension of the core function. On a microscopic scale we want $psi_1$ to perturbed in such a way that at the interface of $A_"core"$ and $A_"wing"$ we have $psi_1(x) approx q(x_2) + "const."$, where $q$ is as in @fig:q. This profile should extend into $A_"wing"$ but lose its high points at $x_2 = plus.minus 1.0$ before the end of the wing, see @fig:q. The initial datum $psi_1$ achieves its maximum at $x = (pi\/2 +m, 0)$, which is on the boundary. The heat extensions of the profile $q$ and its trimmed version both achieve their maximum at the origin at any time. However, the heat extension of $q$ will be strictly larger than the one of the trimmed versions, due to the additional energy in the tails. The full solution $h$ will capture the same phenomenon, but  due to the additional reaction term in @eq:limit-ivp, the extension $h$ will be strictly larger near $t=1/8$ (or $r=0$),

$
h(x,r) = sum_(j,k) c_(j,k) X_(j,k) (x) exp(-r^2 (lambda_(x,j,k) - lambda_1) / 8),
$

hence the maximum of $h$ will be in the interior near $x_2=0, r=0$ and $x_1 in (pi/2, pi/2 + m)$.


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


=== Deriving a potenital to obtain the desired initial datum

In order to build a potential $V$ such that $psi_1$ has the before-mentioned properties in $A_"core"$, we consider the operator $L = -∆ + nabla V dot nabla$ to be a perturbation of the Laplacian $-∆$. Let $V_epsilon = epsilon^(-1) V$, then we can write $L$ as

$
L = -∆ + epsilon nabla V_epsilon dot nabla.
$

Expand both $lambda_1$ and $psi_1$ in powers of $epsilon$,

$
lambda_1 &= tilde(lambda)_1 + epsilon mu + O(epsilon^2) \
psi_1 &= tilde(psi)_1 + epsilon beta + O(epsilon^2).
$

Substituting the expansions into $L psi_1 = lambda_1 psi_1$ and collecting terms of the same order in $epsilon$ we get

$
tilde(psi)_1 &= sqrt(pi/2) sin(x_1), quad tilde(lambda)_1 = 1 \
(-∆-1) beta &= sqrt(pi/2) (mu sin(x_1) - (∂_1 V_epsilon) cos(x_1)).
$

We already obtain the correct macroscopic behavior of $psi_1$ since $epsilon$ is small. The rest of the desired properties will be enabled by $beta$. Let $beta_0$ be

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

*Extending the potential to the wings.* In the previous paragraph we built $V_0$ such that the _Neumann_ eigenfunction $psi_1$ has the perscribed boundary profile $q$. In order to mantain this profile at the core-wing interface while extending the domain, we need $V$ to act as a virtual Neumann boundary condition. We can achieve this by setting $V(x) = V(pi/2, 1.0) + 10^7 (abs(x_1)-pi/2)$ in $A_"wing"$, this way particles coming from $A_"core"$ have a high probability of reflecting at $x_1 = pi/2$, this simulates the reflecting boundary conditions. 

#inline-note-a[
  Here I explain why $psi_1$ is a approximately constant along the flow lines of $nabla V$, if $nabla V$ is large. We can do this through the probabilistic interpretation:
  $
  P_t psi_1 = e^(-lambda_1 t) psi_1 \
  P_t f(x) = EE[f(X_t) | X_0 = x] \
  dif X_t = - nabla V (X_t) dif t + sqrt(2) dif B_t
  $
  But all of this would have to be introduced beforehand. Maybe there is a simpler way. Let $h$ be the extension of $psi_1$, then, near $r=1$ we have $nabla V dot nabla h approx ∆_x h + lambda_1 h = 0$.
]

$
V(x) = 
cases(
  V_0(x) "if" x in A_"core",
  V(pi/2, 1.0) + 10^7 (abs(x_1)-pi/2)
)
$

#inline-note-a[
  In the wing region the eigenfunction is approximately constant on the flow lines. Use this to transport the central part of the boundary profile outwards. 
]