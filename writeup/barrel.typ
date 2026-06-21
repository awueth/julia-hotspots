#import "template.typ": *

= Barrel sets <barrels>

#definition[Barrel domain][
  Let $Q subset RR^2$ be a bounded convex domain and let $V : Q -> RR$ be a convex potential. We define the convex domain

  $ F_d (Q, V) := {(x,w) in Q times RR^(d+1) : |w| ≤ 1/2 (sqrt(d) - V(x)/sqrt(d))}, $

  which we call a barrel domain.
]

The constant factor $1/2$ is completely arbitrary, we chose it for consistency with @pont_convex_2024. The sets barrel sets get their name from the fact that for an interval $I$ the set $F_1 (I, V)$ looks like a barell, see @fig:barell.

#figure(
  image("barrel.svg", width: 25%),
  caption: [For $Q$ an interval $F_1(Q,V)$ looks like a barell. The curved wall corresponds to the graph of the potential $V$. The actual counterexample, the cylinder axis $Q$ is a rectangle.]
) <fig:barell>

For the purpose of this work, we use $Q = [-2 pi, 2 pi] times [-1,1]$. We also use a fixed potential $V :Q -> RR$ which is symmetric in both coordinates. We will denote the eigenvalues of the Neumann Laplacian on $F_d (Q,V)$ by $0 = lambda_(0,d) < lambda_(1,d) ≤ lambda_(2,d) ≤ ...$, with multiple eigenvalues listed separately. The corresponding eigenfunctions will be denoted by $phi.alt_(0,d), phi.alt_(1,d), phi.alt_(2,d)$ and so on.  We will refer to $(lambda_(1, d), lambda_(1, d))$ as the "first" eigenvalue and eigenfunction.

// #inline-note-a[
//   *outdated content:*

//   The constant factor $1/4$ in the radius is completely arbitrary. In the @pont_convex_2024 it was chosen to be $1/2$, however, we chose $1/4$ since it allows us to show that eigenfunctions with eigenvalue less than $4^2$ are radial in $w$ for large $d$. With the factor $1/2$ we can only prove the same result for eigenvalues less than $2^2$. 
// ]

// Since the boundary of $F_d (Q, V)$ is not differentiable at points $(x, w)$ where $w in ∂ Omega$, we define the Neumann eigenfunctions using the weak formulation. A non-zero function $u in H^1 (Omega)$ is a Neumann eigenfunction with eigenvalue $lambda ≥ 0$ if it satisfies the equality $integral_Omega nabla u dot nabla v dif x = lambda integral_Omega u v dif x$, for all $v in H^1 (Omega)$.

== The low barrel eigenfunctions are radial

The counterexample is in high dimension $d$. In order to compute its eigenfunction, we have to reduce the number of effective dimensions by showing that the ground eigenfunction is radial in the $w$-coordinate. The proof goes in two steps: First we prove that any eigenfunction with a radial dependence must have a high eigenvalue. In a second step we compute an approximate radial eigenfuntion and compute its Rayleigh quotient, showing that the ground eigenvalue is small. By contradiction the eigenfunction bust be radial. 

#lemma[
  Let $phi.alt_(k, d)$ be the $k$-th Neumann eigenfunction of the Laplacian on $F_d (Q, V)$ and let $lambda_(k, d)$ be the corresponding eigenvalue. If $phi.alt_(k,d)(x, w)$ is not radial in the $w$-coordinate, then 
  $
  lambda_(k,d) ≥ 4 (d/(d-V(0)))^2.
  $
] <lem:eig-radial>
#proof[
  In cylindrical coordinated the Lapalacian is $∆ = ∆_x +  ∂_r^2 + d/r ∂_r + 1/r^2 ∆_(S_d)$. We separate $phi.alt_(k,d)$ into eigenfunctions of $-(∆_x +  ∂_r^2 + d/r ∂_r)$ and $-1/r^2 ∆_(S_d)$, that is

  $
  phi.alt_(k,d) (x, w) = sum_(ell≥0, m) A_(ell, m)(x, abs(w)) Y_(ell, m) (theta),
  $
  
  where $-∆ A_(ell, m) = lambda_(ell, m) A_(ell, m)$ and $-∆ Y_(ell, m) = ell (ell + d -1) Y_(ell, m)$. Each component $A_(ell, m) Y_(ell, m)$ is itself an eigenfunction of $-∆$ corresponding to the eigenvalue $lambda_(k,d)$. Therefore, each component is an eigenfunction and its Rayleigh quotient exactly evaluates to the eigenvalue, thus
  
  $
  lambda_(k,d) = (∫ abs(nabla A_(ell, m) Y_(ell, m))^2) / (∫ abs(A_(ell, m) Y_(ell, m))^2)
  ≥ ell (ell + d - 1) (∫_Ω∫_0^(rho(x)) abs(A_(ell, m))^2 r^(d-2) dif r dif x) / (∫_Ω∫_0^(rho(x)) abs(A_(ell, m))^2 r^d dif r dif x)
  ≥ (ell (ell + d - 1)) / rho_"max"^2.
  $

  Therefore, whenever $A_(ell, m) ≠ 0$ for some $ell ≥ 1$, we have $lambda_(k,d) ≥ d / rho_"max"^2$ and
  $rho_max = 1/2 (sqrt(d) - V(0)/sqrt(d))$.
]

A direct consequence of the above lemma is that any eigenfunction with eigenvalue less than $4$ is radial in high enough dimension. If we restrict the problem to radial eigenfunctions, then the change of variables

$
r = 2 abs(w) d^(-1/2), 
$

transforms the Barel set $F_d (Q, V)$ into

$
Omega_d = {(x,r) in Q times RR_(≥ 0) : r ≤ 1 - V(x)/d}.
$

In the $(x,r)$-coordinates, the Laplacian becomes

$
Delta_(x,r) = Delta_x + 4/d ∂_r^2 + 4/r ∂_r.
$

with Neumann boundary conditions at $r=0$.

On $Omega_d$ we have the eigenvalues $0 = lambda_(0, Omega_d) < ...$ and eigenfunctions $phi.alt_(0, Omega_d), phi.alt_(1, Omega)$ and so on. From now on, we will simply assume that the eigenfunctions $phi.alt_(1,d), phi.alt_(2,d)$ on the full Barrel are radial. We will therefore overload the notation and also write $phi.alt_(i,d)$ for $phi.alt_(i, Omega_d)$. In @certificate we will show that $lambda_(2, Omega_d) < 4$, therefore $lambda_(i, Omega_d) = lambda_i$ for $i=0,1,2$ and our choice of $d$. Consequently, $phi.alt_(1,d), phi.alt_(2,d)$ are indeed radial, and, therefore if $phi.alt_(1, Omega_d)$ achieves its maximum in ${(x,r) in Q^circle times RR_(≥ 0) : r < 1 - V(x)/d}$ then $phi.alt_(1,d)$ attains its maximum in the interior of the full Barrel.

// #inline-note-a[
//   *How the above scales with scaling of the domain.* If we rescale $Q$ to $s Q$ and $V$ to $V_s = V(x\/s, y\/s)$, then the lower bound if the eigenfunction is not radial stays the same. However, the eigenvalue $lambda_(1,d)$ multiplies by $1/s^2$, hence the conclusion of the above lemma holds for $s Q, V_s$ whenever $lambda_(Q, V) < 4 s^2$.
// ]


== The effective problem in the limit of dimension

In this subsection we study the eigenvalue problem on $Omega_d$ as $d -> oo$. Recall that the Laplacian on $Omega_d$ is given by $Delta_x + 4/d ∂_r^2 + 4/r ∂_r -> Delta_x + 4 r^(-1) ∂_r$ as $d-> oo$, therefore, formaly

$
∂_r phi.alt_1 = -r/4 (Delta_x + lambda_1) phi.alt_1,
$

for all $r in [0, 1)$. The boundary $r=1$ is defined by $0=G(x,w) = abs(w) - sqrt(d)/2 (1 - V(x)/d)$ on the barrel $F(Q,V)$. The outward normal vector is proportional to $nabla_(x, w) G = (1/(2 sqrt(d)) nabla_x V, w/abs(w))$. By the chain rule $nabla_w phi.alt_(1, Omega_d) = ∂_r phi.alt_(1, Omega_d) 2/(sqrt(d)) w/abs(w)$. Now,

$
0 
= nabla_(x, w) G dot nabla_(x, w) phi.alt_1 
&= 1/(2 sqrt(d)) nabla_x V dot nabla_x phi.alt_1 + 2/sqrt(d) ∂_r phi.alt_1 \
&= 1/(2 sqrt(d)) nabla_x V dot nabla_x phi.alt_1 - 2/sqrt(d) 1/4 (Delta_x + lambda_1) phi.alt_1 \
&= 1/(2 sqrt(d)) (-Delta_x + nabla_x V dot nabla - lambda) phi.alt_1.
$

Therefore, if we let $L := - Delta + nabla V dot nabla$ and write $psi_1 (x) := phi.alt_1(x, r=1)$, then $psi_1$ is an eigenfunction of $L$ with eigenvalue $lambda_1$. In conclusion, the eigenvalue problem on $F_d (Q, V)$ turns into a boundary value problem at $d=oo$:

$
∂_r phi.alt &= -r/4 (∆_x + lambda_1) phi.alt && "for" x in Q, r in [0, 1) \
phi.alt(x, 1) &= psi (x) && "for" x in Q \
// ∂_r h(x, r) &= 0 "at" r = 0 \ 
∂_arrow(n) h &= 0 && "for" x in ∂ Q.
$ <eq:limit-problem>

=== The conncection to the log-concave problem

In the introduction we claimed that the spectrum of barell sets $F_d (Q, V)$ approximates the spectrum of $Q$ with respect to the log-concave measure $dif mu(x) = e^(-V(x)) dif (x)$, as $d -> oo$. In order to make this connection, note that the divergence form of $L psi_1 = lambda psi_1$ is $-nabla dot (e^(-V) nabla psi_1) = lambda_1 e^(-V) psi_1$. To obtain the weak formulation, we multiply by a test function $v$ and integrate with respect to the Lebesgue measure:

$
- integral_Q nabla dot (e^(-V) nabla psi_1) v dif x = lambda_1 integral_Q psi_1 v e^(-V) dif x
$

Integrating by parts and using $partial_arrow(n) psi_1 = 0$ we obtain

$
integral_Q nabla psi_1 dot nabla v space e^(-V) dif x = lambda_1 integral_Q psi_1 v space e^(-V) dif x, 
$

i.e. $psi_1$ is an eigenfunction with respect to the log-concave measure $mu$ in the weak sense. We will not make this rigorous since we only want to motivate why constructing a counterexample at $d=oo$ is useful. However, we need explicit a priori bounds of the eigenvalues at finite dimensions. We will therefore now derive explicit bounds of $lambda_(i,d)$ in terms of $lambda_(i, oo)$.

Consider the diffeomorphism,

$ Phi : F_d (Q, V) -> Q times B_(sqrt(d)/2), quad (x,w) |-> (x, a(x) w) $

where $a(x) := (1 - V(x)/d)^(-1)$, which "compresses" the barrel into a cylinder. Let $cal(L)_d$ denote the lebesgue measure on $F_d$. The pushforward $Phi_hash cal(L_d)$ of $cal(L)_d$ has density

$
dif (Phi_hash cal(L)_d)(x, w) = det(D Phi)^(-1) dif x dif w.
$

And,

$
det (D Phi)^(-1)
= det mat(
  I_2, 0;
  w (nabla a)^T, a I_(d+1)
)^(-1)
= (1 - V(x)/d)^(d+1)
-> exp(-V).
$

In a sense, $Phi$ is almost mass preserving, more explicitly:

#lemma[
  Let $cal(L)$ be the Lebesgue measure on $F_d (Q, V)$ and let $kappa_d$ be the Lebesgue measure on $B_(sqrt(d)/2)$.
  
  Then, 
  $
  d/(d+norm(V)_oo) ≤ abs((dif Phi_hash cal(L)_d)/(dif (mu times.o kappa_d))) ≤ d/(d-norm(V)_oo) exp(norm(V)_oo^2/(d-norm(V)_oo)).
  $

  If $V$ is nonnegative on $Q$, then we even have $1 ≤ abs((dif Phi_hash cal(L)_d)/(dif (mu times.o kappa_d)))$.

]<lem:mass-preservation>

We furthermore claim that $Phi$ is almost an isometry:

#lemma[
  $
  norm(D Phi - I)_"op" ≤ sqrt(norm(V)_oo^2 + d/4 norm(nabla V)_oo^2) / (d-norm(V)_oo) = O(d^(-1/2)).
  $
]<lem:isometry>

The two lemmas are exactly the assumptions of the following lemma, which shows that the eigenvalues converge. 

#lemma[
  Let $(X, mu)$ and $(Y, nu)$ be measure spaces, $Phi : X -> Y$. Assume that $Phi$ is approximately mass preserving, i.e. $underline(beta) ≤ abs((dif Phi_hash mu)/(dif nu) (y)) ≤ overline(beta)$ for all $y in Y$. Assume $norm(D Phi - I)_"op" < epsilon$. Then,

  $
  (1-epsilon)^2 underline(beta) / overline(beta) lambda_k (nu) ≤ lambda_k (mu) ≤ (1+epsilon)^2 overline(beta) / underline(beta) lambda_k (nu).
  $
]

#proof[
  Write $a_mu (u, v) = integral nabla u dot nabla v dif mu$ and $m_mu (u,v) = integral u v dif mu$.

  Let $u, v in Y$,

  $
  m_mu (u compose Phi, v compose Phi) = integral_Y u v dif (Phi_hash mu) = integral_Y u v (dif Phi_hash mu)/(dif nu) dif nu. 
  $

  Therefore, since $underline(beta) ≤ (dif Phi_hash mu)/(dif nu) (y) ≤ overline(beta)$ for all $y in Y$,

  $
  underline(beta) m_nu (u, v) ≤ m_mu (u compose Phi, v compose Phi) ≤ overline(beta) m_nu (u, v).
  $

  For the Dirichlet energy: Let $M(x) := D Phi(x) D Phi^T (x)$

  $
  a_mu (u compose Phi, u compose Phi)
  &= integral_X nabla u(Phi(x))^T M(x) nabla u(Phi(x)) dif mu(x) \
  &= integral_Y nabla u(y)^T M(Phi^(-1)(y)) nabla u(y) (dif Phi_hash mu)/(dif nu) (y) dif nu(y).
  $

  Since $norm(D Phi - I)_"op" < epsilon$, the eigenvalues of $M = D Phi space D Phi^T$ lie in $((1-epsilon)^2, (1+epsilon)^2)$. Combined with the mass-preservation bounds on $(dif Phi_hash mu)/(dif nu)$ this gives

  $
  (1-epsilon)^2 underline(beta) space a_nu (u, u) ≤ a_mu (u compose Phi, u compose Phi) ≤ (1+epsilon)^2 overline(beta) space a_nu (u, u).
  $

  For the Rayleigh quotient it follows that

  $
  (1-epsilon)^2 underline(beta) / overline(beta) R_nu (u) ≤ R_mu (u compose Phi) ≤ (1+epsilon)^2 overline(beta) / underline(beta) R_nu (u).
  $

  By Courant-Fischer and the fact that $Phi$ is a bijection, we have

  $
  (1-epsilon)^2 underline(beta) / overline(beta) lambda_k (nu) ≤ lambda_k (mu) ≤ (1+epsilon)^2 overline(beta) / underline(beta) lambda_k (nu).
  $

]

#proof[Proof of @lem:mass-preservation][
  We have to bound $abs((dif Phi_hash cal(L))/(dif (mu times.o kappa_d)) - 1)$ uniformly. Recall that

  $
  dif (Phi_hash cal(L)_d)(x, w) = (1 - V(x)/d)^(d+1) dif x dif w,
  quad dif(mu times.o kappa_d)(x,w) = e^(-V(x)) dif x dif w.
  $

  Hence we need a lower and an upper bound on

  $
  f(x) = underbrace((e^(-V))/((1 - V/d)^d), =: A) dot 1/(1 - V/d).
  $

  The inequality $1 + t ≤ e^t$ with $t = -V/d$ gives $(1 - V/d)^d ≤ e^(-V)$, hence $A ≥ 1$ pointwise. 
  
  For the upper bound write $h(V) := log(A) =  -V - d log(1 - V/d)$, so that $h(0) = 0$ and
  $
  h'(V) = -1 + 1/(1 - V/d) = V/(d - V).
  $
  Thus $h ≥ 0$, and for $abs(V) ≤ M$, integrating along the segment from $0$ to $V$ (where the integration variable satisfies $abs(s) ≤ M$, so $d - s ≥ d - M$),
  $
  0 ≤ h(V) = integral_0^V s/(d-s) dif s ≤ M/(d-M) integral_0^(abs(V)) dif s ≤ M^2/(d-M).
  $
  Together with $1 - V/d in [1 - M/d, 1 + M/d]$ this yields the explicit two-sided bound
  $
  d/(d+M) ≤ f(x) ≤ d/(d-M) exp(M^2/(d-M)).
  $
  In particular, if $V ≥ 0$ on $Q$ then $(1 - V/d)^(d+1) ≤ (1 - V/d)^d ≤ e^(-V)$, so $f ≥ 1$ and the lower side is sharp. 
]

#proof[Proof of @lem:isometry][
  We have

  $
    D Phi - I =
    mat(
      0_2, 0;
      w (nabla a)^T, (a-1) I_(d+1)
    ).
  $

  Therefore,

  $
    norm(D Phi - I)_"op" = sqrt((a-1)^2 + abs(w)^2 abs(nabla a)^2).
  $

  Now, $a-1 = V / (d - V)$ and $nabla a = (nabla V) / d (1 - V/d)^(-2)$ and $abs(nabla w) abs(nabla a) ≤ abs(nabla V) / (2 sqrt(d)) (1 - V(x)/d)^(-1) = (sqrt(d) abs(nabla V))/(2(d-V))$ . Therefore, 

  $
  norm(D Phi - I)_"op" 
  ≤ (V^2 / (d - V)^2 + (d abs(nabla V)^2) / (4 (d-V)^2))^(1/2)
  = sqrt(V^2 + d/4 abs(nabla V)^2) / (d-V).
  // ≤ sqrt(4 norm(V)_oo^2 + d norm(nabla V)_oo^2) / (2 (d - norm(V)_oo))
  $
]

Combining the three lemmas gives the explicit convergence rate.

#corollary[
  Let $M := norm(V)_oo$ and
  $
  epsilon_d := sqrt(M^2 + d/4 norm(nabla V)_oo^2) / (d - M) = O(d^(-1/2)).
  $
  With $underline(beta)_d = d/(d+M)$ and $overline(beta)_d = d/(d-M) exp(M^2/(d-M))$, the Neumann eigenvalues of $F_d (Q, V)$ satisfy

  $
  (1-epsilon_d)^2 underline(beta)_d / overline(beta)_d lambda_k (mu times.o kappa_d) ≤ lambda_k (cal(L)_d) ≤ (1+epsilon_d)^2 overline(beta)_d / underline(beta)_d lambda_k (mu times.o kappa_d).
  $

  In particular, $lambda_k (cal(L)_d) = (1 + O(d^(-1/2))) lambda_k (mu times.o kappa_d)$ as $d -> oo$.

  If $V ≥ 0$ on $Q$, then $underline(beta)_d = 1$ and the lower bound improves to $(1-epsilon_d)^2 / overline(beta)_d lambda_k (mu times.o kappa_d) ≤ lambda_k (cal(L)_d)$.
]<thm:eigenvalue-convergence>
#proof[
  Apply the convergence lemma with $mu = cal(L)_d$, $nu = mu times.o kappa_d$ and $Phi$ as above. The bounds on $(dif Phi_hash cal(L)_d)/(dif (mu times.o kappa_d))$ come from @lem:mass-preservation, giving $underline(beta) = underline(beta)_d$ and $overline(beta) = overline(beta)_d$, and the operator-norm bound $epsilon = epsilon_d$ comes from @lem:isometry. As $d -> oo$ we have $underline(beta)_d, overline(beta)_d -> 1$ and $epsilon_d -> 0$, whence the stated asymptotics.
]

== Parity of eigenfunctions

Since $Q$ is symmetric and $V(-x_1,x_2)=V(x_1,x_2)=V(x_1,-x_2)$, lower eigenfunctions of $Omega_d$, are even or odd in each coordinate:

#lemma[
  Every Neumann eigenspace of $F_d (Q,V)$ has an orthonormal basis whose elements are even/odd, odd/even, even/even or odd/odd in $(x_1,x_2)$.
] <lem:parity>
#proof[
  Let $Omega=F_d (Q,V)$ and define the reflections
  $
    r_1(x_1,x_2,w)=(-x_1,x_2,w), quad
    r_2(x_1,x_2,w)=(x_1,-x_2,w).
  $
  Since $Q$ and $V$ are invariant under both reflections, $r_i (Omega)=Omega$ for $i=1,2$. Hence the operators $R_i u := u compose r_i$ are unitary involutions on $L^2(Omega)$ and preserve $H^1(Omega)$.
  If $u$ is an eigenfunction with eigenvalue $lambda$, then for every test function $v$,
  $
    integral_Omega nabla (R_i u) dot nabla v dif z
    = integral_Omega nabla u dot nabla (R_i v) dif z
    = lambda integral_Omega u R_i v dif z
    = lambda integral_Omega (R_i u) v dif z.
  $
  Hence $R_i u$ is an eigenfunction with the same eigenvalue, so $R_i$ leaves each eigenspace invariant.

  The two reflections commute. Therefore their restrictions to any eigenspace $E_lambda$ are commuting self-adjoint operators on the finite-dimensional Hilbert space $E_lambda$, and can be simultaneously diagonalized. For a common eigenvector $u$ we have $R_i u = epsilon_i u$ with $epsilon_i in {+1,-1}$. The sign $+1$ means that $u$ is even in $x_i$, while the sign $-1$ means that $u$ is odd in $x_i$. This gives the claimed parity basis. In particular, every eigenfunction in a simple eigenspace has one of these four parity types.
]

#lemma[
  The eigenspace for $lambda_1$ does not contain odd/odd functions.
] <lem:no-odd-odd-even-even>
#proof[
  By Courant's nodal domain theorem, every eigenfunction for $lambda_1$ has at most two nodal domains.
]

We know empiricaly that the first eigenspaces are all simple, and, that the first eigenfunction is odd in $x_1$ and even in $x_2$. We will assume from now on that $phi.alt_(1, d)$ indeed has these symmetries. We will prove later on via the eigenvalue bounds, that the obtained approximation of the eigenfunction, indeed approximates the principal eigenfunction. This assumption will allow us to reduce the number of free coefficients in the MPS basis by a factor of 4. 

== MPS basis for Barrel sets <mps_basis>

The goals is to construct a set of functions ${X_(j,k) R_(j,k)}_(j,k)$ such that any linear combination of those functions

$
phi_* (x,r) = ∑_(j,k) c_(j,k) X_(j,k)(x) R_(j,k) (r),
$

satisfies $-∆ phi_* = lambda_* phi_*$ in $Omega_d$. Recall that on $Omega_d = {(x, r) in [-l_1/2, l_1/2] times [-l_2/2, l_2/2] times RR_(≥0) : r≤(1-V(x)/d)}$, the Laplacian is given by $∆_x + 4/d ∂_r^2 + 4/r ∂_r$. We choose $X_(j,k)$ and $R_(j,k)$ such that

$
-∆_x X_(j,k) = lambda_(x,j,k) X_(j,k), quad -4 (d^(-1) ∂_r^2 + r^(-1) ∂_r) R_(j, k) = lambda_(r,j,k) R_(j,k)
$

for real numbers $lambda_(x,j,k)$ and $lambda_(r,j,k)$ satisfying $lambda_(x,j,k) + lambda_(r,j,k) = lambda_*$ For $-∆_x$ the eigenfunctions are simply:

$
X_(j,k)(x) =sin(x_1 ((2j+1)pi)/l_1) cos(x_2 (2k pi)/l_2),
$

this prescribes $lambda_(x,j,k) = (((2j+1)pi)/l_1)^2 + ((2k pi)/l_2)^2$ and $lambda_(r, j, k) = lambda_* - lambda_(x,j,k)$. Notice that we already factored in the fact that $phi.alt_(1,d)$ is odd in the first and even in the second coordinate. Observe that $lambda_(r, j, k)$ can be negative, in fact in our case it is negative for most $j,k$. Consequently, $R_(j,k)$ has two branches

$
R_(j,k)(r) tilde 
cases(
  r^(-alpha_d) (I_alpha_d (beta_(d,j,k) r)) / (I_alpha_d (beta_(d,j,k))) "if" lambda_(r, j, k) < 0,
  r^(-alpha_d) (J_alpha_d (beta_(d,j,k) r)) / (J_alpha_d (beta_(d,j,k))) "if" lambda_(r, j, k) ≥ 0
),
$

where $alpha_d = (d-1)/2$ and $beta_(d,j, k) = sqrt(d/4 abs(lambda_(r,j,k)))$. The functions $I_alpha, J_alpha$ denote the (modified) Bessel functions of the first kind of order $alpha$. We defer the proof of $-∆ X_(j, k) = lambda_(x,j,k) X_(j,k)$ to  @sec:mps-radial.

For large $d$ and $r < 1$, $r^(-alpha_d)$ is very small and $I_alpha_d (beta_(d, j, k) r)$ is extremely large. Computing $R_(j, k)$ naively therefore induces fatal floating points round off errors. The solution is to work with the  _Jahnke-Emden lambda functions_ @jahnke_tables_1945

$
Lambda_alpha^J (z) &= Gamma(alpha + 1) (J_alpha (z)) / (z slash 2)^alpha = sum_(i ≥ 0) ((-1)^i)/(i! (alpha+1)_i) (z/2)^(2i), \
Lambda_alpha^I (z) &= Gamma(alpha + 1) (I_alpha (x)) / (z slash 2)^alpha = sum_(i ≥ 0) 1/(i! (alpha+1)_i) (z/2)^(2i), 
$

where $(alpha+1)_i = Gamma(alpha+i+1)/Gamma(alpha+1)$. Each is a power series in $z^2$ which can be computed recursively and converges very quickly for $r<1$. With this definition the $r^(-alpha_d)$ cancels exactly against the $r^(alpha_d)$ hidden in the Bessel function,

$
r^(-alpha_d) J_(alpha_d) (beta_(d,j,k) r) = (beta_(d,j,k) slash 2)^(alpha_d) / Gamma(alpha_d + 1) Lambda_(alpha_d)^J (beta_(d,j,k) r),
$

and likewise for the $I$-branch. The remaining constant $(beta_(d,j,k) slash 2)^(alpha_d) slash Gamma(alpha_d + 1)$ is independent of $r$ but would itself over/underflow for large $d$; it is removed by normalizing $R_(j,k)$ at $r = 1$, so that the problematic prefactor is never formed

$
R_(j,k)(r) = (Lambda_(alpha_d)^(I slash J) (beta_(d,j,k) r)) / (Lambda_(alpha_d)^(I slash J) (beta_(d,j,k))).
$

The series for $Lambda_(alpha_d)^(I slash J)$ is summed by the recurrence $rho_i = rho_(i-1) dot (minus.plus x^2 slash 4) slash (i (i + alpha_d))$, accumulated until the term falls below machine precision relative to the partial sum, and its derivative  is obtained from the analogous series with $alpha_d -> alpha_d + 1$.

In the $d = oo$ limit the lambda functions collapse to a Gaussian, which the implementation evaluates directly. Indeed, from $(alpha_d + 1)_i -> alpha_d^i$ as $d -> oo$, it follows that

$
Lambda_(alpha_d)^(I slash J) (x) -> sum_(i ≥ 0) 1/(i!) (plus.minus x^2 / (4 alpha_d))^i = exp(plus.minus x^2 / (4 alpha_d)).
$

Setting $x = beta_(d,j,k) r$ and using

$
beta_(d,j,k)^2 / (4 alpha_d) = (d abs(lambda_(r,j,k)) slash 4) / (4 dot (d-1) slash 2) = (d abs(lambda_(r,j,k))) / (8(d-1)) -> abs(lambda_(r,j,k)) / 8,
$

we obtain,

$
R_(j,k)(r) ->_(d -> oo) exp(minus.plus abs(lambda_(r,j,k)) / 8 (r^2 - 1)) = exp(- lambda_(r,j,k) / 8 (r^2 - 1)),
$

since $minus.plus abs(lambda_(r,j,k)) = - lambda_(r,j,k)$ on the $J$/$I$ branch respectively. // The same limit follows independently from the radial ODE $-4(d^(-1) R'' + r^(-1) R') = lambda_(r,j,k) R$: inserting $R = e^(c r^2)$ gives $-4(d^(-1)(2c + 4 c^2 r^2) + 2c) -> -8c = lambda_(r,j,k)$, i.e. $c = - lambda_(r,j,k) slash 8$, the discarded terms being $O(1 slash d)$.

