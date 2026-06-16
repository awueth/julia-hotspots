#import "template.typ": *

= Barrel sets <barrels>

In this section we introduce the class of _barrel sets_ which are used in @pont_convex_2024 to construct a counterexample and on which we will base our counterexample as well. 

#definition([Barrel domain])[
  Let $Q subset RR^2$ be a bounded convex domain and let $V : Q -> RR$ be a convex potential. We define the convex domain

  $ F_d (Q, V) := {(x,w) in Q times RR^(d+1) : |w| ≤ 1/2 (sqrt(d) - V(x)/sqrt(d))}, $

  which we call a barrel domain.
]

#inline-note-a[
  *outdated content:*

  The constant factor $1/4$ in the radius is completely arbitrary. In the @pont_convex_2024 it was chosen to be $1/2$, however, we chose $1/4$ since it allows us to show that eigenfunctions with eigenvalue less than $4^2$ are radial in $w$ for large $d$. With the factor $1/2$ we can only prove the same result for eigenvalues less than $2^2$. 
]

The sets barrel sets get their name from the fact that for an interval $I$ the set $F_1 (I, V)$ looks like a barell, see @fig:barell.

#figure(
  image("image.png", width: 30%),
  caption: [(Preliminary AI slop) The sets $F_1$ look like barells for $Q=[-1,1]$]
) <fig:barell>

For the purpose of this work, we use $Q = [-2 pi, 2 pi] times [-1,1]$. We also use a fixed potential $V :Q -> RR$ which is symmetric in both coordinates. We will denote the eigenvalues of the Neumann Laplacian on $F_d (Q,V)$ by $0 = lambda_(0,d) < lambda_(1,d) ≤ lambda_(2,d) ≤ ...$, with multiple eigenvalues listed separately. The corresponding eigenfunctions will be denoted by $phi.alt_(0,d), phi.alt_(1,d), phi.alt_(2,d)$ and so on.  We will refer to $(lambda_(1, d), lambda_(1, d))$ as the "first" eigenvalue and eigenfunction.

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

A direct consequence of the above lemma is that any eigenfunction with eigenvalue less than $4$ is radial in high enough dimension. 
#lemma[
  The first non-trivial eigenvalue of $F_(d) (Q, V)$ satisfies
  $
    lambda_1 < 3.9
  $
  for $d$ large enough and for our choice of $Q, V$.
]
#proof[
  Use the numerical counterexample in some way. Worst case, compute its Rayleigh quotient.
]

// #inline-note-a[
//   *How the above scales with scaling of the domain.* If we rescale $Q$ to $s Q$ and $V$ to $V_s = V(x\/s, y\/s)$, then the lower bound if the eigenfunction is not radial stays the same. However, the eigenvalue $lambda_(1,d)$ multiplies by $1/s^2$, hence the conclusion of the above lemma holds for $s Q, V_s$ whenever $lambda_(Q, V) < 4 s^2$.
// ]

We will be working exclusively with readial eigenfunctions, therefore we perform the change of variables

$
r = 2 abs(w) d^(-1/2).
$

In the $(x,r)$-coordinates, the Laplacian becomes

$
Delta_x + 4/d ∂_r^2 + 4/r ∂_r.
$

The Barrel set transforms into

$
Omega_d = {(x,r) in Q times RR_(≥ 0) : r ≤ 1 - V(x)/d}.
$

On $Omega_d$ we have the eigenvalues $0 = lambda_(0, Omega_d) < ...$. We will show that $lambda_(2, Omega_d) < 4$, therefore $lambda_(i, Omega_d) = lambda_i$ for $i=0,1,2$ and $d$ large enough.


== The effective problem in the limit of dimension

The counterexample is effectively built in the limit $d -> oo$, we first have to understand how the eigenvalue problem behaves in this limit. The sequence of principal eigenfunctions $phi_(1,d)$ of $F_d$ at dimension $d$ converges to a function $h$ that is an eigenfunction of $L = -∆ + nabla V dot nabla$ at $r=1$ and satisfyes the following initial value problem:

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

#lemma[
  Let $(X, mu)$ and $(Y, nu)$ be measure spaces, $Phi : X -> Y$. Assume that $Phi$ is approximately mass preserving, i.e. $abs((dif Phi_hash mu)/(dif nu) (y) - 1) < epsilon$ for all $y in Y$. Assume $norm(D Phi - I)_"op" < epsilon$. Then, 

  $
  (1-epsilon)^3 / (1 + epsilon) lambda_k (nu) ≤ lambda_k (mu) ≤ (1+epsilon)^3 / (1-epsilon) lambda_k (nu).
  $
]
#proof[
  Write $a_mu (u, v) = integral nabla u dot nabla v dif mu$ and $m_mu (u,v) = integral u v dif mu$.

  Let $u, v in Y$,

  $
  m_mu (u compose Phi, v compose Phi) = integral_Y u v dif (Phi_hash mu) = integral_Y u v (dif Phi_hash mu)/(dif nu) dif nu. 
  $

  Therefore, since $(1-epsilon) ≤ (dif Phi_hash mu)/(dif nu) (y) ≤ (1+epsilon)$ for all $y in Y$,

  $
  (1-epsilon) m_nu (u, v) ≤ m_mu (u compose Phi, v compose Phi) ≤ (1+epsilon) m_nu (u, v).
  $

  For the Dirichlet energy: Let $M(x) := D Phi(x) D Phi^T (x)$

  $
  a_mu (nabla (u compose Phi), nabla (v compose Phi)) 
  &= integral_X nabla f(Phi(x))^T M nabla u(Phi(x)) dif mu(x) \
  &= integral_Y nabla u(y)^T M(Phi^(-1)(y)) nabla u(y) (dif Phi_hash mu)/(dif nu) (y) dif nu(y) \
  $

  Therefore, using both assumptions, we have

  $
  (1-epsilon)^3 a_nu (u, v) ≤ a_mu (nabla (u compose Phi), nabla (v compose Phi)) ≤ (1+epsilon)^3 a_nu (u, v).
  $

  For the Rayleigh quotient it follows that

  $
  (1-epsilon)^3 / (1 + epsilon) R_nu (u) ≤ R_mu (u compose Phi) ≤ (1+epsilon)^3 / (1-epsilon) R_nu (u).
  $

  By Courant-Fischer and the fact that $Phi$ is a bijection, we have

  $
  (1-epsilon)^3 / (1 + epsilon) lambda_k (nu) ≤ lambda_k (mu) ≤ (1+epsilon)^3 / (1-epsilon) lambda_k (nu).
  $

]

#lemma[
  Let $Phi : (F_d (Q, V), dif x) -> (Q times B_(sqrt(d)/2), 1/Z_d e^(-V) dif x dif w)$ be given by $(x,w) |-> (x, (1-V(x)/d)^(-1) w)$. Then $norm(D Phi - I)_"op" = O(d^(-1/2))$ and $det D Phi approx e^(V)$

  Assume $d > V$ so that $(1-V(x)/d) > 0$ for all $x in Q$.
]
#proof[
  Let $a(x) := (1-V(x)/d)^(-1)$, then $Phi(x,w) = (x, a(x) w)$.

  $
    D Phi - I =
    mat(
      0_2, 0;
      w (nabla a)^T, (a-1) I_(d+1)
    )
  $

  therefore

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

  For the determinant: $D Phi$ is block triangular, therefore $det D Phi = (1-V(x)/d)^(-(d+1))$. Hence, 
  
  $
  (dif Phi_hash mu) / (dif nu) = e^V (1-V/d)^(d+1).
  $

  We claim that this converges to $1$ uniformly in $x$ as $d -> oo$. 
]

== Symmetry considerations

Symmetries of the potential $V$ translate to further symmetries of the eigenfunctions, this is the subject of the following lemmas.

#lemma[
  Assume that $V(-x_1,x_2)=V(x_1,x_2)=V(x_1,-x_2)$. Then every Neumann eigenspace of $F_d (Q,V)$ has an orthonormal basis whose elements are even/odd, odd/even, even/even or odd/odd in $(x_1,x_2)$.
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
  Assume that $V(-x_1,x_2)=V(x_1,x_2)=V(x_1,-x_2)$ and let $lambda_1$ be the first non-zero Neumann eigenvalue of $F_d (Q,V)$. Then the eigenspace for $lambda_1$ does not contain odd/odd functions.
] <lem:no-odd-odd-even-even>
#proof[
  By Courant's nodal domain theorem, every eigenfunction for $lambda_1$ has at most two nodal domains.
]

#lemma[
  The lowest radial eigenvalue in the odd/even sector is strictly less than the lowest radial eigenvalues of each of the other sectors.
]
#proof[
  Verify numerically.
]

#corollary[
  The first eigenspace is simple and the first eigenfunction is odd/even.
] <lem:eig-odd-even>


#inline-note-a[
  *The following is now outdated:*

  As a result of @lem:eig-radial and @lem:eig-odd-even, after the change of variables $r = 2abs(w)\/sqrt(d)$, we can restrict the eigenvalue problem to the space

  $
  Ω_d (V) = {(x,w) in [0, l_1\/2] times [0, l_2\/2] times RR_(≥0) : r ≤  1 - V(x)/d}
  $

  with Dirichlet boundary conditions at the face $x_1=0$ and Neumann boundary conditions everywhere else.
]

== MPS basis for Barrel sets <mps_basis>

We aim to construct a set of functions ${X_(j,k) R_(j,k)}_(j,k)$ such that any linear combination of those functions

$
phi_* (x,r) = ∑_(j,k) c_(j,k) X_(j,k)(x) R_(j,k) (r),
$

satisfies $-∆ phi_* = lambda_* phi_*$ in $F_d$. Under the change of variables $r = 2abs(w)\/sqrt(d)$ the Laplacian transforms into

$
∆ = ∆_x + 4/d ∂_r^2 + 4/r ∂_r.
$

We choose $X_(j,k)$ and $R_(j,k)$ such that

$
-∆_x X_(j,k) = lambda_(x,j,k) X_(j,k), quad -4 (d^(-1) ∂_r^2 + r^(-1) ∂_r) R_(j, k) = lambda_(r,j,k) R_(j,k)
$

for real numbers $lambda_(x,j,k)$ and $lambda_(r,j,k)$ satisfying $lambda_(x,j,k) + lambda_(r,j,k) = lambda_*$ For $∆_x$ the eigenfunctions are straight forward:

$
X_(j,k)(x) =sin(x_1 ((2j+1)pi)/l_1) cos(x_2 (2k pi)/l_2)
$

this prescribes $lambda_(x,j,k) = (((2j+1)pi)/l_1)^2 + ((2k pi)/l_2)^2$ and $lambda_(r, j, k) = lambda_* - lambda_(x,j,k)$. Observe that $lambda_(r, j, k)$ can be negative, in fact it is negative for most $j,k$. As a result $R_(j,k)$ has two branches

$
R_(j,k)(r) tilde 
cases(
  r^(-alpha_d) I_alpha (beta_(d,j,k) r) "if" lambda_(r, j, k) < 0,
  r^(-alpha_d) J_alpha (beta_(d,j,k) r) "if" lambda_(r, j, k) ≥ 0
),
$

where $alpha_d = (d-1)/2$ and $beta_(d,j, k) = sqrt(d/4 abs(lambda_(r,j,k)))$. See @sec:mps-radial for a proof of $-∆ X_(j, k) = lambda_(x,j,k) X_(j,k)$.
