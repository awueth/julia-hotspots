#import "template.typ": *

= Barrel sets <barrels>

In this section we introduce the class of _barrel sets_ which are used in @pont_convex_2024 to construct a counterexample and on which we will base our counterexample as well. 

#definition[
  Let $Q := [-l_1\/2, l_1\/2] times [-l_2\/2, l_2\/2]$ and let $V : Q -> RR$ be a convex potential. We define the convex domain

  $ F_d (Q, V) := {(x,w) in Q times RR^(d+1) : |w| ≤ 1/2 (sqrt(d) - V(x)/sqrt(d))}, $

  which we call a barrel domain.
]

The sets barrel sets get their name from the fact that for an interval $I$ the set $F_1 (I, V)$ looks like a barell, see @fig:barell.

#figure(
  image("image.png", width: 30%),
  caption: [(Preliminary AI slop) The sets $F_1$ look like barells for $Q=[-1,1]$]
) <fig:barell>

Since the boundary of $F_d (Q, V)$ is not differentiable at points $(x, w)$ where $w in ∂ Omega$, we define the Neumann eigenfunctions using the weak formulation. A non-zero function $u in H^1 (Omega)$ is a Neumann eigenfunction with eiganvalue $lambda ≥ 0$ if it satisfies the equality $integral_Omega nabla u dot nabla v dif x = lambda integral_Omega u v dif x$, for all $v in H^1 (Omega)$.

Symmetries of the potential $V$ translate to symmetries of the eigenfunctions, this is the subject of the following lemma.

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
  The eigenspaces 1 and 2 are simple.
]

#corollary[
  $phi_(1,d)$ and $phi_(2,d)$ are even/odd and odd/even
]

#lemma[
  $F_d (Q, V)$ has a spectral gap.
]
#proof[
  This we can do numerically since we know the parity of the eigenfunctions.
]

#lemma[
  The principal eigenfunction $phi_(1,d)$ on $F_d (V)$ is radial in $w$ for $d ≥ 4 norm(V)_oo$
] <lem:eig-radial>
#proof[
  In cylindrical coordinated the Lapalacian is $∆ = ∆_x +  ∂_r^2 + d/r ∂_r + 1/r^2 ∆_(S_d)$. We separate $phi_(1,d)$ into eigenfunctions of $-(∆_x +  ∂_r^2 + d/r ∂_r)$ and $-1/r^2 ∆_(S_d)$, that is

  $
  phi_(1,d) (x, abs(w)) = sum_(ell≥0, m) A_(ell, m)(x, abs(w)) Y_(ell, m) (theta),
  $
  
  where $-∆ A_(ell, m) = lambda_(ell, m) A_(ell, m)$ and $-∆ Y_(ell, m) = ell (ell + d -1) Y_(ell, m)$. Each component $A_(ell, m) Y_(ell, m)$ is itself an eigenfunction of $-∆$ corresponding to the eigenvalue $lambda_(1,d)$. Therefore, each component minimizes the Rayleigh quotient and thus
  
  $
  lambda_(1,d) = (∫ abs(nabla A_(ell, m) Y_(ell, m))^2) / (∫ abs(A_(ell, m) Y_(ell, m))^2)
  ≥ ell (ell + d - 1) (∫_Ω∫_0^(rho(x)) abs(A_(ell, m))^2 r^(d-2) dif r dif x) / (∫_Ω∫_0^(rho(x)) abs(A_(ell, m))^2 r^d dif r dif x)
  ≥ (ell (ell + d - 1)) / rho_"max"^2.
  $

  Therefore, whenever $A_(ell, m) ≠ 0$ for some $ell ≥ 1$, we have $lambda_1 ≥ d / rho_"max"^2$ and
  $rho_max = 1/2 (sqrt(d) - V(x)/sqrt(d)) ≤ 5/8 sqrt(d)$ #inline-note-a[Here I think we are giving away too much: $1/2 (sqrt(d) - V(x)/sqrt(d)) ≤ 1/2 (sqrt(d) -V(0)/sqrt(d)) approx 1/2 sqrt(d)$. Then we can conclude that if the eigenfunction is not radial then, $lambda_(1,d) ≥4$.], so $lambda_(1,d) ≥64/25 = 2.56$ whenever $A_(ell, m) ≠ 0$ for some $ell ≥ 1$. Since $lambda_(1,d) ≤ 2$ for $d$ large enough, only $A_(0,0) Y_(0,0)$ survives which is constant in $w$.
]

#inline-note-a[
  *How the above scales with scaling of the domain.* If we rescale $Q$ to $s Q$ and $V$ to $V_s = V(x\/s, y\/s)$, then the lower bound if the eigenfunction is not radial stays the same. However, the eigenvalue $lambda_(1,d)$ multiplies by $1/s^2$, hence the conclusion of the above lemma holds for $s Q, V_s$ whenever $lambda_(Q, V) < 2.5 s^2$.
]


#lemma[
  $phi_(1,d)$ is odd in $x_1$ and even in $x_2$ for $d$ large enough.
]<lem:eig-odd-even>
#proof[
  #inline-note-a[
    $phi_(1,d)$ is either symmetric or antisymmetric by symmetry of domain and potential and $phi_(1,d)$ converges to an antisymmetric function. We need an explicit dimension for which $phi_(1,d)$ is antisymmetric in $x$ and symmetric in $x$. Can we do something similar to the previous lemma, i.e. the symmetric case adds too much energy.

    $
    integral abs(nabla (X_(j, k) R_(j,k)))^2
    ≥ integral abs(nabla X_(j,k))^2 abs(R_(j,k))^2
    $
  ]
]

As a result of @lem:eig-radial and @lem:eig-odd-even, after the change of variables $r = 2abs(w)\/sqrt(d)$, we can restrict the eigenvalue problem to the space

$
Ω_d (V) = {(x,w) in [0, l_1\/2] times [0, l_2\/2] times RR_(≥0) : r ≤  1 - V(x)/d}
$

with Dirichlet boundary conditions at the face $x_1=0$ and Neumann boundary conditions everywhere else.

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
-∆_x X_(j,k) = lambda_(x,j,k) X_(j,k), quad (d^(-1) ∂_r^2 + r^(-1) ∂_r) R_(j, k) = lambda_(r,j,k) R_(j,k)
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
