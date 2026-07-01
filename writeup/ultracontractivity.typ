#import "template.typ": *

= Ultracontractivity <sec:ultracontractivity>

#theorem[Wang's Harnack inequality @wang_logarithmic_1997][
  Let $Omega subset RR^n$ be convex. Let $f$ be a bounded continuous function on $Omega$. For any $t≥0$ we have

  $
  abs(P_t f (x))^2 ≤ exp(abs(x-y)^2/(2 t)) P_t abs(f)^2 (y),
  $

  for all $x, y in Omega$.
]<thm:wang-harnack>

#theorem[Ultracontractivity @rockner_supercontractivity_2003][
  If $(integral exp(-abs(x)^2/s) dif mu)^(-1/2) ≤ C_1$ and $norm(P_t exp (abs(x)^2 / s))_(L^(oo)) ≤ C_2$, then

  $
  norm(P_(s + t) f)_(L^oo) ≤ C_1 C_2 norm(f)_(L^2).
  $
]<thm:ultracontractivity>
#proof[

  From Wang's Harnack inequality (@thm:wang-harnack), and using $abs(x-y)^2 ≤ 2abs(x)^2 + 2abs(y)^2$ it follows that

  $
  e^(-abs(y)^2/(s)) abs(P_s f (x))^2
  ≤ e^(abs(x)^2/(s)) P_s abs(f)^2 (y)
  $

  for all $x, y in Omega$. By integrating over $y in Omega$, we obtain

  $
  abs(P_s f (x))^2 integral_Omega e^(-abs(y)^2/s) dif mu(y) ≤ e^(abs(x)^2/s) integral P_s abs(f)^2 (y) dif mu(y) = e^(abs(x)^2/s) norm(f)^2_(L^2),
  $

  or equivalently,

  $
  abs(P_s f(x)) ≤ (integral_Omega e^(-abs(y)^2/s) dif mu(y))^(-1/2) e^(abs(x)^2/s) norm(f)_(L^2).
  $

  Now, using the above and the maximum principle, 

  $
  norm(P_(s+t) f)_(L^oo)
  = norm(P_t (P_s f))_(L^oo)
  ≤ (integral_Omega e^(-abs(y)^2/s))^(-1/2) norm(f)_(L^2) norm(P_t e^(abs(x)^2/s))_(L^oo)
  ≤ C_1 C_2 norm(f)_(L^2).
  $
]

== Ultracontractivity constants at $d=oo$

The first constant we want to find is $C_1(t_1)$ such that

$
1/Z ∫_Q e^(-abs(x)^2/(t_1)) e^(- V(x)) dif x ≥ 1/C_1^2,
$

where $Z = integral_Q e^(-V) dif x$. Since $abs(x)^2$ is convex we can integrate $exp(-(abs(x)^2 + V(x)))$ the same way as we did to obtain $Z$ using interval arithmetic quadrature. 

For the second constant, that is for $C_2 (t_1, t_2)$ such that 

$
norm(P_t_2 exp(norm(x)^2/t_1))_oo ≤ C_2,
$

we will use a barrier estimate. We want to find $G(t,x)$ such that

$
∂_t G(t,x) + L G(t,x) &≥ 0  \
G(0,x) &≥ e^(alpha norm(x)^2),
$

where $alpha = t_1^(-1)$ Then we have $G(t,x) ≥ P_t e^(alpha norm(x)^2)$. First we separate out $x_2$, i.e. $P_t e^(alpha norm(x)^2) ≤ e^(alpha x_2^2) P_t e^(alpha x_1^2)$ making the problem effectively one dimensional. We now try to find a supersolution $G(t, x_1)$, i.e.

$
- L_1 G(t,x_1) &≤ ∂_t G(t,x_1)  \
G(0,x_1) &≥ e^(alpha x_1^2),
$

where $L_1 = -∂_1^2 + (∂_1 V) ∂_1$. We try the ansatz $G(t,x_1) = exp(a(t) x_1^2 + b(t))$, with $a(0) = alpha$ and $b(0) = 0$.

$
∂_t G(t,x_1) &= (a'(t) x_1^2 + b'(t))G(t,x_1) \
∂_1 G(t,x_1) &= 2 a(t)x_1 G(t,x_1) \
-L_1 G(t,x) &= (2 a(t) + 4 a(t)^2 x_1^2 - 2 a(t) x_1 ∂_1 V) G(t,x_1)
$

To get an upper bound of $-L_1 G$ we lower bound $x_1 ∂_1 V$. Let 

$
M ≤ inf_(x in Q) (∂_1 V(x))/x_1.
$

We obtain the bound

$
-L_1 G(t,x) ≤ ((4 a(t)^2 - 2 M a(t)) x_1^2 + 2 a(t)) G(t,x_1).
$

If we find $a$ such that $a'(t) = 4 a(t)^2 - 2 M a(t)$ and $b$ such that $b'(t) = 2a(t)$, then 

$
-L_1 G(t,x_1) ≤ (a'(t) x_1^2 + b'(t)) G(t,x_1) = ∂_t G(t,x_1)
$

and we win. The following functions should do the trick:

$
a(t) &= M / (2 + (M/alpha - 2) e^(2 M t)) \
b(t) &= M t + 1/2 ln(a(t)/alpha)
$

#inline-note-a[
  The problem is that with the current potential $M approx 0.1$, therefore $a(t)$ decays too slowly to beat the norm $(2 pi)^2$ at the wing end. We have to choose $t_1$ large enough for $a(t)$ to be decreasing. Therefore, we first should flow for a short time, in order to reduce the problem to the core. 

  Note that increasing the curvature in order to have a larger $M$ does not work. We pay the price by making the eigenvalue larger since the domain becomes effectively shorter.
]

#let M_calc = 0.1
#let t_1_calc = 10.0
#let alpha_calc = 1.0 / t_1_calc
#let a(t) = M_calc / (2.0 + (M_calc / alpha_calc - 2.0) * calc.exp(2.0 * M_calc * t))
#let b(t) = M_calc * t + 0.5 * calc.ln(a(t) / alpha_calc)
#let bound(t) = calc.exp(alpha_calc + (0.5 * calc.pi) * (0.5 * calc.pi) * a(t) + b(t))

#figure(
  lq.diagram(
    let ts = lq.linspace(1.0, 3.0),
    lq.plot(ts, t => bound(t))
  ),
  caption: [$C_2(t_2)$ evaluated at $t_1 = #t_1_calc$ for $M=#M_calc$ and $max x_1 = 2 pi$]
)

// #inline-note-a[
//   *A better estimate of the second constant at small times.*

//   We have

//   $
//   P_(t_2) exp(norm(x)^2/(4 t_1)) = EE[ exp(norm(X_t)^2/(4 t_1)) | X_0 = x ]
//   $

//   where $dif X_t = - nabla V(X_t) dif t + sqrt(2) dif B_t$. Int the wings $nabla V$ dominates, therefore, $dif X_t approx - nabla V(X_t) dif t$. Assume for now that $∂_y V = 0$ in the wings, then

//   $
//   X_t approx X_0 - t nabla V.
//   $

//   Assuming the wing has length $l_"wing"$, then $X_t$ reaches the core boundary at $t = (l_"wing" - X_0) \/ nabla V$. Since $nabla V approx - 10^(7)$ this time is very small. Now do the analysis from the semester paper. This should lead a constant that is approximately the infinity norm in the core.
// ]


=== Reduction to the core

We first let the drift act for a short time $tau$, so that the strong inward push in the wings carries the mass into the core before we invoke the barrier #margin-note-a[Probably just using the maximum principle instead of the barrier is better.]). Write the one-dimensional semigroup $P_t = e^(-t L_1)$ as the reflected diffusion

$
dif X_t = -∂_1 V(X_t) dif t + sqrt(2) dif B_t, quad X_t in [0, R],
$

on $[0, R]$, reflecting (Neumann) at both ends, so that $P_t f(x_0) = EE[f(X_t) | X_0 = x_0]$. Let $ell$ be the core half-width. We add a small buffer $delta > 0$ and split the interval into the buffered core

$
C_delta := [0, ell + delta]
$

and the far wing

$
W_delta := (ell + delta, R].
$

The unbuffered core is $C_0 := [0, ell]$. Since $V$ is convex and even we have $∂_1 V ≥ 0$. In the whole wing $W_0$, the potential is steep, $∂_1 V ≥ Lambda$.

Fix $0 < tau < t_2$ and let $u := P_(t_2 - tau) e^(alpha x_1^2)$. On the buffered core $u ≤ norm(u)_(L^oo (C_delta))$ and on the far wing $u ≤ e^(alpha R^2)$, hence pointwise

$
u(x_1) ≤ norm(u)_(L^oo (C_delta)) + e^(alpha R^2) bold(1)_(W_delta) (x_1).
$

Now,

$
(P_tau u) (x_0) 
&= EE_(x_0) [u(X_tau)] \
&≤ EE_(x_0) [norm(u)_(L^oo (C_delta)) + e^(alpha R^2) bold(1)_(W_delta) (x_1)] \
&= norm(u)_(L^oo (C_delta)) + e^(alpha R^2) EE_(x_0) [bold(1)_(W_delta) (x_1)] \
&= norm(u)_(L^oo (C_delta)) + e^(alpha R^2) PP_(x_0) [X_tau in W_delta]
$
 implies that

$
norm(P_(t_2) e^(alpha x_1^2))_oo 
= norm(P_(tau) u)_oo 
&≤ norm(u)_(L^oo (C_delta)) + e^(R^2) sup_(x_0) PP_(x_0)(X_tau in W_delta) \
&= norm(P_(t_2 - tau) e^(alpha x_1^2))_(L^oo (C_delta)) + e^(R^2) sup_(x_0) PP_(x_0)(X_tau in W_delta).
$ <eq:core-reduction>


It remains to see that

$
q_delta (tau) := sup_(x_0) PP_(x_0)(X_tau in W_delta)
$

is negligible. The worst start is $x_0 = R^2$. Let $sigma := inf {t : X_t ≤ ell}$ be the first entry into the unbuffered core $C_0$. We decompose according to whether the path has entered $C_0$ by time $tau$:

$
PP_(x_0)(X_tau in W_delta)
≤ PP_(x_0)(tau < sigma) + PP_(x_0)(tau ≥ sigma, X_tau in W_delta).
$

For $t ≤ min(sigma, tau)$ the path stays in $W_0$, where the drift is $≤ -Lambda$, so

$
X_t ≤ R - Lambda t + sqrt(2) B_t.
$

If ${sigma > tau}$ we have $X_tau > ell$, hence $sqrt(2) B_tau > Lambda tau - w$ with wing width $w := R - ell$. The Gaussian tail $PP(B_tau > a) ≤ e^(-a^2 slash 2 tau)$ therefore yields, for $Lambda tau > w$,

$
PP_(x_0)(tau < sigma) ≤ exp(- (Lambda tau - w)^2 / (4 tau)).
$

For the second term, after the process has entered $C_0$, ending in $W_delta$ forces it to cross the buffer of width $delta$ against the drift. By the strong Markov property at $sigma$ and comparison with the one-dimensional process $sqrt(2) B_t - Lambda t$,

$
PP_(x_0)(sigma ≤ tau, X_tau in W_delta)
≤ PP(sup_(0≤s≤tau) (sqrt(2) B_s - Lambda s) ≥ delta)
≤ exp(- Lambda delta).
$

Thus

$
q_delta (tau) ≤ exp(- (Lambda tau - w)^2 / (4 tau)) + exp(- Lambda delta).
$ <eq:core-error>


The deterministic crossing time of the wing is $w slash Lambda tilde 10^(-6)$. Choosing $tau tilde 10^(-4)$, makes $Lambda tau tilde 10^2 >> w$. Choose $delta$ so that $Lambda delta >> alpha R^2$ while $delta ≪ ell$. Then

$
e^(alpha R^2) q_delta (tau) ≤ e^(alpha R^2) (exp(-(Lambda tau - w)^2 / (4 tau)) + exp(-Lambda delta)),
$

which is negligible, while $tau ≪ t_2$ costs essentially nothing in flow time and the buffer only replaces $ell$ by $ell + delta$ in the core radius.

== Ultracontractivity constants in finite dimensions

In finite dimensions the choice of origin matters. The Wang–Harnack estimate of @thm:wang-harnack relies on the bound

$
exp(abs(x-y)^2 / (2t)) ≤ exp(abs(x)^2 / (2t)) exp(abs(y)^2 / (2t)),
$

which comes from $abs(x - y)^2 ≤ abs(x)^2 + abs(y)^2$. In infinite dimensions this is harmless, since every point has radius $r = 1$. In finite dimensions, however, the barrel has radius of order $sqrt(d) slash 2$, so applied to the radial coordinate the same bound,
$
abs((x,r) - (x',r'))^2 ≤ abs((x,r))^2 + abs((x',r'))^2,
$
costs us a factor of order $e^(sqrt(d) slash 2)$.

To avoid this, we move the origin to the boundary of the barrel, where most of the mass concentrates. We replace the radial coordinate $abs(w)$ by

$
s = sqrt(d) slash 2 - abs(w),
$

which measures the distance to the barrel's rim. The estimate
$
abs((x,s) - (x',s'))^2 ≤ abs((x,s))^2 + abs((x',s'))^2
$
still holds, but now the $s^2$ terms are small precisely where the mass lives, so the integral in @thm:ultracontractivity no longer pays the exponential price. Accordingly, we work on the set

$
S_d (Q, V) = {(x, sqrt(d) slash 2 - abs(w)) : (x, w) in F_d (Q, V)}.
$

In the new coordinate the measure becomes

$
dif mu_d
= abs(w)^d dif w dif x
= (sqrt(d) slash 2 - s)^d dif s dif x
= e^(d log(sqrt(d) slash 2 - s)) dif s dif x,
$

corresponding to the potential $W(x,s) = -d log(sqrt(d) slash 2 - s)$.

The substitution $s = sqrt(d) slash 2 - abs(w)$ leaves $Delta_x$ unchanged and flips the sign of the radial derivative, so the Laplacian reads

$
Delta_(x,s) = Delta_x + ∂_s^2 - d / (sqrt(d) slash 2 - s) ∂_s,
$

and the Neumann condition at the boundary $s = V(x) slash (2 sqrt(d))$ becomes

$
∂_arrow(n_d) u
:= 1 / (2 sqrt(d)) nabla V dot nabla_x u - ∂_s u = 0.
$

== The first constant

From $Omega_d$ to $S_d$ the map $(x, r) |-> (x, (1-r) sqrt(d) slash 2)$ is mass preserving, therefore
$
integral_(S_d) f(x,s) dif mu_(S_d) (x, s) = integral_Omega_d f(x,(1-r)sqrt(d)slash 2) dif mu_(Omega_d) (x, r).
$

It follows that $C_1^(S_d)$ is defined by

$
(integral_Omega_d exp(-1/t (abs(x)^2 + (1-r)^2 d slash 4)) dif mu_(Omega_d))^(-1/2) ≤ C_1^(S_d).
$

#lemma[
  Let $(X, mu)$ and $(Y, nu)$ be measure spaces, $Phi : X -> Y$. Assume that $Phi$ is approximately mass preserving, i.e. $abs((dif Phi_hash mu)/(dif nu) (y) - 1) < epsilon$ for all $y in Y$.  Then, 

  $
  (1-epsilon) integral_Y f dif nu ≤ integral_X f compose Phi dif mu ≤ (1+epsilon) integral_Y f dif nu.
  $
]

Now from $Omega_d$ to $Q times [0, 1]$ we have the diffeomorphism

$
Phi(x,r) = (x, a(x) r) quad "with" a(x) = (1-V(x) slash d)^(-1).
$

On $Omega_d$ the measure carries the radial weight $dif mu_(Omega_d) prop r^d dif r dif x$. Pushing forward by $Phi$, we obtain the density
$
dif Phi_hash mu_(Omega_d) prop (1 - V(x) slash d)^(d+1) r^d dif r dif x.
$

The infinite-dimensional measure on $Q times [0,1]$ has density $dif mu_(Omega_oo) prop e^(-V(x)) r^d dif r dif x$. The shared radial weight $r^d dif r$ cancels, leaving

$
(dif Phi_hash mu_(Omega_d)) / (dif mu_(Omega_oo))
= (1 - V(x) slash d)^(d+1) e^(V(x))
-> 1
$

as $d -> oo$, so $Phi$ is approximately mass preserving.

#lemma[
  $
  1-epsilon ≤ (dif Phi_hash mu_(Omega_d)) / (dif mu_(Omega_oo)) ≤ 1 + epsilon
  $
]

We can now bound $C_1^(S_d)$ in terms of the infinite-dimensional constant $C_1^oo$. Throughout, $mu_(Omega_d)$ and $mu_(Omega_oo)$ are the normalized measures, and we define the infinite-dimensional constant on the same space $Omega_oo = Q times [0,1]$,

$
(C_1^oo)^(-2) := integral_(Omega_oo) e^(-abs(x)^2 slash t) dif mu_(Omega_oo)
= (integral_Q e^(-abs(x)^2 slash t) e^(-V) dif x) / (integral_Q e^(-V) dif x),
$ <eq:c1oo>

where the radial weight $r^d dif r$ cancels between numerator and denominator. This is the first constant of the previous subsection.

Since $C_1^(S_d)$ is the smallest admissible constant, it is given by

$
(C_1^(S_d))^(-2) = integral_(Omega_d) exp(-1/t (abs(x)^2 + (1-r)^2 d slash 4)) dif mu_(Omega_d).
$

The integrand equals $f compose Phi$ where,

$
f(x, r) = exp(-1/t (abs(x)^2 + d/4 (1 - (1 - V(x) slash d) r)^2)), quad (x,r) in Q times [0,1].
$

Applying the lemma with $Phi : Omega_d -> Omega_oo$, $mu = mu_(Omega_d)$, $nu = mu_(Omega_oo)$ - admissible by the density bound just established - the lower-bound half gives

$
(C_1^(S_d))^(-2) = integral_(Omega_d) (f compose Phi) dif mu_(Omega_d) ≥ (1-epsilon) integral_(Omega_oo) f dif mu_(Omega_oo)
$ <eq:c1sd-lemma>

It remains to bound $EE[f] := integral_(Omega_oo) f dif mu_(Omega_oo)$ from below. Write 

$
f(x, r) = exp(-1/t (abs(x)^2 + d/4 u(x)^2)), quad (x,r) in Q times [0,1],
$

where $c(x)=1-V(x)/d$ and $u(x)=1-c(x) r$.

Suppose we have a uniform bound

$
EE_r [e^(-d slash (4t) u^2) | x] ≥ 1-eta_d
$

for all $x$. Then,

$
EE_(x,r)[f]
= EE_x [ e^(-abs(x)^2 slash t) EE_r [e^(-d slash (4t) u^2) | x] ]
≥ (1 - eta_d) EE_x [ e^(-abs(x)^2 slash t) ]
= (1 - eta_d)(C_1^oo)^(-2).
$



Under $mu_(Omega_oo)$ the conditional law of $r$ given $x$ is $(d+1) r^d dif r$ on $[0,1]$. Using $e^(-z) ≥ 1 - z$,

$
EE[e^(-d/(4t) u(x)^2) | x]_r ≥ 1 - d/(4t) EE_r [u(x)^2 | r].
$

Since $u = (1-r) + r V slash d$, the inequality $(a+b)^2 ≤ 2a^2 + 2b^2$ gives $u^2 ≤ 2(1-r)^2 + 2 V^2 slash d^2$, decoupling $x$ and $r$. The exact moment $EE_r [(1-r)^2 | x] = 2 slash ((d+2)(d+3))$ then yields, uniformly in $x in Q$,

$
d/(4t) EE_r [u^2 | x]
≤ d/(4t) (4/((d+2)(d+3)) + (2 V^2)/d^2)
≤ 1/(t d) (1 + norm(V)_oo^2 / 2) =: eta_d.
$

Hence $EE_r [e^(-d slash (4t) u^2) | x] ≥ 1 - eta_d$ for every $x$.



Combining with @eq:c1sd-lemma,

$
C_1^(S_d) ≤ C_1^oo / sqrt((1-epsilon)(1 - eta_d)),
quad eta_d = 1/(t d)(1 + norm(V)_oo^2 / 2).
$


== The second constant

Put $rho := abs(w)$. The shifted radial term is $sqrt(d) slash 2 - rho$. The relevant bound is

$
norm(P_t^(S_d) e^(alpha (abs(x)^2 + (sqrt(d) slash 2 - rho)^2)))_(L^oo)
≤ exp(alpha max_(x in Q) abs(x)^2)
   norm(P_t^(S_d) e^(alpha (sqrt(d) slash 2 - rho)^2))_(L^oo).
$

We construct a barrier $b(t, x, rho)$ for $e^(alpha (sqrt(d) slash 2 - rho)^2)$, i.e.

$
∂_t b - Delta_(x,rho) b &≥ 0 \
∂_arrow(n_d) b &≥ 0 \
b(0, x, rho) &≥ e^(alpha (sqrt(d) slash 2 - rho)^2).
$

The radial Laplacian in the $rho$-coordinate is

$
Delta_(x,rho) = Delta_x + ∂_rho^2 + d/rho ∂_rho.
$

Fix $T>0$, choose $alpha < beta < 2$ and $gamma > 0$. Assume

$
d ≥ (4 gamma / (beta - alpha))^2.
$

Set

$
z(t) := 1 + 4 beta t, quad
V_+ := max_(x in Q) V(x).
$

Define

$
h(t,rho)
 := exp(beta d slash 4 - gamma sqrt(d))
    z(t)^(-(d+1)/2)
    exp(- beta rho^2 / z(t)),
$

$
K_(0,d) := exp(alpha ((2 gamma) / (beta - alpha))^2),
$

$
A_(d,T) := beta d sup_(x in Q, 0≤tau≤T)
  h(tau,sqrt(d) slash 2 - V(x)/(2 sqrt(d))) / z(tau),
$

and choose $A ≥ A_(d,T)$. The barrier is

$
b(t,x,rho)
 := K_(0,d) + h(t,rho) + A/d rho^2 + 2 A (1 + 1/d)t.
$ <eq:finite-ultra-barrier>

We verify the three conditions.

*Interior equation.* The function $h$ is a constant multiple of the heat kernel in $RR^(d+1)$, hence solves

$
∂_t h = (∂_rho^2 + d/rho ∂_rho) h.
$

Also $(∂_rho^2 + d/rho ∂_rho) rho^2 = 2(d+1)$, so

$
∂_t (A/d rho^2 + 2 A (1+1/d)t)
= Delta_(x,rho)(A/d rho^2).
$

Thus $∂_t b - Delta_(x,rho) b = 0$.

*Initial condition.* Since $V ≥ 0$, we have $0≤rho≤sqrt(d) slash 2$ on the domain and hence $0≤sqrt(d) slash 2-rho≤sqrt(d) slash 2$. If $sqrt(d) slash 2-rho≤(2 gamma)/(beta - alpha)$, then the constant $K_(0,d)$ already dominates $e^(alpha (sqrt(d) slash 2-rho)^2)$. If $sqrt(d) slash 2-rho>(2 gamma)/(beta - alpha)$, write $eta := sqrt(d) slash 2-rho$. Then

$
log h(0,rho) - alpha eta^2
&= beta d slash 4 - beta rho^2 - gamma sqrt(d) - alpha eta^2 \
&= eta ((beta - alpha) sqrt(d) slash 2 + (alpha + beta) rho) - gamma sqrt(d) \
&≥ (beta - alpha) sqrt(d) slash 2 eta - gamma sqrt(d) ≥ 0.
$

Therefore $b(0,x,rho) ≥ e^(alpha (sqrt(d) slash 2-rho)^2)$.

*Boundary inequality.* On the boundary $rho=sqrt(d) slash 2 - V(x)/(2 sqrt(d))$, a function independent of $x$ satisfies $∂_arrow(n_d) b = ∂_rho b$. Therefore

$
∂_arrow(n_d) b(t,x,sqrt(d) / 2 - V(x)/(2 sqrt(d)))
&= 2 (sqrt(d) / 2 - V(x)/(2 sqrt(d))) (A/d - beta h(t,sqrt(d) / 2 - V(x)/(2 sqrt(d))) / z(t)) ≥ 0
$

by the definition of $A_(d,T)$.

The maximum principle gives

$
P_T^(S_d) e^(alpha (sqrt(d) slash 2-rho)^2) ≤ b(T,x,rho).
$

Consequently,

$
norm(P_T^(S_d) e^(alpha (sqrt(d) slash 2-rho)^2))_(L^oo)
≤ K_(0,d)
&+ 2 A (1 + 1/d) T
  + A/4 \
&+ exp(beta d slash 4 - gamma sqrt(d))
   (1 + 4 beta T)^(-(d+1)/2).
$

#corollary[
  Let $0 < alpha < 2$ and

  $
  T > (exp((alpha + 2) / 4) - 1) / (2 (alpha + 2)).
  $

  Then

  $
  norm(P_T^(S_d) e^(alpha (sqrt(d) slash 2-rho)^2))_(L^oo) -> 1 quad (d -> oo).
  $

  Consequently, the full second constant tends to $exp(alpha max_(x in Q) abs(x)^2)$.
]
#proof[
  In the estimate above choose

  $
  beta := (alpha + 2) / 2, quad gamma := d^(-1/4), quad A := A_(d,T).
  $

  Then $alpha < beta < 2$ and the assumption $d ≥ (4 gamma / (beta - alpha))^2$ holds for all sufficiently large $d$. Moreover

  $
  K_(0,d) = exp(alpha ((2 gamma) / (beta - alpha))^2) -> 1,
  $

  Next we control $A_(d,T)$. For $z=z(tau)$,

  $
  log h(tau,sqrt(d) slash 2 - V(x)/(2 sqrt(d)))
  &= beta d slash 4 - gamma sqrt(d) - (d+1)/2 log z
     - beta (sqrt(d) slash 2 - V(x)/(2 sqrt(d)))^2 / z \
  &≤ - gamma sqrt(d) + beta V_+ / 2
     + d (beta/4 (1 - 1/z) - 1/2 log z) \
  &≤ - gamma sqrt(d) + beta V_+ / 2,
  $

  where $V_+ := max_(x in Q) V(x)$ and we used $log z ≥ 1 - 1/z$ and $beta<2$. Hence

  $
  A_(d,T) ≤ beta d exp(-d^(1/4) + beta V_+ / 2) -> 0.
  $

  Finally, the assumption on $T$ is exactly

  $
  log(1 + 4 beta T) > beta / 2,
  $

  so the last term in the estimate tends to zero as well. The displayed estimate therefore gives the claimed radial limit, and the full constant follows from the initial separation of the $x$-factor.
]
