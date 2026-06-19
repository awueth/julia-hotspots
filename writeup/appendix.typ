#import "template.typ": *

#set heading(numbering: "A.1.1", supplement: [Appendix])
#counter(heading).update(0)

= Computations

== Radial part of the MPS basis <sec:mps-radial>

Let $f(z) = z^(-alpha) I_alpha (z) $

$
f' (z) = -alpha z^(-alpha-1) I_alpha (z) + z^(-alpha) I'_alpha (z) \
f''(z) = alpha (alpha + 1) z^(-alpha-2) I_alpha (z) - 2 alpha z^(-alpha-1)I'_alpha (z) + z^(-alpha) I''_(alpha) (z)
$

$
f'' (z) + (2alpha + 1)/z f'(z)
&= z^(-alpha-2) (z^2 I''_alpha (z) + z I'_alpha (z) - alpha^2 I_alpha (z)) \
&= z^(-alpha-2) (z^2 I_alpha (z)) \
&= z^(-alpha) I_alpha (z) \
&= f (z)
$

For $alpha = (d-1)/2$ this simplifies to

$
f''(z) + d/z f'(z) = f(z).
$

Now substitute $z = beta r$

$
(4/d ∂_r^2 + 4/r ∂_r) f(beta r)
&= 4/d beta^2 f''(beta r) + 4/r beta f'(r) \
&= 4/d beta^2 f''(beta r) + 4/(beta r) beta^2 f'(beta r) \
&= 4/d beta^2 (f''(beta r) + d/(beta r) f'(beta r)) \
&= 4/d beta^2 f(beta r) \
&= 4/d sqrt(d/4 abs(lambda))^2 f(beta r) \
&= -lambda f(beta r)
$

since $lambda < 0$.

For the branch $lambda ≥ 0$ the proof is analogous with $z^2 J''_alpha (z) + z J'_alpha (z) + (z^2 - alpha^2) I_alpha (z) = 0$ instead.

== Ultracontractivity

#theorem[Wang's Harnack inequality][
  Let $Omega subset RR^n$ be convex. Let $f$ be a bounded continuous function on $Omega$. For any $t≥0$ we have

  $
  abs(P_t f (x))^2 ≤ exp(abs(x-y)^2/(2 t)) P_t abs(f)^2 (y),
  $

  for all $x, y in Omega$.
]<thm:wang-harnack>

#theorem[Ultracontractivity][
  If $(integral exp(-abs(x)^2/(2s)) dif mu)^(-1/2) ≤ C_1$ and $norm(P_t exp (abs(x)^2 / (4 s)))_(L^(oo)) ≤ C_2$, then

  $
  norm(P_(s + t) f)_(L^oo) ≤ C_1 C_2 norm(f)_(L^2).
  $
]<thm:ultracontractivity>
#proof[

  From Wang's Harnack inequality (@thm:wang-harnack) it follows that

  $
  e^(-abs(y)^2/(2s)) abs(P_s f (x))^2
  ≤ e^(abs(x)^2/(2s)) P_s abs(f)^2 (y)
  $

  for all $x, y in Omega$. By integrating over $y in Omega$, we obtain

  $
  abs(P_s f (x))^2 integral_Omega e^(-abs(y)^2/(2s)) dif mu(y) ≤ e^(abs(x)^2/(2s)) integral P_s abs(f)^2 (y) dif mu(y) = e^(abs(x)^2/(2s)) norm(f)^2_(L^2),
  $

  or equivalently,

  $
  abs(P_s f(x)) ≤ (integral_Omega e^(-abs(y)^2/(2 s)) dif mu(y))^(-1/2) e^(abs(x)^2/(4s)) norm(f)_(L^2).
  $

  Now, using the above and the maximum principle, 

  $
  norm(P_(s+t) f)_(L^oo)
  = norm(P_t (P_s f))_(L^oo)
  ≤ (integral_Omega e^(-abs(y)^2/(2 s)))^(-1/2) norm(f)_(L^2) norm(P_t e^(abs(x)^2/(4s)))_(L^oo)
  ≤ C_1 C_2 norm(f)_(L^2).
  $
]

=== Ultracontractivity constants at $d=oo$

*The first constant.* We want to find $C_1(t_1)$ such that

$
∫ exp(-abs(x)^2/(t_1) - V(x)) dif x ≥ 1/C_1^2.
$

In the core $(-pi/2, pi/2) times (-1,1)$ the potential is approximately $M/2 norm(x)^2$. Let $A := t_1^(-1) + M$, then

$
∫_((-pi/2,pi/2) times (-1,1)) exp(-A abs(x)^2) dif x
&= pi/A erf(pi/2 sqrt(A)) erf(sqrt(A)) \
&≥ pi/A erf(sqrt(M))^2

$

For $M approx 15$ and $t_1 approx 1$ we obtain

$
approx pi/16  approx 1/5.
$

Hence, $C_1 approx sqrt(5)$.

*The second constant.* The goal is to find $C_2$ such that

$
norm(P_t_2 exp(norm(x)^2/(4 t_1)))_oo ≤ C_2.
$

This one we estimate with a barrier, i.e. we want to find $G(t,x)$ such that

$
∂_t G(t,x) + L G(t,x) &≥ 0  \
G(0,x) &≥ e^(alpha norm(x)^2).
$

Then we have $G(t,x) ≥ P_t e^(alpha norm(x)^2)$.

First we separate out $x_2$, i.e. $P_t e^(alpha norm(x)^2) ≤ e^(alpha x_2^2) P_t e^(alpha x_1^2)$ making the problem effectively one dimensional. We now try to find a supersolution $G(t, x_1)$, i.e.

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

To get an upper bound of $-L_1 G$ we lower bound $x_1 ∂_1 V$. In the core we have $x_1 ∂_1 V approx M x_1^2$. In the wings $∂_1 V$ is huge, but it does depend on $x_2$, however we claim that is much bigger that $M x_1^2$ everywhere. We obtain the bound

$
-L_1 G(t,x) ≤ ((4 a(t)^2 - 2 M a(t)) x_1^2 + 2 a(t)) G(t,x_1).
$

If we find $a$ such that $a'(t) = 4 a(t)^2 - 2 M a(t)$ and $b$ such that $b'(t) = 2a(t)$, then 

$
-L_1 G(t,x_1) ≤ (a'(t) x_1^2 + b'(t)) G(t,x_1) = ∂_t G(t,x_1)
$

and we win. The following functions should to the trick:

$
a(t) &= M / (2 + (M/alpha - 2) e^(2 M t)) \
b(t) &= M t + 1/2 ln(a(t)/alpha)
$

#let M_calc = 15.0
#let t_1_calc = 1.0
#let alpha_calc = 1.0 / (4.0 * t_1_calc)
#let a(t) = M_calc / (2.0 + (M_calc / alpha_calc - 2.0) * calc.exp(2.0 * M_calc * t))
#let b(t) = M_calc * t + 0.5 * calc.ln(a(t) / alpha_calc)
#let bound(t) = calc.exp(alpha_calc + 25.0 * a(t) + b(t))

#figure(
  lq.diagram(
    let ts = lq.linspace(0.1, 1.0),
    lq.plot(ts, t => bound(t))
  ),
  caption: [$C_2(t_2)$ evaluated at $t_1 = 1$ for $M=15$ and $max x_1 = 5$]
)

#inline-note-a[
  *A better estimate of the second constant at small times.*

  We have

  $
  P_(t_2) exp(norm(x)^2/(4 t_1)) = EE[ exp(norm(X_t)^2/(4 t_1)) | X_0 = x ]
  $

  where $dif X_t = - nabla V(X_t) dif t + sqrt(2) dif B_t$. Int the wings $nabla V$ dominates, therefore, $dif X_t approx - nabla V(X_t) dif t$. Assume for now that $∂_y V = 0$ in the wings, then

  $
  X_t approx X_0 - t nabla V.
  $

  Assuming the wing has length $l_"wing"$, then $X_t$ reaches the core boundary at $t = (l_"wing" - X_0) \/ nabla V$. Since $nabla V approx - 10^(7)$ this time is very small. Now do the analysis from the semester paper. This should lead a constant that is approximately the infinity norm in the core.
]

=== Ultracontractivity constants in finite dimensions

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

*The first constant.*

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


#pagebreak()

*The second constant.*

The relevant bound is now

$
norm(P_t^(S_d) e^(alpha (abs(x)^2 + abs(s)^2)))_(L^oo)
≤ (max_(x in Q) e^(alpha abs(x)^2))  norm(P_t^(S_d) e^(alpha abs(s)^2))_(L^oo).
$

The key is, that the measure $mu_d$ is concentrated near $s=0$ where $e^(alpha abs(s)^2)$ is small. In the neighborhood of $s=sqrt(d)/2$ there is almost no mass. Now we construct a barrier $b(t, x, s)$ for $e^(alpha abs(s)^2)$, i.e.

$
∂_t b - Delta_(x,s) b &≥ 0 \
∂_arrow(n_d) b &≥ 0 \
b(0, x, s) &≥ e^(alpha abs(s)^2).
$

We claim that the following function does the trick:

$
b(t, x, s) = ...
$