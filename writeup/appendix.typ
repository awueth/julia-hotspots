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

== Wang-Li-Yau Constants

=== $d=oo$

*The first constant.* We want to find $C_1(t_1)$ such that

$
∫ exp(-abs(x)^2/(t_1) - epsilon V(x)) dif x ≥ 1/C_1.
$

In the core $(-pi/2, pi/2) times (-1,1)$ the potential is approximately $M/2 norm(x)^2$. Let $A := t_1^(-1) + epsilon M$, then

$
∫_((-pi/2,pi/2) times (-1,1)) exp(-A abs(x)^2) dif x
&= pi/A erf(pi/2 sqrt(A)) erf(sqrt(A)) \
&≥ pi/A erf(sqrt(M))^2

$

For $M approx 15$ and $t_1 approx 1$ we obtain

$
approx pi/16  approx 1/5.
$

Hence, $C_1 approx 5$.

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

To get an upper bound of $-L_1 G$ we lower bound $x_1 ∂_1 V$. In the core we have $x_1 ∂_1 V approx epsilon M x_1^2$. In the wings $∂_1 V$ is huge, but it does depend on $x_2$, however we claim that is much bigger that $epsilon M x_1^2$ everywhere. We obtain the bound

$
-L_1 G(t,x) ≤ ((4 a(t)^2 - 2 epsilon M a(t)) x_1^2 + 2 a(t)) G(t,x_1).
$

If we find $a$ such that $a'(t) = 4 a(t)^2 - 2 epsilon M a(t)$ and $b$ such that $b'(t) = 2a(t)$, then 

$
-L_1 G(t,x_1) ≤ (a'(t) x_1^2 + b'(t)) G(t,x_1) = ∂_t G(t,x_1)
$

and we win. The following functions should to the trick:

$
a(t) &= (epsilon M) / (2 + ((epsilon M)/alpha - 2) e^(2 epsilon M t)) \
b(t) &= epsilon M t + 1/2 ln(a(t)/alpha)
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

  Assuming the wing has length $l_"wing"$, then $X_t$ reaches the core boundary at $t = (l_"wing" - X_0) \/ nabla V$. Since $nabla V approx - epsilon 10^(7)$ this time is very small. Now do the analysis from the semester paper. This should lead a constant that is approximately the infinity norm in the core.
]

=== $d<oo$

*The second constant.*