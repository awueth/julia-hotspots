#import "template.typ": *

= Certifying the numerical counterexample <certificate>

In this section we aim to show that whenever we have an approximate eigenfunction $phi_*$ of $phi_1$ obtained through the method of particular solutions, which has small error on the boundary conditions, then $phi_*$ is close to $phi$ pointwise. The theorems used to certify this all have in common that they need some type of a priori bounds of the eigenvalues. 

== Bounds of the eigenvalue

The problem with Moler-Payne and similar methods is that they do not give the position in the spectrum. 
We first separate the first two eigenvalues using a Galerkin method and certify the bounds as described in @liu_guaranteed_2024.

#theorem([Theorem 3.2 in @liu_guaranteed_2024])[
  Suppose that for the interpolation error $Pi_h$ it holds that $integral abs(nabla (u-Pi_h u))^2 dif x ≤ C_h integral abs(u-Pi_h u)^2 dif x$ for all $u in H^1$. Then
  $
  lambda_(k, h)/(1 + lambda_(k, h) C_h^2) ≤ lambda_k.
  $
]

=== Bounding the global eigenvalues by the eigenvalues in the core

In this section we prove that we can approximate the spectrum of $L$ on $Q$ by the spectrum of $L$ restricted to a subset of $Q$ which carries most of the mass. 

Let $dif mu = 1/Z e^(-V)$, assume $norm(phi_1)_(L^2 (mu)) = 1$, then

$
lambda_1 
&= integral_Q abs(nabla phi_1)^2 dif mu \
&>= inf_f (integral_"Core" abs(nabla f)^2 dif mu) / (integral_"Core" abs(f)^2 dif mu ) integral_"Core" abs(phi_1)^2 dif mu \
&= lambda_1^"Core" integral_"Core" abs(phi_1)^2 dif mu.
$

We have

$
integral_"Core" abs(phi_1)^2 dif mu = 1 - integral_"Wing" abs(phi_1)^2 dif mu
$

The simplest bound of this is

$
integral_"Wing" abs(phi_1)^2 dif mu
≤ norm(phi_1)^2_oo mu("Wing").
$

Now we bound the infinity norm using Wang-Li-Yau: $norm(phi_1)_oo ≤ e^(lambda_1 t) C norm(phi_1)_2 =  e^(lambda_1 t) C$.

Now lets try to lower bound $lambda_2$. Let $S = op("span"){phi_1, phi_2}$ (first two nonconstant eigenfunctions)

$
lambda_2^"Core" &≤ max_(1 perp h in S)  (integral_"Core" abs(nabla h)^2 dif mu) / (integral_"Core" abs(h)^2 dif mu) \
&≤  max_(1 perp h in S) (integral_Q abs(nabla h)^2 dif mu) / (integral_Q abs(h)^2 dif mu) (integral_Q abs(h)^2 dif mu) / (integral_"Core" abs(h)^2 dif mu) \
&≤ lambda_2 max_(1 perp h in S) (integral_Q abs(h)^2 dif mu) / (integral_"Core" abs(h)^2 dif mu)
$

we have $integral_Q abs(phi_1 + phi_2)^2 dif  = $


== Pointwise bounds of the eigenfunction

Let $∂Ω = Gamma_0 union Gamma_1 union Gamma_2$ where $Gamma_0 = {(x, r) in partial Ω : x_1 = 0}$, $Gamma_1 = {(x, r) in ∂Ω : |x_1|=1 or |x_2|=1 or r=0}$ and $Gamma_2 = {(x,r) in ∂Ω : r = 1-V(x)/d}$. We construct $phi_*$ such that the boundary conditions on $Gamma_0 union Gamma_1$ are satisfied exactly.

=== Pointwise bounds of the eigenfunction in $d=oo$ using Wang-Li-Yau

Let $phi_*$ be an approximate eigenfunction, i.e. $norm((L-lambda_*) phi_*)_oo ≤ epsilon$ and $norm(phi_*)_2 = 1$.

Denote by $P_t$ the semigroup for $L$, such that $∂_t P_t = -L P_t$. Let $Q_t := e^(lambda_1 t) P_t$, then $Q_t phi_1 = phi_1$ and $Q_t phi_* -> phi_1$ as $t -> oo$. Therefore, we decompose the approximation error into 

$
norm(phi_* - phi_1)_oo 
≤ norm(Q_t phi_* - phi_*)_oo + norm(Q_t phi_* - phi_1)_oo.
$

The first term we can bound by $∫_0^t norm(Q_s (L-lambda_1)phi_*)_oo dif s$ since $∂_t Q_t = -(L-lambda_1)Q_t$. The second term we can bound by $C_t norm(phi_* - phi_1)_2$ after applying Wang-Li-Yau. Or even better, if we can send $t->oo$, then the second term vanishes completely.

To bound the first term, we could use the naive bound $norm(Q_s (L-lambda_1)phi_*)_oo ≤ e^(lambda_1 s) norm((L-lambda_1) phi_*)_oo$ by the maximum principle. However, this is to pessimistic. Since $phi_*$ is a good approximation of $phi_1$ and $Q_t phi_1 = phi_1$, we expect $Q_t phi_* approx phi_*$ as well. The key is that $(L-lambda_1) phi_*$ is orthogonal to $phi_1$. Therefore,

$
norm(Q_s (L-lambda_1) phi_*)_2
&= e^(lambda_1 s) norm(P_s (L-lambda_1) phi_*)_2 \
&≤ e^(lambda_1 s) e^(-lambda_2 s) norm((L-lambda_1) phi_*)_2 \
&= e^(-(lambda_2-lambda_1) s) norm((L-lambda_1) phi_*)_2.
$

To get a bound in $L^oo$ we use Wang-Li-Yau, we obtain

$
norm(Q_(s_1 + s_2) (L-lambda_1) phi_*)_oo 
≤ C_(s_1) norm(Q_(s_2)(L-lambda_1) phi_*)_(L^2(e^(-V)))
≤ C_(s_1) e^(-(lambda_2-lambda_1) s_2) norm((L-lambda_1) phi_*)_(L^2(e^(-V))).
$

The problem here is that $C_(s_1)$ has a singularity at $s_1 = 0$. A simple workaround is to use the naive maximum principle bound for small times and the above bound for later times, that is

$
norm(phi_* - phi_1)_oo
≤ norm((L-lambda_1) phi_*)_oo ∫_0^s_1 e^(lambda_1 s) dif s + norm((L-lambda_1) phi_*)_2 ∫_(s_1)^oo C_(s\/2) e^(-2(lambda_2-lambda_1) s) dif s.
$

#inline-note-a[
  We can get a very crude estimate by setting $s_1 = 3.0$. Then, the first integral evaluates to

  $
  ∫_0^s_1 e^(lambda_1 s) dif s
  = (e^(3lambda_1) - 1) / lambda_1 
  approx 20.
  $

  For the second integral we split $s/2 = (s-1)/2 + 0.5$ to obtain the constant
  
  $
  C_(s\/2) ≤ sqrt(M)/3  3/2 = sqrt(M)/2 approx 2
  $

  As a result, the second interval evaluates to

  $
  ∫_3^oo C_(s\/2) e^(-2(lambda_2-lambda_1) s) dif s
  ≤ 2 ∫_3^(oo) e^(-2(lambda_2-lambda_1) s) dif s
  = e^(-6(lambda_2 - lambda_1)) / (lambda_2 - lambda_1)
  approx 0.0025,
  $
  assuming $lambda_2 - lambda _1 approx 1$.

  I left a lot of slack everywhere, so we can do much better, in particular for the first constant. We should be able to get away with a significantly smaller $s_1$.
]


/*
Now, since $Q_t phi_1 = phi_1$ and by the triangle inequality, we have

$
norm(phi_* - phi_1)_oo 
&≤ norm(Q_t phi_* - phi_*)_oo + norm(Q_t phi_* - phi_1)_oo \
&≤ e^(lambda_1 t)(t epsilon + t abs(lambda_* - lambda_1) norm(phi_*)_oo + C_t norm(phi_* - phi_1)_2),
$

where we used the Wang-Li-Yau inequality. We reduced the problem to showing that $phi$ is close to $phi_1$ in $L^2$.



observe that $∂_t Q_t = lambda_1 e^t P_t - L e^t P_t = -(L-lambda_1)Q_t$. Therefore, 

$
norm(∂_t Q_t phi_*)_oo 
&= norm(Q_t (L-lambda_1) phi_*)_oo \
&≤ e^(lambda_1 t) norm((L-lambda_1)phi_*)_oo \
&≤ e^(lambda_1 t) (norm((L-lambda_*) phi_*)_oo + abs(lambda_* - lambda_1) norm(phi_*)_oo) \
&≤ e^(lambda_1 t) (epsilon + abs(lambda_* - lambda_1) norm(phi_*)_oo)
$ <eq:partialQphi-bound>

by the maximum principle. It follows that

$
norm(Q_t phi_* - phi_*)_oo 
≤ norm(Q_0 phi_* - phi_*)_oo +  t norm(∂_t (Q_t phi_* - phi_*))_oo
= t norm(∂_t Q_t phi_*)_oo
≤ t e^(lambda_1 t) (epsilon + abs(lambda_* - lambda_1) norm(phi_*)_oo)
$

*Can we get rid of the $e^(lambda_1 t)$ factor?* #margin-note-a[Can you please check this?] In the step $norm(Q_t (L-lambda_1) phi_*)_oo ≤ e^(lambda_1 t) norm((L-lambda_1)phi_*)_oo$ in @eq:partialQphi-bound we only used the maximum principle, which is probably too pessimistic. Remember that $Q_t phi_1 = phi_1$. Since $phi_*$ is a good approximation of $phi_1$, we expect $Q_t phi_* approx phi_*$ and thus $norm(Q_t (L-lambda_1) phi_*)_oo approx epsilon + abs(lambda_* - lambda_1)norm(phi_*)_oo$. Let us write $phi_* = sum_j c_j phi_j$ with $c_j = inner(phi_j, phi_*)$, assume $norm(phi_*)_(L^2(e^(-V))) = 1$ then $sum c_j^2 = 1$ and $c_1 ≤ 1$. Now,

$
Q_t (L-lambda_1) 
=  sum_(j≥2) e^((lambda_1-lambda_j)t) (lambda_j - lambda_1) c_j phi_j.
$

By Cauchy-Schwarz we have

$
abs(sum_(j≥2) e^((lambda_1-lambda_j)t) (lambda_j - lambda_1) c_j phi_j)^2
&≤ (sum_(j≥2) (lambda_j - lambda_1)^2 c_j^2) (sum_(j≥2) e^(2(lambda_1-lambda_j)t) phi_j^2) \
&≤ norm((L-lambda_1) phi_*)_(L^2(e^(-V)))^2 (e^(2 lambda_1 t) p_(2t) (x,x) - phi_1(x)^2),
$
where $p_t (x,y) = sum_j e^(-lambda_j t) phi_j (x) phi_j (y)$ is the heat kernel. So

$
norm(Q_t (L-lambda_1) phi_*)_oo
≤ norm((L-lambda_1) phi_*)_(L^2(e^(-V))) sup_x sqrt(e^(2 lambda_1 t) p_(2t) (x,x) - phi_1(x)^2).
$

Now we need a bound of $norm((L-lambda_1) phi_*)_(L^2(e^(-V)))$ instead of $norm((L-lambda_1) phi_*)_oo$, which is much better since the point where the residual is worst is in the wing where $e^(-V)$ is very small. 

Lets try a nicer version of the same argument. Let $f in {phi_1}^perp$, then

$
norm(Q_t f)_2
= e^(lambda_1 t) norm(P_t f)_2
≤ e^(lambda_1 t) e^(-lambda_2 t) norm(f)_2
≤ e^(-(lambda_2-lambda_1) t) norm(f)_2.
$

Now, we have $norm(Q_(s + t) f)_(oo) ≤ C_s norm(Q_t f)_2 ≤ C_s e^(-(lambda_2-lambda_1)t) norm(f)_2$. Since $(L-lambda_1) phi_*$ is orthogonal to $phi_1$, it follows immediately that

$
norm(Q_t (L-lambda_1) phi_*)_oo 
&≤ C_(t\/2) e^(-2(lambda_2-lambda_1)t) norm((L-lambda_1) phi_*)_(L^2(e^(-V))). // \
// &≤ C_(t\/2) e^(-(lambda_2-lambda_1)t) (norm((L-lambda_*) phi_*)_(L^2(e^(-V))) + abs(lambda_*-lambda_1))

$
*/

=== Barriers for the finite dimensional case

Suppose $phi_*$ is an approximation of $phi_1$ in the sense that

$
-∆ phi_* &= lambda_* phi_* \
phi_* &= 0 "on" Gamma_0 \
∂_arrow(n) phi_* &= 0 "on" Gamma_1 \
norm(∂_arrow(n) phi_*)_oo &≤ epsilon "on" Gamma_2.
$

We cannot find barriers for $phi$ directly, instead we bound the correction (see Moler-Payne) needed for $phi_*$ satisfy the boundary condition. However then the corrected approximate eigenfunction is not a perfect eigenfunction in the interior any more. Moler-Payne show that this corrected eigenfunction is close to the true eigenfunction in the $L^2$ sense. We then upgrade this $L^2$ estimate to an $L^oo$ estimate using the Wang-Li-Yau inequality.

Let $w$ be the correction we have to add to $phi_*$ to satisfy the Neumann boundary condition, i.e.

$
-∆w &= 0 \
w &= 0 "on" Gamma_0 \
∂_arrow(n) w(x,r) &= -∂_arrow(n) phi_*(x,r) "on" Gamma_1 union Gamma_2.
$

Now $phi_* + w$ satisfies the boundary conditions exactly at the cost $-∆(phi_* + w) = lambda phi_* ≠ lambda_* (phi_* + w)$. According to Moler-Payne it follows that 

#lemma[@moler_bounds_1968][
  $
  norm(phi_*-phi_1)_2 ≤ norm(w)_2/alpha (1 + norm(w)_2^2/alpha^2)^(1/2),
  $

  where $alpha= abs(lambda_2 - lambda_*)\/lambda_2 = 1 - lambda_* / lambda_2 ≤ 1 - lambda_* / overline(lambda_2)$. 

  $
  abs(lambda_* - lambda_1) ≤ (sqrt(2) norm(w) + norm(w)^2)/(1-norm(w)) lambda_* \
  lambda_1 ≤ lambda_* / (1-norm(w)_2)
  $
]

#inline-note-a[
  Maybe the Rayleigh-quotient is a better upper bound of $lambda_1$:
  $
  lambda_1 ≤ integral abs(nabla phi_*)^2
  &= -integral phi_* ∆ phi_* + integral phi_* ∂_arrow(n) phi_* \
  &= lambda_* + integral_Gamma_1 phi_* ∂_arrow(n) phi_*
  $
]

However, we want pointwise bounds and can now, since we transformed to and interior problem, use the same procedure as in the infinite dimensional case: $norm((-∆-lambda_*)(phi_* + w))_oo = lambda_* norm(w)_oo$, by the same argument as before we conclude that 

#inline-note-a[
  Let's redo this.

  $
  norm(phi_* + w - phi_1)_oo \
  ≤ norm((-∆-lambda_1) (phi_* + w))_oo ∫_0^s_1 e^(lambda_1 s) dif s + norm((-∆-lambda_1) (phi_* + w))_2 ∫_(s_1)^oo C_(s\/2) e^(-2(lambda_2-lambda_1) s) dif s.
  $

  We have 
  
  $
  norm((-∆-lambda_1) (phi_* + w)) 
  &≤ norm((-∆-lambda_1) phi_*) + lambda_1 norm(w) \
  &≤ epsilon + abs(lambda_* - lambda) norm(phi_*) + lambda_1 norm(w),
  $

  and $abs(lambda_* - lambda_1) ≤ (sqrt(2) norm(w) + norm(w)^2)/(1-norm(w)) lambda_*$. 

]

$
norm(phi_* + w - phi_1)_oo 
≤ e^(lambda_1 t)(lambda_* norm(w)_oo + abs(lambda_* - lambda_1) norm(phi_* + w)_oo + C_t norm(phi_* + w - phi_1)_2) \
≤ e^(lambda_1 t)(lambda_* norm(w)_oo + abs(lambda_* - lambda_1) (norm(phi_*)_oo + norm(w)_oo) + C_t (norm(phi_* - phi_1)_2 + norm(w)_2)) \
$







In order to bound $w$ we use a barrier estimate, let $v$ be such that

$
-∆v &≥ 0 \
v &= 0 "on" Gamma_0 \
∂_arrow(n) v &≥ 0 "on" Gamma_1 \
∂_arrow(n) v(x,r) &≥ abs(∂_arrow(n) phi_*) "on" Gamma_2.
$

Let $f := w - v$, then

$
-∆ f &≤ 0 \
∂_arrow(n) f &≤ 0
$

Now,

$
norm(nabla f^+)_2^2 = ∫_Ω nabla f dot nabla f^+ = ∫_Ω (-∆ f) f^+ + ∫_(Gamma_1 + Gamma_2) (∂_arrow(n) f) f^+ ≤ 0,
$

showing $f^+$ is constant. Since $f^+$ vanishes on $Gamma_0$ it must be zero everywhere, we conclude that $w ≤ v$. The same argument with $-w$ inplace of $w$ gives $abs(w) ≤ v$.