#import "template.typ": *

= Pointwise bounds of the eigenfunction <sec:pointwise>

In this section we show that whenever we have an approximate eigenfunction $phi.alt_*$ of $phi.alt_1$ obtained through the method of particular solutions, which has small error on the boundary conditions, then $phi.alt_*$ is close to $phi.alt$ pointwise. Let $∂Ω = Gamma_0 union Gamma_1 union Gamma_2$ where $Gamma_0 = {(x, r) in partial Ω : x_1 = 0}$, $Gamma_1 = {(x, r) in ∂Ω : |x_1|=1 or |x_2|=1 or r=0}$ and $Gamma_2 = {(x,r) in ∂Ω : r = 1-V(x)/d}$. We construct $phi.alt_*$ such that the boundary conditions on $Gamma_0 union Gamma_1$ are satisfied exactly.

==  In $d=oo$ using Wang-Li-Yau

Let $phi.alt_*$ be an approximate eigenfunction, i.e. $norm((L-lambda_*) phi.alt_*)_oo ≤ epsilon$ and $norm(phi.alt_*)_2 = 1$.

Denote by $P_t$ the semigroup for $L$, such that $∂_t P_t = -L P_t$. Let $Q_t := e^(lambda_1 t) P_t$, then $Q_t phi.alt_1 = phi.alt_1$ and $Q_t phi.alt_* -> phi.alt_1$ as $t -> oo$. Therefore, we decompose the approximation error into 

$
norm(phi.alt_* - phi.alt_1)_oo 
≤ norm(Q_t phi.alt_* - phi.alt_*)_oo + norm(Q_t phi.alt_* - phi.alt_1)_oo.
$

The first term we can bound by $∫_0^t norm(Q_s (L-lambda_1)phi.alt_*)_oo dif s$ since $∂_t Q_t = -(L-lambda_1)Q_t$. The second term we can bound by $C_t norm(phi.alt_* - phi.alt_1)_2$ after applying Wang-Li-Yau. Or even better, if we can send $t->oo$, then the second term vanishes completely.

To bound the integrand in the first term, we could use the naive bound $norm(Q_s (L-lambda_1)phi.alt_*)_oo ≤ e^(lambda_1 s) norm((L-lambda_1) phi.alt_*)_oo$ by the maximum principle. However, this is to pessimistic. Since $phi.alt_*$ is a good approximation of $phi.alt_1$ and $Q_t phi.alt_1 = phi.alt_1$, we expect $Q_t phi.alt_* approx phi.alt_*$ as well. The key is that $(L-lambda_1) phi.alt_*$ is orthogonal to $phi.alt_1$. Therefore,

$
norm(Q_s (L-lambda_1) phi.alt_*)_2
&= e^(lambda_1 s) norm(P_s (L-lambda_1) phi.alt_*)_2 \
&≤ e^(lambda_1 s) e^(-lambda_2 s) norm((L-lambda_1) phi.alt_*)_2 \
&= e^(-(lambda_2-lambda_1) s) norm((L-lambda_1) phi.alt_*)_2.
$

To get a bound in $L^oo$ we use Wang-Li-Yau, we obtain

$
norm(Q_(s_1 + s_2) (L-lambda_1) phi.alt_*)_oo 
≤ C_(s_1) norm(Q_(s_2)(L-lambda_1) phi.alt_*)_(L^2(e^(-V)))
≤ C_(s_1) e^(-(lambda_2-lambda_1) s_2) norm((L-lambda_1) phi.alt_*)_(L^2(e^(-V))).
$

The problem here is that $C_(s_1)$ has a singularity at $s_1 = 0$. A simple workaround is to use the naive maximum principle bound for small times and the above bound for later times, that is

$
norm(phi.alt_* - phi.alt_1)_oo
≤ norm((L-lambda_1) phi.alt_*)_oo ∫_0^s_1 e^(lambda_1 s) dif s + norm((L-lambda_1) phi.alt_*)_2 ∫_(s_1)^oo C_(s\/2) e^(-2(lambda_2-lambda_1) s) dif s.
$

All in all:

#theorem[
  Let $phi.alt_1$ be the first non-trivial eigenfunction of $-Delta + nabla V dot nabla$ on $Q$. Let $phi.alt_*$ be an approximation of this eigenfunction in the sense that $(L-lambda_*) phi.alt_* = 0$, for some $lambda)*$ Finally, suppose we have the eigenvalue bounds
  $
  underline(lambda_1) ≤ lambda_1 ≤ overline(lambda_1) \
  underline(lambda_2) ≤ lambda_2 ≤ overline(lambda_2).
  $

  Then,

  $
  norm(phi.alt_* - phi.alt_1)_oo 
  ≤& min_(s_1, s_2 ≥ 0) [ (norm((L-lambda_*) phi.alt_*)_oo + norm(phi.alt_*)_oo abs(lambda_1 - lambda_1^*)) (e^(s_1 overline(lambda_1))-1)/(underline(lambda_1)) \
  &+ (norm((L-lambda_*) phi.alt_*)_2 + abs(lambda_1 - lambda_1^*)) ∫_(s_1)^oo C_(s\/2) e^(-2(underline(lambda_2)-overline(lambda_1)) s) dif s], 
  $
  
  where $abs(lambda_1 - lambda_1^*) ≤ max(lambda_1^* - underline(lambda_1), overline(lambda_1) - lambda_1)$.
]


/*
Now, since $Q_t phi.alt_1 = phi.alt_1$ and by the triangle inequality, we have

$
norm(phi.alt_* - phi.alt_1)_oo 
&≤ norm(Q_t phi.alt_* - phi.alt_*)_oo + norm(Q_t phi.alt_* - phi.alt_1)_oo \
&≤ e^(lambda_1 t)(t epsilon + t abs(lambda_* - lambda_1) norm(phi.alt_*)_oo + C_t norm(phi.alt_* - phi.alt_1)_2),
$

where we used the Wang-Li-Yau inequality. We reduced the problem to showing that $phi.alt$ is close to $phi.alt_1$ in $L^2$.



observe that $∂_t Q_t = lambda_1 e^t P_t - L e^t P_t = -(L-lambda_1)Q_t$. Therefore, 

$
norm(∂_t Q_t phi.alt_*)_oo 
&= norm(Q_t (L-lambda_1) phi.alt_*)_oo \
&≤ e^(lambda_1 t) norm((L-lambda_1)phi.alt_*)_oo \
&≤ e^(lambda_1 t) (norm((L-lambda_*) phi.alt_*)_oo + abs(lambda_* - lambda_1) norm(phi.alt_*)_oo) \
&≤ e^(lambda_1 t) (epsilon + abs(lambda_* - lambda_1) norm(phi.alt_*)_oo)
$ <eq:partialQphi.alt-bound>

by the maximum principle. It follows that

$
norm(Q_t phi.alt_* - phi.alt_*)_oo 
≤ norm(Q_0 phi.alt_* - phi.alt_*)_oo +  t norm(∂_t (Q_t phi.alt_* - phi.alt_*))_oo
= t norm(∂_t Q_t phi.alt_*)_oo
≤ t e^(lambda_1 t) (epsilon + abs(lambda_* - lambda_1) norm(phi.alt_*)_oo)
$

*Can we get rid of the $e^(lambda_1 t)$ factor?* #margin-note-a[Can you please check this?] In the step $norm(Q_t (L-lambda_1) phi.alt_*)_oo ≤ e^(lambda_1 t) norm((L-lambda_1)phi.alt_*)_oo$ in @eq:partialQphi.alt-bound we only used the maximum principle, which is probably too pessimistic. Remember that $Q_t phi.alt_1 = phi.alt_1$. Since $phi.alt_*$ is a good approximation of $phi.alt_1$, we expect $Q_t phi.alt_* approx phi.alt_*$ and thus $norm(Q_t (L-lambda_1) phi.alt_*)_oo approx epsilon + abs(lambda_* - lambda_1)norm(phi.alt_*)_oo$. Let us write $phi.alt_* = sum_j c_j phi.alt_j$ with $c_j = inner(phi.alt_j, phi.alt_*)$, assume $norm(phi.alt_*)_(L^2(e^(-V))) = 1$ then $sum c_j^2 = 1$ and $c_1 ≤ 1$. Now,

$
Q_t (L-lambda_1) 
=  sum_(j≥2) e^((lambda_1-lambda_j)t) (lambda_j - lambda_1) c_j phi.alt_j.
$

By Cauchy-Schwarz we have

$
abs(sum_(j≥2) e^((lambda_1-lambda_j)t) (lambda_j - lambda_1) c_j phi.alt_j)^2
&≤ (sum_(j≥2) (lambda_j - lambda_1)^2 c_j^2) (sum_(j≥2) e^(2(lambda_1-lambda_j)t) phi.alt_j^2) \
&≤ norm((L-lambda_1) phi.alt_*)_(L^2(e^(-V)))^2 (e^(2 lambda_1 t) p_(2t) (x,x) - phi.alt_1(x)^2),
$
where $p_t (x,y) = sum_j e^(-lambda_j t) phi.alt_j (x) phi.alt_j (y)$ is the heat kernel. So

$
norm(Q_t (L-lambda_1) phi.alt_*)_oo
≤ norm((L-lambda_1) phi.alt_*)_(L^2(e^(-V))) sup_x sqrt(e^(2 lambda_1 t) p_(2t) (x,x) - phi.alt_1(x)^2).
$

Now we need a bound of $norm((L-lambda_1) phi.alt_*)_(L^2(e^(-V)))$ instead of $norm((L-lambda_1) phi.alt_*)_oo$, which is much better since the point where the residual is worst is in the wing where $e^(-V)$ is very small. 

Lets try a nicer version of the same argument. Let $f in {phi.alt_1}^perp$, then

$
norm(Q_t f)_2
= e^(lambda_1 t) norm(P_t f)_2
≤ e^(lambda_1 t) e^(-lambda_2 t) norm(f)_2
≤ e^(-(lambda_2-lambda_1) t) norm(f)_2.
$

Now, we have $norm(Q_(s + t) f)_(oo) ≤ C_s norm(Q_t f)_2 ≤ C_s e^(-(lambda_2-lambda_1)t) norm(f)_2$. Since $(L-lambda_1) phi.alt_*$ is orthogonal to $phi.alt_1$, it follows immediately that

$
norm(Q_t (L-lambda_1) phi.alt_*)_oo 
&≤ C_(t\/2) e^(-2(lambda_2-lambda_1)t) norm((L-lambda_1) phi.alt_*)_(L^2(e^(-V))). // \
// &≤ C_(t\/2) e^(-(lambda_2-lambda_1)t) (norm((L-lambda_*) phi.alt_*)_(L^2(e^(-V))) + abs(lambda_*-lambda_1))

$
*/

== Barriers for the finite dimensional case

Suppose $phi.alt_*$ is an approximation of $phi.alt_1$ in the sense that

$
-∆ phi.alt_* &= lambda_* phi.alt_* \
phi.alt_* &= 0 "on" Gamma_0 \
∂_arrow(n) phi.alt_* &= 0 "on" Gamma_1 \
norm(∂_arrow(n) phi.alt_*)_oo &≤ epsilon "on" Gamma_2.
$

We cannot find barriers for $phi.alt$ directly, instead we bound the correction (see Moler-Payne) needed for $phi.alt_*$ satisfy the boundary condition. However then the corrected approximate eigenfunction is not a perfect eigenfunction in the interior any more. Moler-Payne show that this corrected eigenfunction is close to the true eigenfunction in the $L^2$ sense. We then upgrade this $L^2$ estimate to an $L^oo$ estimate using the Wang-Li-Yau inequality.

Let $w$ be the correction we have to add to $phi.alt_*$ to satisfy the Neumann boundary condition, i.e.

$
-∆w &= 0 \
w &= 0 "on" Gamma_0 \
∂_arrow(n) w(x,r) &= -∂_arrow(n) phi.alt_*(x,r) "on" Gamma_1 union Gamma_2.
$

Now $phi.alt_* + w$ satisfies the boundary conditions exactly at the cost $-∆(phi.alt_* + w) = lambda phi.alt_* ≠ lambda_* (phi.alt_* + w)$. According to Moler-Payne it follows that 

#lemma[@moler_bounds_1968][
  $
  norm(phi.alt_*-phi.alt_1)_2 ≤ norm(w)_2/alpha (1 + norm(w)_2^2/alpha^2)^(1/2),
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
  lambda_1 ≤ integral abs(nabla phi.alt_*)^2
  &= -integral phi.alt_* ∆ phi.alt_* + integral phi.alt_* ∂_arrow(n) phi.alt_* \
  &= lambda_* + integral_Gamma_1 phi.alt_* ∂_arrow(n) phi.alt_*
  $
]

However, we want pointwise bounds and can now, since we transformed to and interior problem, use the same procedure as in the infinite dimensional case: $norm((-∆-lambda_*)(phi.alt_* + w))_oo = lambda_* norm(w)_oo$, by the same argument as before we conclude that 

#inline-note-a[
  Let's redo this.

  $
  norm(phi.alt_* + w - phi.alt_1)_oo \
  ≤ norm((-∆-lambda_1) (phi.alt_* + w))_oo ∫_0^s_1 e^(lambda_1 s) dif s + norm((-∆-lambda_1) (phi.alt_* + w))_2 ∫_(s_1)^oo C_(s\/2) e^(-2(lambda_2-lambda_1) s) dif s.
  $

  We have 
  
  $
  norm((-∆-lambda_1) (phi.alt_* + w)) 
  &≤ norm((-∆-lambda_1) phi.alt_*) + lambda_1 norm(w) \
  &≤ epsilon + abs(lambda_* - lambda) norm(phi.alt_*) + lambda_1 norm(w),
  $

  and $abs(lambda_* - lambda_1) ≤ (sqrt(2) norm(w) + norm(w)^2)/(1-norm(w)) lambda_*$. 

]

$
norm(phi.alt_* + w - phi.alt_1)_oo 
≤ e^(lambda_1 t)(lambda_* norm(w)_oo + abs(lambda_* - lambda_1) norm(phi.alt_* + w)_oo + C_t norm(phi.alt_* + w - phi.alt_1)_2) \
≤ e^(lambda_1 t)(lambda_* norm(w)_oo + abs(lambda_* - lambda_1) (norm(phi.alt_*)_oo + norm(w)_oo) + C_t (norm(phi.alt_* - phi.alt_1)_2 + norm(w)_2)) \
$







In order to bound $w$ we use a barrier estimate, let $v$ be such that

$
-∆v &≥ 0 \
v &= 0 "on" Gamma_0 \
∂_arrow(n) v &≥ 0 "on" Gamma_1 \
∂_arrow(n) v(x,r) &≥ abs(∂_arrow(n) phi.alt_*) "on" Gamma_2.
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