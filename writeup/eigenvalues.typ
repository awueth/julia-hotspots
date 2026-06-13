#import "template.typ": *

= Bounds for the eigenvalues

The problem with Moler-Payne and similar methods is that they do not give the position in the spectrum. 
We lower bound the first two eigenvalues using a Galerkin method and certify the bounds as described in @liu_guaranteed_2024.

== Upper Bounds

These are relatively easy to obtain, we just compute the Rayleigh-quotient of our candidates

== Lower bounds

#theorem([@liu_guaranteed_2024, Theorem 3.2])[
  Suppose that for the interpolation error $Pi_h$ it holds that $integral abs(u-Pi_h u)^2 dif x ≤ C_h^2 integral abs(nabla(u-Pi_h u))^2 dif x$ for all $u in H^1$. Then
  $
  lambda_(k, h)/(1 + lambda_(k, h) C_h^2) ≤ lambda_k.
  $
]

#lemma[
  For our problem we have
  $ C_h ≤ max_(K in cal(T)_h) e^((V_"max" - V_"min")/2) C_("FE", K) 0.1893 h. $
]
#proof[
  For Courzeix-Raviart $Pi^"CR"$ on a simplex $K$ we have $norm(u - Pi^"CR" u)_K ≤ C^"CR" (K) norm(nabla (u - Pi^"CR" u))_K$ where $C^"CR" (K) ≤ 0.1893 h_K$. Therefore, 
  $
  integral_K abs(u - Pi_h u)^2 e^(-V) dif x
  &≤ e^(-min_(x in K) V(x)) integral_K abs(u - Pi_h u)^2 dif x \
  &≤ e^(-min_(x in K) V(x)) C^"CR" (K)^2 integral_K abs(nabla (u - Pi_h u))^2 dif x \
  &≤ e^(-min_(x in K) V(x)) C^"CR" (K)^2 integral_K abs(nabla (u - Pi_h u))^2 exp(-V(x) + max_(y in K) V(y)) dif x \
  &= e^(max_(x in K) V(x) - min_(x in K) V(x)) C^"CR" (K)^2 integral_K abs(nabla (u - Pi_h u))^2 e^(-V) dif x.
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

#lemma[
  Let $overline(lambda_1) ≥ lambda_1$ then, 
  $
  lambda_1 > lambda_1^"Core"  (1- e^( 2overline(lambda_1) t) C_t^2 mu("Wing")).
  $
]

Now lets try to lower bound $lambda_2$. Let $S = op("span"){phi_1, phi_2}$ (first two nonconstant eigenfunctions)

$
lambda_2^"Core" &≤ max_(1 perp h in S)  (integral_"Core" abs(nabla h)^2 dif mu) / (integral_"Core" abs(h)^2 dif mu) \
&≤  max_(1 perp h in S) (integral_Q abs(nabla h)^2 dif mu) / (integral_Q abs(h)^2 dif mu) (integral_Q abs(h)^2 dif mu) / (integral_"Core" abs(h)^2 dif mu) \
&≤ lambda_2 max_(1 perp h in S) (integral_Q abs(h)^2 dif mu) / (integral_"Core" abs(h)^2 dif mu) \
&≤ lambda_2 1/(1 - mu("Wing")(norm(phi_1)_oo^2 + norm(phi_2)_oo^2))
$

#lemma[
  Let $overline(lambda_2) ≥ lambda_2$ then,
  $
  lambda_2 > lambda_2^"Core" (1-2 e^(2 overline(lambda_2) t) C_t^2 mu("Wing"))
  $
]

=== The bounds

#let lambda1_core_lower = 3.8
#let lambda2_core_lower = 9.5

With Courzeix-Raviart implemented using Gridap.jl we obtain

$
lambda_(1, h)^"Core" &≥ #lambda1_core_lower \
lambda_(2, h)^"Core" &≥ #lambda2_core_lower
$