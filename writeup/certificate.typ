#import "template.typ": *

#let results_limit = toml("results/log_concave_extension/high-resolution/summary.toml")
#let results_fd_1e9 = toml("results/finite_dim/hot-spot-d1e9/summary.toml")
#let results_fd_high = toml("results/finite_dim/finite-dim/summary.toml")

#let limit_hot_spot = results_limit.mps_candidate.hot_spot_effect
#let limit_error = results_limit.pointwise_limit.pointwise_linf_bound
#let limit_residual = results_limit.pointwise_limit.residual_bound
#let limit_linf = results_limit.pointwise_limit.phi_linf_bound
#let limit_lambda_error = results_limit.pointwise_limit.lambda_error
#let limit_amplification = (
  results_limit.pointwise_limit.head_term
  + results_limit.pointwise_limit.tail_term
)
#let limit_error_budget = limit_hot_spot / (2 * limit_amplification)
#let limit_lambda_target = limit_error_budget / limit_linf
#let limit_failure_factor = 2 * limit_error / limit_hot_spot
#let limit_residual_reduction = limit_residual / limit_error_budget
#let limit_lambda_reduction = limit_lambda_error / limit_lambda_target

= Certification status and outlook <certificate>

The preceding sections provide all ingredients of a numerical counterexample: an approximate eigenfunction with a strict interior maximum, enclosures that separate the first two eigenvalues, and estimates that convert an eigenpair residual into the pointwise distance to the true eigenfunction. In this section we assemble these ingredients and address whether their present numerical values suffice for a certificate. They do not yet do so. The log-concave limit has a complete pointwise error estimate, but its error is too large. In finite dimensions, the estimate additionally depends on a barrier that has not been constructed.

Unless stated otherwise, the numerical values in this chapter are floating-point approximations and are not yet interval-arithmetic certificates.

== A sufficient condition for certification

For an approximate eigenfunction $phi.alt_*$ define its computed hot-spot effect by

$
H_* := sup_(Omega^circle) phi.alt_* - sup_(partial Omega) phi.alt_*.
$

The pointwise estimates of @thm:pointwise-limit and @thm:pointwise-finite compare $phi.alt_*$ with a suitably scaled principal eigenfunction. More precisely, suppose that, for $a = inner(phi.alt_*, phi.alt_1)$ in the log-concave limit, or for the corresponding coefficient after the harmonic correction in finite dimensions, one has

$
norm(phi.alt_* - a phi.alt_1)_oo <= E_*.
$

Then

$
sup_(Omega^circle) a phi.alt_1 - sup_(partial Omega) a phi.alt_1 >= H_* - 2 E_*.
$

After choosing the sign of $phi.alt_1$ so that $a > 0$, a strict interior maximum of the true principal eigenfunction is therefore certified whenever
$
H_* > 2 E_*.
$ <eq:certificate-criterion>

== The log-concave limit

In the limit $d -> oo$, @thm:pointwise-limit gives

$
E_* <= A_* (R_* + delta_lambda norm(phi.alt_*)_oo),
$

where

$
R_* := norm((L - lambda_*) phi.alt_*)_oo,
quad
delta_lambda := max{lambda_* - underline(lambda)_1, overline(lambda)_1 - lambda_*},
$

and $A_*$ is the sum of the short- and long-time contributions in the semigroup estimate. The full set of intermediate quantities is recorded in @tab:pointwise-limit-inputs, the values relevant to the certification criterion are summarized in @tab:limit-certificate.

#figure(
  table(
    columns: (1.5fr, 1fr),
    align: (center, left),
    [*Quantity*], [*Value*],
    [$H_*$], [$approx #num(limit_hot_spot)$],
    [$R_*$], [$lt.tilde #num(limit_residual)$],
    [$delta_lambda$], [$lt.tilde #num(limit_lambda_error)$],
    [$norm(phi.alt_*)_oo$], [$lt.tilde #num(limit_linf)$],
    [$A_*$], [$lt.tilde #num(limit_amplification)$],
    [$E_*$], [$lt.tilde #num(limit_error)$],
    [$2E_* slash H_*$], [$lt.tilde #num(limit_failure_factor)$],
  ),
  caption: [Error budget for the log-concave limit. The symbols $lt.tilde$ and $approx$ indicate that the reported values have not been enclosed using interval arithmetic.],
  kind: table,
) <tab:limit-certificate>

The decisive comparison is

$
H_* approx #num(limit_hot_spot)
quad "and" quad
2 E_* lt.tilde #num(2 * limit_error).
$

Thus the present upper bound fails @eq:certificate-criterion by a factor of approximately $num(#limit_failure_factor)$. With the present hot-spot effect and amplification factor, certification would require

$
R_* + delta_lambda norm(phi.alt_*)_oo
< #num(limit_error_budget).
$ <eq:limit-error-budget>

== Finite-dimensional barrels

We provide two finite-dimensional computations with different purposes. In each case the corresponding barrel has ambient dimension $d + 3$.

The run at $d = num(#results_fd_1e9.mps_candidate.dimension)$ demonstrates the hot-spot mechanism at the smallest dimension considered here. It gives
$
H_* approx #num(results_fd_1e9.eigenfunction.hot_spot_effect),
quad
norm(partial_arrow(n) phi.alt_*)_oo
lt.tilde #num(results_fd_1e9.residual.normal_derivative_inf).
$
The approximate eigenfunction visibly has a strict interior maximum, but the sampled residual does not identify the eigenpair's position in the spectrum and does not control the true eigenfunction pointwise. This run is therefore numerical evidence for a candidate counterexample, not a certified counterexample. // TODO: We should add some figures here to show that the construction works as intended. 

The run at $d = num(#results_fd_high.mps_candidate.dimension)$ is large enough for the comparison with the log-concave problem to yield
$
H_* approx #num(results_fd_high.eigenfunction.hot_spot_effect),
quad
norm(partial_arrow(n) phi.alt_*)_oo
lt.tilde #num(results_fd_high.residual.normal_derivative_inf),
$
together with the spectral enclosures
$
lambda_1 in #interval(
  results_fd_high.eigenvalue_bounds.lambda1_lower,
  results_fd_high.eigenvalue_bounds.lambda1_upper,
),
quad
lambda_2 >= #num(results_fd_high.eigenvalue_bounds.lambda2_lower).
$
These intervals separate the principal eigenvalue from the rest of the spectrum. For the full set of quantities entering the pointwise estimate see @tab:pointwise-finite-inputs. Nevertheless, @thm:pointwise-finite also requires a supersolution $v$ controlling the harmonic correction $w$, through
$
norm(w)_oo <= norm(v)_oo.
$
No such barrier has been obtained, so the finite-dimensional pointwise error $E_*$ cannot currently be evaluated.

== Remaining steps

The computations above leave three main difficulties: the eigenvalue enclosures are too wide for @eq:limit-error-budget, the pointwise estimates contain large ultracontractivity constants, and in finite dimensions the MPS defect occurs in the Neumann boundary condition.

The finite element bounds are necesary for locating the candidate in the spectrum, while sharper errors could come from a posteriori estimates applied to the MPS approximation. This is the approach used by @dahne_counterexample_2021 for Dirichlet boundary conditions based on @moler_bounds_1968. The obstacle in our setting are the Neumann boundary conditions, which do not allow us to bound the boundary value correction term using the maximum principle as in the Dirichlet case. A more general framework than to one in @moler_bounds_1968 can be found in  @kuttler_bounding_1978.

There are two conceptually different ways to proceed in finite dimensions. The first is to continue working directly with the finite-dimensional MPS approximation. See for instance @ennenbach_inclusion_1995 which develops @kuttler_bounding_1978 for the free membrane. Its bounds use the interior defect and the normal boundary residual, with an auxiliary Neumann--Steklov eigenvalue controlling the harmonic correction in $L^2$. This may lead to sharper eigenvalue and eigenfunction estimates, but a lower bound for the auxiliary eigenvalue would be required. The paper also treats planar analytic or polygonal domains, so its extension to the barrel or its weighted radial reduction is not immediate.

Other sources for the direct approach are @still_computable_1989 and @barnett_comparable_2018. The latter is an explicit application of the MPS on a Neumann problem in a two dimensions. However its focus are high eigenvalues, and contains a non-explicit domain constant which is only empiricaly estimated.

The second approach is to avoid the finite-dimensional MPS boundary residual altogether. One could first obtain a sufficiently accurate eigenfunction for the log-concave problem, where the Neumann condition is satisfied and only an interior residual remains, and then lift this eigenfunction to the barrel. The remainig task would be to show convergence of the finite dimensional eigenfunction to the lifted log-concave eigenfunction first in $L^2$ by concentration of the measure as we did for the eigenvalues. Then we would upgrade to a pointwise estimate using ultracontractivity.

Both approaches therefore depend on ultracontractivity constants. The constants derived here are based on several coarse estimates and are not expected to be sharp.
