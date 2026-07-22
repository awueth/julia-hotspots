#import "template.typ": *
#let results = toml("results/log_concave_extension/high-resolution/summary.toml")
#let results_fd = toml("results/finite_dim/finite-dim/summary.toml")

= Certifying the numerical counterexample and further work <certificate>

*$d=oo$*

For the log-concave extension we can compute the pointwise approximation error, the intermediate values are in the table below:

#figure[
  #let linf_residual = results.pointwise_limit.residual_bound
  #let linf_eigenf = results.pointwise_limit.phi_linf_bound
  #let eigval_lower = results.eigenvalue_bounds.lambda1_lower
  #let eigval_approx = results.mps_candidate.lambda
  #let eigval_upper = results.eigenvalue_bounds.lambda1_upper
  #let eigval2_lower = results.eigenvalue_bounds.lambda2_lower
  #let s1 = results.pointwise_limit.s1
  #let alpha_time = results.pointwise_limit.alpha
  #let pointwise_distance = results.pointwise_limit.pointwise_linf_bound
  #let hot_spot = results.mps_candidate.hot_spot_effect

  #table(
    columns: 2,
    align: (center, left),
    [*Quantity*], [*Value*],
    [$norm((L-lambda_*) phi.alt_*)_oo$], [$lt.tilde #num(linf_residual)$],
    [$norm(phi.alt_*)_oo$], [$≤ #num(linf_eigenf)$], // We also have the lower bound so we should put the interval here
    [$sup_(Omega^circle) phi.alt_* - sup_(∂ Omega) phi.alt_*$], [$approx #num(hot_spot)$],
    [$lambda_1$], [$in interval(#eigval_lower, #eigval_upper)$],
    [$lambda_*$], [$=#eigval_approx$],
    [$lambda_2$], [$gt.tilde num(#eigval2_lower)$],
    [$s_1$], [$= #s1$],
    [$alpha$], [$= #alpha_time$],
    [$(e^(s_1 lambda_1) - 1) / (lambda_1)$], [$≤ num(#results.pointwise_limit.head_term)$],
    [$integral_(s_1)^(oo) C_(alpha s) e^((lambda_1 - lambda_2 (1 - alpha))s) dif s$], [$≤ num(#results.pointwise_limit.tail_term)$],
    [$norm(phi_1 - phi_*)_oo$], [$lt.tilde num(#pointwise_distance)$]

  )
]

This is not enough to verify a counterexample, for this we would need the pde residual to come down to X and the eigenvalue error to Y.

*$d = 10^9$*

At this dimension we can observe the hot spot effect and visualized it. However, we cannot separate the eigenvalues because they only converge at rate $O(d^(-1 slash 2))$.

*$d = 10^18$*

We do not have a barrier so we cannot deduce the pointwise error but here are the intermediate values.

#figure[
  #let linf_residual = results_fd.residual.normal_derivative_inf
  #let linf_eigenf = results_fd.eigenfunction.sampled_linf
  #let eigval_lower = results_fd.eigenvalue_bounds.lambda1_lower
  #let eigval_approx = results_fd.mps_candidate.lambda
  #let eigval_upper = results_fd.eigenvalue_bounds.lambda1_upper
  #let eigval2_lower = results_fd.eigenvalue_bounds.lambda2_lower
  #let s1 = results_fd.pointwise.s1
  #let alpha_time = results_fd.pointwise.alpha
  #let hot_spot = results_fd.eigenfunction.hot_spot_effect

  #table(
    columns: 2,
    align: (center, left),
    [*Quantity*], [*Value*],
    [$norm(∂_arrow(n) phi.alt_*)_oo$], [$lt.tilde #num(linf_residual)$],
    [$norm(phi.alt_*)_oo$], [$lt.tilde #num(linf_eigenf)$],
    [$sup_(Omega^circle) phi.alt_* - sup_(∂ Omega) phi.alt_*$], [$approx #num(hot_spot)$],
    [$lambda_1$], [$in interval(#eigval_lower, #eigval_upper)$],
    [$lambda_*$], [$= #eigval_approx$],
    [$lambda_2$], [$in [num(#eigval2_lower), oo)$],
    [$s_1$], [$= #s1$],
    [$alpha$], [$= #alpha_time$],
    [$(e^(s_1 lambda_1) - 1) / (lambda_1)$], [$≤ num(#results_fd.pointwise.head_term)$],
    [$integral_(s_1)^(oo) C_(alpha s) e^((lambda_1 - lambda_2 (1 - alpha))s) dif s$], [$≤ num(#results_fd.pointwise.tail_term)$]
  )
]

== Further work

This subsection should make the following points:

- The eigenvalue enclosures are good enough to separate the eigenvalues, but not good enough for a certificate. We should only use the FEM eigenvalues to determine the position in the spectrum. Once this is established it should be possible the prove much tighter enclosures from the MPS-residual.

- We only know how to separate the eigenvalues in high dimensions by using the eigenvalues from the log-concave extension and the convergence of the measures. Due to the reliance on measure convergence we already need $d tilde 10^18$. In this dimension we should be able to prove that the lift of the eigenfunction in the log-concave problem is close to the eigenfunction in the finite dimensional problem in the $L^2$ sense. To upgrade to a pointwise bounds we would still need ultracontractivity, but this time the constant would multiply against a quantity that decays in $d$. We would no longer rely on the barrier we do not know how to construct to transfer to an interior problem. 
