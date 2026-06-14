#import "template.typ": *

= Certifying the numerical counterexample <certificate>

Here goes the conclusion of the proof. It should be some something like "if we manage to get down the pde residual to X then we have a counterexample in infinite dimension". And "if we additionally manage to build a numerical barrier at dimension D with sup norm Y then we have a counterexample at dimension D".

#figure(
  table(
    columns: 2,
    [*Quantity*], [*Floating point value*],
    [$norm((L-lambda_*) phi_*)_oo$], [max_pde_residual],
    [$norm(phi_*)_oo$], [sup_norm_eigenfunction],
    [$underline(lambda_1)$], [eigval_lower],
    [$lambda_*$], [eigval_approx],
    [$overline(lambda_1)$], [eigval_upper],
    [$underline(lambda_2)$], [4.0],
    // [$s_1$], [s1],
    // [$s_2$], [s2],
    // [$(e^(s_1 overline(lambda_1))-1)/(underline(lambda_1))$], [],
    // [$integral$], [],
    [$norm(phi_1 - phi_*)_oo$], []

  )
)