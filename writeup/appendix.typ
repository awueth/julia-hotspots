#import "template.typ": *

#counter(heading).update(0)
#set heading(numbering: "A.1")
#set-theorion-numbering("A.1")

#set math.equation(numbering: num => {
    let section = counter(heading).get().first()
    numbering("(A.1)", section, num)
  })

= Supporting Computations

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