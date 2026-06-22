#import "template.typ": template, inline-note-j, margin-note-j, inline-note-a, margin-note-a
#import "coverpage.typ": coverpage

#set document(
  title: [Low dimensional counter examples to the Hot Spots conjecture]
)

#coverpage(
  title: [Numerical counter examples to the convex hot spots conjecture],
  author: "Adrian Wüthrich",
  reporttype: "Master Thesis",
  advisors: (
    (name: "Dr. Jaume de Dios"),
    (name: "Prof. Dr. Svitlana Mayboroda"),
  ),
)

#show: template

#align(center)[
  #set par(justify: false)
  #heading(
    numbering: none,
    [Acknowledgements]
    )
  #lorem(80)
]

#pagebreak()

#let abstract = [
  Rauch's hot spots conjecture asserts that the first nontrivial Neumann eigenfunction of the Laplacian attains its extrema on the boundary of the domain. The conjecture was long believed to hold for convex sets, it was recently disproved by J. de Dios in high dimension, though without any explicit dimension at which a convex counterexample appears.

  This thesis makes such a counterexample explicit. Exploiting symmetry, we reduce the high-dimensional eigenvalue problem to an effectively three-dimensional one and compute its principal eigenfunction by the method of particular solutions. The computed eigenfunction visibly attains its maximum in the interior. We separate the first two eigenvalues rigorously and bound the distance to the true eigenfunction, yielding a candidate convex counterexample in roughly a $10^9$ dimensions, and with it a numerical, not yet certified, upper bound on the dimension where the hot spots property first fails.
]

#align(center)[
  #set par(justify: false)
  #heading(
    numbering: none,
    [Abstract]
    )
  #abstract
]

#pagebreak()


#include "introduction.typ"

#include "barrel.typ"

#include "construction.typ"

#include "numerics.typ"

#include "eigenvalues.typ"

#include "pointwise.typ"

#include "certificate.typ"

#bibliography("zotero.bib")

#include "appendix.typ"