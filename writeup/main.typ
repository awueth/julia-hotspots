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

// #let ack = [
//   TODO 
// ]

// #align(center)[
//   #set par(justify: false)
//   #heading(
//     numbering: none,
//     [Acknowledgements]
//     )
//   #ack
// ]

#pagebreak()

#let abstract = [
  Rauch's hot spots conjecture asserts that the first nontrivial Neumann eigenfunction of the Laplacian attains its extrema on the boundary of the domain. The conjecture was long believed to hold for convex sets, it was recently disproved by J. de Dios in high dimension, though without any explicit dimension at which a convex counterexample appears.

  This thesis makes such a counterexample explicit. Exploiting symmetry, we reduce the high-dimensional eigenvalue problem to an effectively three-dimensional one and compute first non-trivial eigenfunction by the method of particular solutions. The computed eigenfunction visibly attains its maximum in the interior. We separate the first two eigenvalues rigorously and bound the distance to the true eigenfunction, yielding a candidate convex counterexample in roughly $10^9$ dimensions, and with it a numerical, not yet certified, upper bound on the dimension where the hot spots property first fails.
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

#include "ultracontractivity.typ"

#include "eigenvalues.typ"

#include "pointwise.typ"

#include "certificate.typ"

#heading(numbering: none)[Use of Large Language Models]

The author used large language models (LLMs) to review and correct grammar, spelling, and wording. LLMs were also used extensively to write, review and debug code. The algorithmic choices, however, are the author's own. Attempts to use LLMs for mathematical work were less successful, with the notable exception of the reduction to the core in @sec:finite-core-reduction, which was derived entirely by Claude Fable 5. In particular, we tried to improve the various ultracontractivity constants with the help of LLMs. The derived bounds often looked promising at first, but  would typically break down once implemented and computed explicitly. These constants often depend on quantities such as gradients or the curvature of the potential, which were not directly available to the LLM. In our experience, LLMs need some form of feedback loop to test against in order to make meaningful progress. A possible approach would be to specify a contract for a numerical implementation of the ultracontractivity constants, allowing the LLM to test each proposed constant within the pipeline used to compute the pointwise estimates. However, this pipeline had not yet been implemented at the time we were exploring how to derive the pointwise estimates.

#bibliography("zotero.bib")

#include "appendix.typ"