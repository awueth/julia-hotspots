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
    [Aknowledgements]
    )
  #lorem(80)
]

#pagebreak()


#align(center)[
  #set par(justify: false)
  #heading(
    numbering: none,
    [Abstract]
    )
  #lorem(80)
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