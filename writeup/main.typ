#import "template.typ": template, inline-note-j, margin-note-j, inline-note-a, margin-note-a
#show: template

#set document(
  title: [Low dimensional counter examples to the Hot Spots conjecture]
)

#title()

#heading(
  numbering: none,
  [How to use Typst]
)

Sorry for forcing Typst on you, I promise its super easy. There is a guide for Latex users here, the only important thing is the section on math-mode: https://typst.app/docs/guides/for-latex-users/#maths

#inline-note-j[
  loool no worries this is good!
  
  you should cite me as "de Dios", not "Pont"
]


#inline-note-a[
  My notes will be in blue.
]



#include "introduction.typ"

#include "barrel.typ"

#include "construction.typ"

#include "numerics.typ"

#include "eigenvalues.typ"

#include "pointwise.typ"

#include "certificate.typ"

#bibliography("zotero.bib")

#include "appendix.typ"