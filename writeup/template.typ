#import "@preview/theorion:0.6.0": *
#import "@preview/drafting:0.2.2": *
#import "@preview/cetz:0.5.2"
#import "@preview/lilaq:0.6.0" as lq

#let template(doc) = [
  #show: show-theorion
  #set-inherited-levels(1)
  #set cite(style: "alphanumeric")
  #set page(numbering: "1")
  #set heading(numbering: "1.1")

  // 1. Reset the equation counter at every top-level heading
  #show heading.where(level: 1): it => {
    counter(math.equation).update(0)
    it
  }

  // 2. Dynamically stitch the top-level heading number onto the equation counter
  #set math.equation(numbering: num => {
    let section = counter(heading).get().first()
    numbering("(1.1)", section, num)
  })

  #doc
]

#let custom-title(prefix, title) = {
  // Catch none, empty strings, empty arrays, and empty Typst sequences
  if title == none or title == "" or title == [] or repr(title) == "sequence()" {
    prefix
  } else {
    // If a title exists, append it with the bold weight stripped
    [#prefix #strong(delta: -300)[(#title)]]
  }
}

#let definition = definition.with(get-full-title: custom-title)
#let theorem = theorem.with(get-full-title: custom-title)
#let lemma = lemma.with(get-full-title: custom-title)
#let conjecture = conjecture.with(get-full-title: custom-title)

#let margin-note-j = margin-note.with(stroke: red, fill: red.lighten(80%))
#let inline-note-j = inline-note.with(stroke: red, fill: red.lighten(80%))

#let margin-note-a = margin-note.with(stroke: blue, fill: blue.lighten(80%))
#let inline-note-a = inline-note.with(stroke: blue, fill: blue.lighten(80%))

#let inner(a, b) = $lr(chevron.l #a, #b chevron.r)$
#let erf = $op("erf")$
#let LSE = $op("LSE")$