#import "@preview/theorion:0.6.0": *
#import "@preview/drafting:0.2.2": *
#import "@preview/cetz:0.5.2"
#import "@preview/lilaq:0.6.0" as lq

#let template(doc) = [
  #show: show-theorion
  #set page(numbering: "1")
  #set heading(numbering: "1.1")
  #set math.equation(numbering: "(1)")
  #doc
]

#let margin-note-j = margin-note.with(stroke: red, fill: red.lighten(80%))
#let inline-note-j = inline-note.with(stroke: red, fill: red.lighten(80%))

#let margin-note-a = margin-note.with(stroke: blue, fill: blue.lighten(80%))
#let inline-note-a = inline-note.with(stroke: blue, fill: blue.lighten(80%))

#let inner(a, b) = $lr(chevron.l #a, #b chevron.r)$
#let erf = $op("erf")$
#let LSE = $op("LSE")$