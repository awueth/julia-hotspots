// Standalone thesis cover page, adapted from the ETH IIS thesis template.

// ETH logo only (the lab/PULP logo from the original eth-header is removed).
#let eth-header = grid(
  columns: (auto, 1fr),
  rows: 80pt,
  align: (left + horizon, horizon),
  image("figures/eth_logo_kurz_pos.svg", height: 80%),
  [],
)

#let coverpage(
  title: none,
  author: none,
  date: datetime.today(),
  reporttype: none,
  advisors: (),
  logo: none,
) = {
  page(
    paper: "a4",
    margin: (top: 25mm, bottom: 25mm, left: 30mm, right: 30mm),
    numbering: none,
    header: eth-header,
    {
      set text(size: 12pt, lang: "en")
      show link: set text(fill: blue)

      line(length: 100%)
      v(1em)

      align(center, {
        smallcaps(text(size: 12pt)[
          Department of Mathematics
        ])
        v(2em)
        text(size: 28pt, weight: "bold", title)
        v(1em)
        smallcaps(text(size: 16pt, reporttype))
      })

      v(2em)
      if logo != none {
        align(center, logo)
      }

      v(1fr)

      align(center, {
        text(size: 18pt, author)
        linebreak()
        v(0.5em)
        text(size: 12pt, date.display("[month repr:long] [year]"))
      })

      v(1fr)

      line(length: 100%)
      v(0.5em)
      [Advisors:]
      for advisor in advisors [
        - #advisor.name
      ]
    },
  )
}
