(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "booktabs"
    "amsmath"
    "listings"
    "tikz"))
 :latex)

