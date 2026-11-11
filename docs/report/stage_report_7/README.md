# Stage report 7

Three PDFs and editable XeLaTeX sources are supplied: `stage_report_7_en`, `stage_report_7_zh` and `summary_plain_zh`. The `figs/` files use TikZ/PGFPlots; `assets/` contains two predetermined original P25 result images; `evidence/` contains the machine-readable numerical summaries. The update includes the completed P26 two-seed gate, cross-round comparisons and enlarged fixed examples. `NUMBER_SOURCES.md` maps claims to evidence. Run `python verify_numbers.py` to check principal P25 numbers against the bundled files.

The original build used TeX Live 2022 with XeLaTeX, PGFPlots 1.18 and TeX Gyre Termes. The Chinese sources use `ctexart` with `fontset=windows` and the Windows Chinese fonts. On Windows, with these on PATH:

```powershell
powershell -NoProfile -File build.ps1
```

The script compiles three times into `build/` and checks for overfull boxes, missing glyphs and unresolved references. On Linux install TeX Live 2022 or a compatible newer release, CTeX, PGFPlots 1.18, TeX Gyre Termes and Chinese fonts. Change `fontset=windows` to `fontset=fandol` in the two Chinese `.tex` files if using TeX Live's Fandol fonts; recheck line breaks and glyph coverage after any font substitution. Then run, from this directory:

```sh
mkdir -p build
for doc in stage_report_7_en stage_report_7_zh summary_plain_zh; do
  for pass in 1 2 3; do
    xelatex -output-directory=build -interaction=nonstopmode -halt-on-error "$doc.tex"
  done
done
```

Inspect compiler logs and render all pages after rebuilding. The Windows script's checked build and actual P26 status are documented in the delivered report.
