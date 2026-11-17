# LaTeX report delivery

Detailed English: 20 pages. Detailed Chinese: 19 pages. Chinese reader: 7 pages.
All 46 final pages were rendered and visually reviewed. Statistical figures use
TikZ/PGFPlots. Tissue figures preserve real original pixels and decoded coordinates.
BUILD_MANIFEST.json binds the inputs and PDFs; ../REPORT_QA.json records page checks.

From the parent directory, with existing Pandoc and XeLaTeX:

```bash
python build_latex_reports.py --root .
```

The supplied TeX can also be compiled directly without Pandoc. From this directory,
run `xelatex -interaction=nonstopmode -halt-on-error NAME.tex` twice for each of
RESULTS_EN, RESULTS_ZH and READER_ZH. Retain the sibling figures directory.
Chinese reports use ctex and the TeX Live Fandol fonts.
