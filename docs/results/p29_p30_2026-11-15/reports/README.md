# P20–P29 stage reports, compiled with LaTeX

- [Detailed English report](RESULTS_EN.pdf): 15 pages.
- [详细中文报告](RESULTS_ZH.pdf): 15 页。
- [中文导读](READER_ZH.pdf): 5 页。

The full English/Chinese reports cover the ten P20–P29 paired experiments and both P29 seeds. The guide explains selected recovered/lost nuclei. Seven statistical figures are native PGFPlots graphics; the microscopy panels use the existing SHA-verified image overlays. Every page was rendered and visually reviewed. No missing glyphs were found. BUILD_MANIFEST.json binds scientific inputs, TeX and PDF hashes; QA_MANIFEST.json records page and layout checks.

From the parent report directory, regenerate TeX and PDFs using an existing Pandoc and XeLaTeX installation:

```bash
python reports/build_reports.py --root .
```

Or compile the supplied TeX without Pandoc:

```bash
cd reports
xelatex -interaction=nonstopmode -halt-on-error RESULTS_EN.tex
xelatex -interaction=nonstopmode -halt-on-error RESULTS_EN.tex
xelatex -interaction=nonstopmode -halt-on-error RESULTS_ZH.tex
xelatex -interaction=nonstopmode -halt-on-error RESULTS_ZH.tex
xelatex -interaction=nonstopmode -halt-on-error READER_ZH.tex
xelatex -interaction=nonstopmode -halt-on-error READER_ZH.tex
```

Keep the parent figures directory beside reports. The Chinese documents use ctex and the TeX Live Fandol fonts. There are no package-installation steps. The dated scientific source/report snapshot is preserved; current P30 execution status is recorded in ../P30_SUBMISSION.md.
