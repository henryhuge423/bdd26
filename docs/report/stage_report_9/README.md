# Stage nine: alignment, thresholds and nucleus-center detection

November 18–20, 2026. The report develops the P15/P22 context and D32/D33 observations, then covers P34, P35, P36 and the completed D37 diagnostic with fixed image cases and measured coordinates.

- [English report](stage_report_9_en.pdf) and [LaTeX source](stage_report_9_en.tex).
- [详细中文报告](stage_report_9_zh.pdf)与[LaTeX源码](stage_report_9_zh.tex)。
- [中文导读](summary_plain_zh.pdf)与[LaTeX源码](summary_plain_zh.tex)。
- [Raw evidence and numerical inputs](assets/P36_RESULTS.json), [D37 input hashes and results](assets/D37_RESULTS.json), [measured case responses](assets/CASE_RESPONSES.csv).
- [Numerical validation](NUMBER_CHECKS.json), [rendered-page QA](QA.json), [content hashes](SHA256SUMS.json).

Build all three PDFs with XeLaTeX, two passes, from this directory: `powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1`. On another platform, run `xelatex -interaction=nonstopmode -halt-on-error <name>.tex` twice for each report. The source uses Microsoft YaHei for Chinese and TeX Gyre Termes for English; install available equivalent fonts explicitly or change the font declarations before compiling. Statistical charts are TikZ/PGFPlots sources in `figs/`; actual saved tissue images are in `assets/`.

PDFs were compiled locally from these sources; all pages were rendered and inspected. Validation is self-reviewed; no independent language reviewer was delegated. The source evidence uses the fixed fold-1 training/development division and center matching within three pixels, with last checkpoints and two paired seeds. P34 all-layer seed 1 remains incomplete; P36 preservation recovery required no scientific recomputation.
