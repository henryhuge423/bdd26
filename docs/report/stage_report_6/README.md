# Stage report 6 — 13–26 October 2026

Three companion documents cover visibility supervision, query-mask diagnostics, high-resolution adaptation, the P3–P16 point-detector studies, and the verified P17 submission.

| Document | PDF | LaTeX source |
|---|---|---|
| Detailed English report, 14 pages | [English report](stage_report_6_en.pdf) | [source](stage_report_6_en.tex) |
| 详细中文报告，14 页 | [中文详报](stage_report_6_zh.pdf) | [源码](stage_report_6_zh.tex) |
| 中文导读，7 页、7 幅图 | [中文导读](summary_plain_zh.pdf) | [源码](summary_plain_zh.tex) |

导读按六个问题展开：裁切脆弱性、查询对象流失、拟合与泛化、精度与召回、小核失败分布，以及 P16/P17 的最新进展。手绘核与精度示例均标为概念示意；数据图保留样本数与指标口径。

## Build

All three PDFs are compiled directly from LaTeX using XeLaTeX. Every chart and illustration is drawn in TikZ/PGFPlots during compilation. No externally generated PDF figures are required. Sources are self-contained within this directory. The Chinese sources use `ctex` with the Windows font set; the English source uses TeX Gyre Termes. Packages are supplied by the existing TeX Live installation.

On Windows with XeLaTeX on PATH, run `powershell -NoProfile -File build.ps1`. Else run XeLaTeX twice for each of the three main `.tex` files. The build writes compiler output under `build_logs/` and leaves the three PDFs beside the sources.

The experimental snapshot is **26 October 2026, 09:27 EDT**. P16 seed19 completed; C1/R1 reached epochs85/73. P17 array55297061 was submitted09:08:19EDT and was pending at the snapshot. The report is a dated stage record. This editorial revision does not alter training configurations or the original admission rules.

Numerical evidence is in [verified_numbers.json](verified_numbers.json); compilation, PDF metadata and hashes are in [build_verification.json](build_verification.json). All35pages were rendered and visually inspected. Relevant result records: [P16](../../results/PROPOSAL_P16_PROGRESS_2026-10-26.md), [P17](../../results/PROPOSAL_P17_SUBMISSION_2026-10-26.md).
