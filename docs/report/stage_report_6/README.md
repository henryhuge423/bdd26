# Stage report 6 — 13–26 October 2026

本次修订面向不熟悉项目的读者：先解释研究任务和已有基础，再按问题讲实验。中文导读用实例解释召回、精度和轮廓质量；详报增加术语说明，保留数据表，把计算与实现记录移到附录。

| Document | PDF | Source |
|---|---|---|
| 中文导读，7 页，建议先读 | [中文导读](summary_plain_zh.pdf) | [LaTeX](summary_plain_zh.tex) |
| 详细中文报告，16 页 | [中文详报](stage_report_6_zh.pdf) | [LaTeX](stage_report_6_zh.tex) |
| Detailed English report, 15 pages | [English report](stage_report_6_en.pdf) | [LaTeX](stage_report_6_en.tex) |

## Latest evidence cutoff

**26 October 2026, 11:55 EDT.** P16 completed all four 130-epoch runs. The new objective found 75 versus 70 interior Dead nuclei in seed 19 and 72 versus 63 in seed 1 (145 annotated objects). Both seeds failed the original joint requirements; the fixed recipe closes. At 11:55, P17 control/convolutional seed-19 runs reached epochs 58/81 of 130; both seed-1 tasks remained queued. No completed P17 outcome is available.

This replaces the earlier report's 09:27 partial P16 / queued P17 snapshot. It uses the [verified P16 closeout](../../results/PROPOSAL_P16_RESULTS_2026-10-26.md) and existing [P17 protocol](../../results/PROPOSAL_P17_SUBMISSION_2026-10-26.md). Historical experimental numbers and admission rules are retained. Point matching is development evidence, not segmentation PQ or a new official three-split test score. No experiments were submitted or changed during this revision.

## Build

All three PDFs are compiled from LaTeX with XeLaTeX; charts and illustrations use TikZ/PGFPlots. No datasets, checkpoints or external PDF figures are needed. Chinese uses the Windows font set through ctex; English uses TeX Gyre Termes.

With XeLaTeX on PATH, run `powershell -NoProfile -File build.ps1` from this directory. The script runs three passes per document to settle contents and references, keeps compiler files in build/ and console logs in build_logs/, checks overfull boxes, missing characters and unresolved rerun requests, then copies the finished PDFs beside the sources. Elsewhere, install equivalent fonts or adjust the Chinese font set, then run XeLaTeX three times for each main source.

All 38 pages were rendered and visually checked after this revision. Delivery contains final PDFs, sources, required TeX inputs/figures and build instructions. Previous versions, renderings, numerical verification and publication receipts remain outside this directory in the owner's operations archive. Research records are indexed in [results](../../results/README.md).
