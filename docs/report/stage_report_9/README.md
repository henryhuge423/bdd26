# 阶段报告 9 — 2026-11-21 修订

- [详细中文报告（46 页）](stage_report_9_zh.pdf)
- [Detailed English report (46 pages)](stage_report_9_en.pdf)
- [中文导读（14 页）](summary_plain_zh.pdf)

报告逐项解释 P15、P22、D32、D33、P34、P35、P36、D37、P38 的操作、对照、结果和研究意义。正文采用 14 点级字号，图表字号为 11–13 点，真实组织图按两列放大，另有核 10 的局部对照。

截至 2026-11-21 07:54:57 EST：本人 Expanse 队列为空，P35/P36/P38 完成；P36 保存恢复完成；P34 全层种子 1 的完整评估仍待补齐。P38 相比支持数筛选减少 676/377 个额外点，当前开发精度 71.99%/70.17%。

## 复现 PDF

安装已存在的 TeX Live/XeLaTeX，运行：

```text
python build_pdf.py
```

也可逐份执行两次 `xelatex -interaction=nonstopmode -halt-on-error stage_report_9_zh.tex`，英文和导读同理。所有统计图为随正文编译的 TikZ/PGFPlots 源码，全部必要图片位于 `assets/`。字体优先使用 Microsoft YaHei；没有时使用 TeX Live 的 Fandol 中文字体。交付 PDF 在 Windows/TeX Live 2022 下编译。

## 数字和图片来源

```text
python verify_numbers.py
```

验证使用标准 Python 库，对逐图汇总、跨实验计数、匹配变化和固定对象坐标做 896 项交叉检查。`NUMBER_CHECKS.json` 保存检查名称。`QA.json` 保存逐页渲染与字号记录。`assets/FIGURE_PROVENANCE.json` 记录 27 张裁切图的原始文件摘要与裁切坐标；裁切后的组织像素和预测覆盖与原件逐像素一致。

原始科学证据来自 `docs/results/center_diagnostics_recovery_2026-11-18/`、`docs/results/center_offset_p34_2026-11-18/`、`docs/results/alignment_p34_p36_2026-11-19/`、`docs/results/alignment_results_2026-11-20/`、`docs/results/calibrated_consensus_p38_2026-11-20/`。报告内附结果 JSON、对象表与逐图记录。全部 8 张 P38 固定图仍保存在该原科学结果目录。

Markdown 版本含相同正文、表格与组织图；统计图链接指向 LaTeX 源码，完整图形见 PDF。此次修改为报告修订。
