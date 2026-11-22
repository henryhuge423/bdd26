# 阶段报告 9 — 2026-11-21 沿用阶段报告 8 版式

- [详细中文报告（32 页）](stage_report_9_zh.pdf)
- [Detailed English report (31 pages)](stage_report_9_en.pdf)
- [中文导读（11 页）](summary_plain_zh.pdf)

三份 PDF 采用 12 磅正文，沿用阶段报告 8 的 Fandol 中文字体、Latin Modern 西文字体、A4 版心、22 毫米页边距、自动目录、三线表及编号图注。详报目录按研究问题、各轮实验、综合分析与资料来源排列。

## P34 两种子结果

全层种子 1 补跑作业 55937928 已完成，两种读出各 12 轮，保存清单核验通过。带平移读出的匹配由 7,933 变为 7,909（新增 142、丢失 166），内部死亡核匹配由 84 变为 83。完整结果及原判定见项目 docs/results/alignment_p34_p36_2026-11-19/P34_RECOVERY_2026-11-21.md。

## 从源码构建和验证

使用 XeLaTeX 运行以下命令。构建器逐份编译两次，统计图由 TikZ/PGFPlots 随正文生成。需要 CTeX/Fandol 与 Latin Modern 字体。

```text
python build_pdf.py
python verify_numbers.py
```

原始科学证据来自 `docs/results/center_diagnostics_recovery_2026-11-18/`、`docs/results/center_offset_p34_2026-11-18/`、`docs/results/alignment_p34_p36_2026-11-19/`、`docs/results/alignment_results_2026-11-20/` 和 `docs/results/calibrated_consensus_p38_2026-11-20/`。本次通过 905 项数字与固定对象检查。[字号与分页检查](QA.json) · [数字核验](NUMBER_CHECKS.json) · [图片裁切来源](assets/FIGURE_PROVENANCE.json)

Markdown 文件包含报告正文、表格与组织图，并链接到统计图的 LaTeX 源文件。
