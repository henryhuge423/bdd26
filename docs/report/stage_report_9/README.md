# 阶段报告 9 — 2026-11-21 沿用阶段报告 8 版式

- [详细中文报告（32 页）](stage_report_9_zh.pdf)
- [Detailed English report (30 pages)](stage_report_9_en.pdf)
- [中文导读（11 页）](summary_plain_zh.pdf)

三份 PDF 采用 12 磅正文，沿用阶段报告 8 的 Fandol 中文字体、Latin Modern 西文字体、A4 版心、22 毫米页边距、自动目录、三线表及编号图注。详报目录按研究问题、各轮实验、综合分析与资料来源排列。

## P34 缺项复核

2026-11-21 10:18 EST 核对作业 55855798_3。全层种子 1 已保存原增强读出的第 12 轮权重。带平移读出首轮触及耗时限制后退出，仅保存首轮损失。已核验 522 张开发图像的原预测基线，其中匹配 7,933 个核，发出 10,804 个预测点，145 个内部死亡核中匹配 84 个。[基线摘要与原始记录摘要值](assets/P34_SEED1_BASELINE.json)

完成原成对比较可以复用已保存的原增强读出，还需从固定的初始条件完成带平移读出的 12 轮训练，再按原协议评估。根据同组种子 19 的带平移读出耗时 6,061 秒，以及种子 1 首轮 1,400 秒的外推，CPU 作业约需 2–5 小时，另计排队时间。

## 从源码构建和验证

使用 XeLaTeX 运行以下命令。构建器逐份编译两次，统计图由 TikZ/PGFPlots 随正文生成。需要 CTeX/Fandol 与 Latin Modern 字体。

```text
python build_pdf.py
python verify_numbers.py
```

原始科学证据来自 `docs/results/center_diagnostics_recovery_2026-11-18/`、`docs/results/center_offset_p34_2026-11-18/`、`docs/results/alignment_p34_p36_2026-11-19/`、`docs/results/alignment_results_2026-11-20/` 和 `docs/results/calibrated_consensus_p38_2026-11-20/`。本次通过 905 项数字与固定对象检查。[字号与分页检查](QA.json) · [数字核验](NUMBER_CHECKS.json) · [图片裁切来源](assets/FIGURE_PROVENANCE.json)

Markdown 文件包含报告正文、表格与组织图，并链接到统计图的 LaTeX 源文件。
