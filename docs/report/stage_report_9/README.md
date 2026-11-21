# 阶段报告 9 — 2026-11-21 排版与评估复核

- [详细中文报告（34 页）](stage_report_9_zh.pdf)
- [Detailed English report (38 pages)](stage_report_9_en.pdf)
- [中文导读（11 页）](summary_plain_zh.pdf)

正文与图表文字实测约 14.05–14.35 磅，小标题 14.94 磅，主标题 15.94 磅。三份报告统一页眉、标题层级、表头、图例配色和图注位置。同一组组织对照图与图注保留在同页。正文说明 P35 的跨对齐统计范围、P34 的原预测基线，以及 P38 各模型采用同一校准流程并分别在训练图像上选择规则。

## P34 缺项复核

2026-11-21 10:18 EST 核对作业 55855798_3。全层种子 1 已保存原增强读出的第 12 轮权重。带平移读出首轮触及耗时限制后退出，仅保存首轮损失。已核验 522 张开发图像的原预测基线，其中匹配 7,933 个核，发出 10,804 个预测点，145 个内部死亡核中匹配 84 个。[基线摘要与原始记录摘要值](assets/P34_SEED1_BASELINE.json)

完成原成对比较可以复用已保存的原增强读出，还需从固定的初始条件完成带平移读出的 12 轮训练，再按原协议评估。根据同组种子 19 的带平移读出耗时 6,061 秒，以及种子 1 首轮 1,400 秒的外推，CPU 作业约需 2–5 小时，另计排队时间。

## 从源码构建和验证

使用 XeLaTeX 运行以下命令。构建器逐份编译两次，统计图由 TikZ/PGFPlots 随正文生成。字体优先使用 Microsoft YaHei，缺失时使用 Fandol。公式的数学上下标按排版惯例缩小。

```text
python build_pdf.py
python verify_numbers.py
```

原始科学证据来自 `docs/results/center_diagnostics_recovery_2026-11-18/`、`docs/results/center_offset_p34_2026-11-18/`、`docs/results/alignment_p34_p36_2026-11-19/`、`docs/results/alignment_results_2026-11-20/` 和 `docs/results/calibrated_consensus_p38_2026-11-20/`。本次通过 905 项数字与固定对象检查。[字号与分页检查](QA.json) · [数字核验](NUMBER_CHECKS.json) · [图片裁切来源](assets/FIGURE_PROVENANCE.json)

Markdown 文件包含报告正文、表格与组织图，并链接到统计图的 LaTeX 源文件。
