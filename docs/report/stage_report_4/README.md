# 阶段报告四

**跨尺度 Dead 增益的机制检验与 Dead 专家分支的预注册关闭**
2026-10-08；实验范围为 10 月 6–8 日的 E2a 四臂对照、DSB 四道预注册闸门、E0 推理成本实测与全仓库一致性审计。

## 阅读入口

- [中文正式版](stage_report_4_zh.pdf) · [TeX 源文件](stage_report_4_zh.tex)
- [中文导读](summary_plain_zh.pdf) · [TeX 源文件](summary_plain_zh.tex)
- [English report](stage_report_4.pdf) · [TeX source](stage_report_4.tex)
- [核验数字快照](verified_numbers.json)
- [最终构建与测试核验](build_verification.json) — 中文正式版 10 页、英文版 12 页、导读 3 页；含文件哈希
- 图表及对应数值：[E2a 对比](figs/e2a_contrasts.csv)、[闸门 B 逐批余弦汇总](figs/gate_b_cosines.csv)、[E0 成本](figs/e0_cost.csv)

正式报告解释方法、预注册规则、机制解释边界与统计限制；导读面向不熟悉项目的读者。
前三份报告的历史正文没有被改写为本轮结果。本轮没有读取任何测试折，没有新的三划分测试成绩。

## 结论及适用范围

- **E2a（四臂对照，split1 验证折，双种子）**：x1 降低 HV 门槛只复现 x2 Dead 增益的 **9.59%**（未达 50% 的换基线条件）；在原生像素面积口径配平后，x2−x1 保留全部 bPQ 代价（两个 CI 整体低于零）与大部分 Dead 增益，**strict Dead 在全部四个种子对为负**；x2 内门槛效应 +.0098 但 CI 跨零，仅为候选信号。决策：默认 `hv_min_size=30` 不变。
- **DSB（四道预注册闸门）**：A proceed（基线 x1_hv30）、C proceed（决策解耦峰值 +.0021 < +.005 关闭线）、D pass（内部漏检 Dead oracle 上限 ΔDead **+.156** ≫ +.010）；B 首跑 VOID（顺序取样仅 1 个 Dead 阳性批）后按预注册修正重跑为**真实测量且失败**——64/64 Dead 阳性批解码器复合梯度余弦均值 **+0.749**、最小 +0.594、**0/64 为负**，Dead 与常见类梯度同向。按 spec §10 预注册止损规则**关闭该研究线**：无任何 GPU 训练；分支代码在关闭后合入 master 仅作已测试工具保留。
- **E0（推理成本）**：两模型系统（x1 + x2-du2 + CPU 融合）= **6.09×** 单模型 x1（比值在各负载窗口稳定在 6.1–6.3；绝对毫秒受共享 GPU 负载污染）；x1-TTA = 4.67×。按预注册规则：EF-P2 定位为**机制证据 / 教师候选**，不是部署增益。
- **全仓库审计**：主要结果全部复现（闸门判定、6.09×、三份历史报告全部表格）；更正转录误差（bPQ .6654→.6653 等）、重建两个丢失产物（数值逐位一致）、加装 fold_guard；271 项测试通过。

限制：E2a 单划分双种子、Dead 的四个对比 CI 均跨零；闸门 B 只覆盖训练完成后的检查点；DSB 的 H2（专家分支有效性）从未受试——结论是"前提不成立"，不是"专家分支无效"；E0 为单机共享 GPU 的实际耗时（wall-clock）测量。

## 数字如何核验

没有把 `findings.md` 当成最终数值来源。`scripts/verify_report4.py` 从最原始一层独立重算：

1. 8 份 E2a 审计重推 `summary.json`（`runs/hv_threshold_controls_20261006/<arm>_seed<seed>/audit_val_fold2/eval/`）重算全部 48 个端点 + 16 个臂均值 + 24 个对比均值，与冻结快照逐位核对（容差 1e-12）；
2. 闸门 A 比值由 E2a 原始汇总重推；闸门 B v2 的均值/最小值/负批数/逐层 t 由 64×6 逐批余弦序列重算；闸门 C 峰值与闸门 D 增量由行数据重算；
3. E0 的 6.09×/4.67× 由 cost.json 逐 rep 计时按最佳吞吐规则重推；
4. 审计更正（bPQ .6653）由三份测试汇总精确平均复核；split2_seed2 重建评测与杠杆增量由重建汇总复核；
5. bootstrap 区间取自 E2a 冻结快照并记录全部来源 SHA-256（不重新抽样）。

`scripts/report4_numbers.py` 逐表打印（快照 `runs/analysis/report4_numbers.txt`）；
`scripts/make_report_figures4.py` 由冻结产物生成中英文图表及 CSV。均只读，不改实验产物。

## 复现

在仓库根目录、项目固定依赖环境下执行：

```bash
python scripts/report4_numbers.py
python scripts/verify_report4.py
python scripts/make_report_figures4.py
python -m pytest -q
```

图表以 DejaVu Sans 显示拉丁及希腊字符，中文用 Droid Sans Fallback TrueType（`--zh-font` 可换，
须含 TrueType 轮廓）。制图不评测模型、不改选择规则。

2026-10-08 全量复测：**271 passed，64 warnings**（Python 3.10、torch 2.5.1+cu124、numpy 1.23.5；未安装或升级依赖）。
本轮未触碰训练、推理与评测实现。

## PDF 构建

在本目录使用安装了 ctex、Fandol 字体与 xeCJK 的 TeX Live；各运行两遍：

```bash
pdflatex -interaction=nonstopmode -halt-on-error stage_report_4.tex
pdflatex -interaction=nonstopmode -halt-on-error stage_report_4.tex
xelatex -interaction=nonstopmode -halt-on-error stage_report_4_zh.tex
xelatex -interaction=nonstopmode -halt-on-error stage_report_4_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
```

`figs/` 为自含资源。构建时检查当次新日志中的错误、缺图、缺字与越界，不能以旧日志判断成功。
机器地址、远程构建命令与操作记录仅在本地 `ops/` 维护，不属于本报告。
