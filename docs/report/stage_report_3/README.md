# 阶段报告三

**细胞核实例分割中的跨尺度候选筛选与面积过滤**  
2026-10-06 修订；实验范围为 10 月 2–5 日的代码审计、配对验证与 EF-P2 实验。

## 阅读入口

- [中文正式版](stage_report_3_zh.pdf) · [TeX 源文件](stage_report_3_zh.tex)
- [中文导读](summary_plain_zh.pdf) · [TeX 源文件](summary_plain_zh.tex)
- [English report](stage_report_3.pdf) · [TeX source](stage_report_3.tex)
- [核验数字快照](verified_numbers.json)
- [最终构建与测试核验](build_verification.json) — 中文正式版 11 页、英文版 12 页、导读 3 页；含文件哈希
- 图表及对应数值：[九对差值](figs/paired_deltas.csv)、[逐划分区间](figs/split_intervals.csv)

正式报告解释方法、对照、历史更正与统计限制；导读面向不熟悉项目的读者。
前两份报告的历史正文没有被改写为新结果。本报告是当前进度的主要阅读入口。

## 2026-10-06 审阅与修订

- 区分候选分析中的类别无关匹配与 mPQ、Dead PQ 的同类匹配，补充 PQ 定义。
- 明确 Dead 分型比例以“预测为 Dead 且已匹配的实例”为分母，不解释为真实 Dead 的分类召回。
- 修正 seed19 历史尺度表中 x2+du2 的 Dead PQ 舍入值：0.18312 → **0.18311**；主表和判定不变。
- 复核 18 份模型配置、108 个配对差值及两轮共 18 个逐划分 bootstrap 区间；未重新训练、推理或搜索阈值。
- 按中文论文习惯修订标题、摘要、术语、段落与参考文献，补充 PanNuke、UNI、CellViT、PQ 原始来源；同步修正英文版与导读中的相关口径。
- 重新生成中英文图表；中文 PDF 改用有效的 TrueType 字体嵌入，并补充回归测试。图表数据与原 CSV 一致。

## 结论及适用范围

九个真实 `(split, seed)` 匹配对上，EF-P2 相对同种子 x1+M/S 的三划分种子均值为：

- Dead PQ：0.17255 → 0.17954，绝对增量 **+0.00699**。
- mPQ：0.49760 → 0.49828，绝对增量 **+0.00068**。
- bPQ：0.66530 → 0.66529，绝对变化约 **−0.000009**。
- 八对 Dead 提高，一对 identity 不变；保留了未过滤 P2 的 **94.3%** Dead 增益。

EF-P2 通过预注册规则，但不是统计学非劣性证明。mPQ/bPQ 的逐划分区间均跨零，
strict mPQ 略降，测试折已跨轮复用，完整双尺度推理成本尚未测量。
三个划分的测试折为 3/3/1，不能当成三批互不重叠病例。

## 数字如何核验

没有将 `findings.md` 当成最终数值来源。报告主表重新读取原始评测汇总：

1. 从 `matched_seed_20261004/p0/<pair>/test/base/ms/eval/summary.json` 读取公平基线。
2. 分别读取 `matched_seed_20261004/p2/<pair>/test/eval/summary.json` 与
   `existence_ef_20261005/p2ef/<pair>/test/eval/summary.json`。
3. 重新计算两轮各九对、六个端点的 **108 个差值**；与 `stats/stats.json` 核对，容差 `1e-12`。
4. 独立重算三划分种子均值、配对差值总体标准差（`ddof=0`）、符号计数和预注册判定。
5. 核对九份 EF `off` 行与冻结 P2 行的五项选择量。
6. 从 `existence_audit_20261005_v2/records.npz` 重算面积及边界分组，与 v2 汇总核对。

上列路径均相对 `runs/analysis/`。这些实验产物在本地保存，不随 Git 分发。
`verified_numbers.json` 是不含机器配置的可移植数值快照；其中保留统计文件 SHA-256。

P0–P3 的 seed19 历史对照来自 `p0p3_20261002_v2/results_verified.json`，
分型审计来自 `typing_audit_20261003/`，逐划分区间来自各轮 `stats/bootstrap_split*.json`。
区间以三个已训练种子为条件，仅重采样图像，不能表示全部训练不确定性。
本次没有重新训练、重新推理或重新搜索阈值；重跑的是报告数字核验、制图和测试。

### 更正优先级

- 边界统计只使用 **`existence_audit_20261005_v2`**。旧版全零边界表无效。
- 旧版非边界记录/分组经此前复核与 v2 一致；未覆盖现存 `scripts/report3_numbers.py`。
- EF identity 回退缺陷已修复，现有九对未触发该缺陷，分数没有变化。
- 不修改历史 selection 的源码指纹。重跑历史评测必须使用相应代码快照，
  或在新输出目录明确记录为新的复核执行；不得关闭来源检查。

## 复现

在仓库根目录、项目固定依赖环境下执行：

```bash
# 既有逐表数字提取；只打印，不修改实验产物
python scripts/report3_numbers.py

# 新增：从原始 summary 与 v2 records 独立核验；JSON 输出到 stdout
python scripts/verify_report3.py

# 新增：由冻结 stats / bootstrap 生成中英文图及 CSV
python scripts/make_report_figures3.py

python -m pytest -q
```

图表默认以 DejaVu Sans 显示拉丁及希腊字符，以 Droid Sans Fallback 的 TrueType 字体显示中文。
若字体路径不同，给制图脚本传入 `--zh-font /path/to/chinese-font.ttf`，应使用含 TrueType 轮廓的字体，
避免将 OpenType/CFF 字体作为 Type 42 嵌入 PDF。脚本支持 `--analysis-root` 与 `--out`。
制图不评测模型，也不改变选择规则；对应 CSV 已核验与修订前一致。

2026-10-06 最终复测：**157 passed，63 warnings**（Python 3.10.21、torch 2.5.1+cu124、
numpy 1.23.5；未安装或升级依赖）。新增中文图表 PDF 字体及文字提取回归测试，修复前失败、修复后通过。
其余训练、推理与评测代码未变动；测试通过不能替代对实验独立性和来源限制的披露。

## PDF 构建

在本目录使用安装了 ctex、Fandol 字体与 xeCJK 的 TeX Live；各运行两遍：

```bash
pdflatex -interaction=nonstopmode -halt-on-error stage_report_3.tex
pdflatex -interaction=nonstopmode -halt-on-error stage_report_3.tex
xelatex -interaction=nonstopmode -halt-on-error stage_report_3_zh.tex
xelatex -interaction=nonstopmode -halt-on-error stage_report_3_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
```

`figs/` 为自含资源。构建时检查当次新日志中的错误、缺图、缺字与越界，不能以旧日志判断成功。
机器地址、远程构建命令与操作记录仅在本地 `ops/` 维护，不属于本报告。
