# 阶段报告四

**两倍尺度的机制对照、Dead 专家分支检验与推理成本**

实验范围为 2026 年 10 月 6–8 日。中文正式版和导读于 10 月 10 日完成文字修订，纳入 10 月 9 日已有勘误；没有新增实验成绩。

## 阅读入口

- [中文正式版](stage_report_4_zh.pdf)（11 页）· [TeX](stage_report_4_zh.tex)
- [中文导读](summary_plain_zh.pdf)（3 页）· [TeX](summary_plain_zh.tex)
- [English report](stage_report_4.pdf)（12 页，保留原版本；下列勘误优先适用）· [TeX](stage_report_4.tex)
- [本次文字修订与构建核验](language_revision_verification.json)：源码/PDF 哈希、当次日志检查、数值表比较及回归测试
- [原始数字核验快照](verified_numbers.json)：保留 10 月 8 日的冻结数值及历史成本汇总
- [历史构建记录](build_verification.json)：对应旧版中英报告，不是当前中文 PDF 的哈希或页数
- [最新研究建议](../../RESEARCH_NEXT_2026-10-09.md)：固定几何的轻量分类初筛与独立候选盲审，尚未启动

正式版解释实验设置、判定规则和证据边界；导读面向不熟悉项目的读者。前三份阶段报告未在本轮改写，英文版也未重新编译。

## 已有勘误与本次修改

本次重写了中文摘要、章节衔接和导读，解释首次出现的术语，减少生硬的中英混用。正式版的 **7 张数值表、图路径、标签和参考文献键均与原版一致**；图表及实验快照没有覆盖。

[10 月 9 日审计](../../RESEARCH_AUDIT_2026-10-09.md)的更正已写入中文正文：

1. **成本按图像数加权。** 旧 `mean_ms` 将短尾块与整块等权平均。由同一份 `cost.json` 保存的总吞吐恢复图像加权均值，并重新选择吞吐最高的一次重复后，两模型/x1 为 **6.08447×**、TTA/x1 为 **4.66316×**。旧值 6.08687× / 4.66653× 仍在历史表图中，并已明确标注。这是算术修正，不是新 GPU 测量；约 6.1 倍的结论不变。
2. **前向实际使用 bf16 autocast。** 旧产物的 fp32 标签不准确；分块 p95 也不等于单请求延迟 p95。
3. **限定测试数据访问范围。** E2a、DSB、E0 的新机制实验只用训练折与验证折；审计核查了历史测试汇总，10 月 8 日还从已有预测重建了两个测试评测产物。不能称整个阶段都没有访问测试数据。
4. **限定 DSB 结论。** 最终检查点上的代理损失未显示预注册要求的梯度冲突，因此停止该方案；专家分支未经训练，有效性未受检验。不能推断整个训练过程或所有专家设计都没有价值。
5. **保留复现缺口。** 主要数字可复核，不等于所有历史实验都已重跑。P0–P3 的完整来源检查仍被旧种子清单与后续同名目录冲突阻止，没有绕过检查。

## 主要结论

- **E2a：** 四配置、双种子、split1 验证折对照已完成。x1 降低 HV 门槛所得 Dead 增益为原 x2 增益的 9.59%，不足以触发更换基线的规则。这是判定比值，不是因果贡献比例。按原图面积配平门槛后，x2 的 bPQ 代价仍存在，strict Dead 在四组种子配对中均下降。默认 `hv_min_size=30` 不变。
- **DSB：** A、C、D 允许继续；B 首次无效测量按登记方案修正后，64 个有效批次的解码器复合梯度余弦均值为 +0.749、最小 +0.594、无负值，未满足继续条件。方案按预注册规则停止，没有训练专家分支。探针本身使用过 GPU，不能写成完全没有 GPU 工作。
- **E0：** 两模型系统约为 x1 的 6.1 倍推理时间。共享设备负载和计时边界限制了绝对延迟的外推，M/S 等环节未全部纳入。EF-P2 保留为机制证据及教师候选，不主张部署优势，教师效果也尚未验证。

E2a 仅单划分、双种子，四项 Dead 对比的置信区间均跨零；闸门 B 仅测最终检查点、使用代理损失，批次重复使用图像；本阶段没有新的三划分测试成绩。

## 数字如何核验

没有以 `findings.md` 代替原始产物。本次重新执行：

- `verify_report3.py`：108 项指标差值通过；P2 未通过联合规则，EF-P2 通过。
- `verify_report4.py`：输出与冻结 `verified_numbers.json` 完全一致。E2a 为 48 个单次训练指标、24 个配置均值和 24 个比较均值，共 96 项；闸门统计由逐批或逐行数据重算。
- 成本修正：按 `1000 / img_per_s` 独立恢复图像加权均值；见[成本审计快照](../../results/inference_cost_audit_20261009.json)。
- 正式报告 7 张表的数字与修订前逐项比较一致；中文正文补入的修正值与成本审计一致。bootstrap 区间沿用冻结结果，没有重新抽样。
- `python -m pytest -q`：**277 passed，64 warnings**。警告主要为依赖弃用和既有张量转换提示，没有为消除警告升级依赖。

历史 `verify_report4.py` 刻意复现旧成本快照；通过这一核验不表示旧加权算法正确，图像加权结果另表报告。

图表与对应 CSV 保留在 `figs/`：`e2a_contrasts.csv`、`gate_b_cosines.csv`、`e0_cost.csv`。

## 复现与构建

在仓库根目录、项目固定依赖环境中执行：

```bash
python scripts/report4_numbers.py
python scripts/verify_report4.py
python -m pytest -q
```

`make_report_figures4.py` 可从冻结产物重新生成历史图表，本次文字修订没有重制图表。Python 3.10、torch 2.5.1+cu124、numpy 1.23.5 保持不变。

在本目录使用具有 ctex、Fandol 字体和 xeCJK 的 TeX Live 编译，各运行两遍：

```bash
xelatex -interaction=nonstopmode -halt-on-error stage_report_4_zh.tex
xelatex -interaction=nonstopmode -halt-on-error stage_report_4_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
```

本次两份中文 PDF 的新日志均无 TeX 错误、缺字、未定义引用或越界文本框；已核对构建源码哈希、可提取正文和关键数字，并抽查标题、导读分页和正式版附录。没有声称独立逐页视觉审查。`figs/` 为自含资源；机器配置和构建操作记录仅在本地 `ops/` 保存。
