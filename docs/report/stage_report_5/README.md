# 阶段报告五

**分型缺口的可学习性检验与候选标签质量证据**

实验范围为 2026 年 10 月 9–12 日：分类上限诊断、T0–T3 分型探针（预注册门失败）、
E1 无专家修订执行（E1-C 检测器共识 + E1-M 模型预审）、E1-T 分型共识审计
（Inf/Conn 标签模糊的共鸣证据）、E3 候选质量学习 dev 轮（按预注册规则停线）。

## 阅读入口

- [中文正式版](stage_report_5_zh.pdf) · [TeX](stage_report_5_zh.tex)
- [中文导读](summary_plain_zh.pdf)（面向零背景读者）· [TeX](summary_plain_zh.tex)
- [English report](stage_report_5.pdf) · [TeX](stage_report_5.tex)
- [独立核验快照](verified_numbers.json)：verify_report5.py 的重算结果
  （oracle 锚定官方评测、判定门布尔重推、E1-C 直方图/匹配率联表重算、
  E1-M 一致率与 kappa 重算、E1-T 分组守恒与区间括验、E3 选择规则全行重放）
- 逐表数字快照：`runs/analysis/report5_numbers.txt`（本地 `runs/`，不随 Git 分发）
- 上游文档：[研究计划](../../RESEARCH_PLAN.md)、[10-09 建议稿](../../RESEARCH_NEXT_2026-10-09.md)、
  [结构导读](../../STRUCTURE_AND_STRATEGY_2026-10-09.md)、
  [本阶段预注册规格](../../superpowers/specs/)（T / E1 / E1-T / E3 四份）

## 主要结论

- **上限诊断（10-09，只读审计）：** 固定实例几何、把 IoU>0.5 已匹配对象改成真值类别，
  三划分 mPQ .4995→**.6632**；真值为 Inf/Conn 的子集 →**.6141**，约占全部增益七成
  （选择按真值类别，非可加混淆分解）。该诊断读了历史测试预测，仅用于提出假设。
- **T0–T3（10-10，预注册）：** 门失败。T2 最优 +.0002（CI 跨零）、T3 −.0017；
  未加权线性头 Dead PQ −.0053 触发护栏。真值区域诊断准确率 85.3% ≈ T2 匹配分型
  85.8% ≈ 多数票 85.6%——线性读出与 TP 头用的是同一信息。冻结特征事后重分型
  与 CONCH 负结果合并收档。
- **E1 转向（10-11）：** 人类专家不可得，评审路径未执行收案（材料包与义务保留）。
  **E1-C**：13 席跨折评审团，合成空对照 ≥3 率 0.75%，而与 GT 零重叠候选
  66–73% 获 ≥3 且跨折/跨架构支持（380/751 对 518/1146）——存在性轴的标签可疑
  共鸣证据（非生物学断言）；支持度对匹配率单调（0–2 票 5–12%，≥5 票 40–60%）。
  **E1-M**：两轮模型盲评 kappa −.09～.35，不可用，仅流程验证。
- **E1-T（10-12，预注册）：** 基模型分错的 6,858 个已匹配实例上，12 席评审团多数类
  =基类 **72.5%** [.706,.742]，=真值类 18.2% [.167,.197]（区间分离）；分对实例上
  本征翻票率 1.9%。Inf↔Conn 双向最大格 74.8%/78.5%；GT=Epi/基=Neo 为真值占优的
  例外；Dead 单列。**Inf/Conn 标签边界模糊的检测器共鸣证据按预注册键成立**。
- **E3（10-12，预注册）：** 效用可学（CV AUC .721/.737），但 EF 选择规则的
  strict Dead 门槛拒绝全部 16 个学得行（最高 .1245 < .12957；A1 自己 .1345），
  三臂回落 identity，两门皆负——官方口径的便宜改善全部以缺类 Dead 误报为代价
  （a4_p0.5 strict Dead −.0130）。按停线规则收档；若重开需图像级缺类风险特征。

数据边界：T/E1-C/E1-M/E1-T/E3 未读测试折、无训练；E1-C 含两席 HoVer-Net
验证折推理；上限诊断为只读的历史测试产物访问。本阶段没有新的三划分测试成绩。

## 数字如何核验

- `python scripts/report5_numbers.py`：五份冻结产物逐表导出（含零重叠分层的联表重导，
  该计数在 consensus.json 中只有聚合值）。
- `python scripts/verify_report5.py`：从最底层产物独立重算——oracle 当前值与官方
  基线评测逐位一致；T 门布尔由存储值重推；T0（现行多数票）匹配分型准确率
  85.6% 由冻结预测按官方配对重算（results.json 只存 T1/T2 两值）；E1-C 直方图/匹配率由
  `per_candidate.npz` × 冻结候选记录重算；E1-M 一致率/kappa 由两轮 CSV 重算；
  E1-T 分组总数守恒且区间括住点估计、基模型输入 SHA 与磁盘一致；E3 选择规则
  在全部行上重放并与冻结选择一致——不一致即报错退出。
- `python scripts/make_report_figures5.py`：三张图（E1-C 支持度、E1-T 归属、
  E3 strict 轴）中英双版 + 冻结 CSV。
- `python -m pytest -q`：**335 passed**（本阶段净增 58；2026-10-13 本机复跑确认）。
- E1-T/E3 终版工件在独立评审后重跑，与首版逐位一致（E3 除 bootstrap mPQ 列
  口径修正为官方 NaN 排除外）；文档-产物审计 56 项通过。
- bootstrap 区间沿用冻结结果，未重新抽样。

图表与 CSV 在 `figs/`：`e1c_support.csv`、`e1t_pairs.csv`、`e3_strict.csv`。

## 复现与构建

在仓库根目录、项目固定依赖环境（Python 3.10、torch 2.5.1+cu124、numpy 1.23.5）执行：

```bash
python scripts/report5_numbers.py
python scripts/verify_report5.py
python scripts/make_report_figures5.py
python -m pytest -q
```

`figs/` 自含，PDF 由带 ctex/Fandol/xeCJK 的 TeX Live 编译（EN 两遍 pdflatex，
ZH 两遍 xelatex）；构建记录在本目录 `build_verification.json`，机器与操作细节
仅存本地 `ops/`。
