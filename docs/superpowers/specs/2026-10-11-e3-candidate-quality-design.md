# E3 候选质量学习（dev 轮）设计 spec

起草 2026-10-11｜状态：**待用户批准，未启动。** 依据
[RESEARCH_NEXT_2026-10-06](../../RESEARCH_NEXT_2026-10-06.md) §路线B / §5 E3 行的进入条件，由
E1-C 证据触发（findings 2026-10-11：jury 支持度对候选 GT 匹配率单调，0–2 支持 5–12%、≥5 支持
40–60%；合成空对照 ≥3 率 0.7%）。用户 2026-10-11 批准方向（E3 与 E1-T 并行）。

## 0. 问题与假设

**假设 Q**：x2 合格候选"与 GT 匹配"（IoU≥0.5）这一效用可由廉价特征预测；以学习到的质量
分数替代手工 EF 规则（`a200_p0.9_interior0`），能在验证折上以不劣的 bPQ/strict 代价取得更高的
Dead 增益（主门），或在同等 Dead 增益下显著减少误报添加（次门）。

本轮判定的是**"候选效用是否可学习"**，不是部署收益：EF-P2 已定位为机制证据/teacher 候选
（两模型系统 6.1× x1 成本结论不变）。E3 是效用学习/蒸馏路线的证据门。

## 1. 冻结对象与数据划分（fold3 零读取）

- **应用侧（dev 端点，fold2 = split1 验证折）**：split1_seed1 冻结合格候选 **4,084**
  （`runs/analysis/e1_input_check_20261010_split1_seed1/records.npz`；与
  `existence_audit_20261005_v2/records.npz` 逐列绑定已核，输入 SHA256 在案）。
- **训练侧（fold1 = split1 训练折，OOF）**：`existence_audit_20261005_v2/records.npz` 中
  pair ∈ {split2_seed19, split2_seed1, split2_seed2} 的 **13,332** 行。候选生成模型
  （split2 各 seed 的 x1/x2）训练折 = fold2，fold1 是其验证折 ⇒ OOF 成立。标签 = records
  的 `matched` 列（冻结口径：与任一 GT 实例 IoU > 0.5；`iou_best` 列在案可复核）。
- **禁用**：fold2/fold3 的任何行不得进入训练或特征选择；split1_seed{19,2} 与 split3_* 的
  记录（都在 fold2 上）本轮完全不使用。fold2 行的 `matched` 列只允许出现在端点评测，
  不得进入任何拟合。
- **重放路径**：A0/A1（及各臂添加重放）走与 e1_input_check 相同的 fuse/light 代码路径；
  A1 行相对冻结 EF-P2 选择做硬断言（≤1e-12，沿用 EF 轮 off-row 惯例）。

## 2. 特征（冻结菜单，两折同代码同列）

**低维（A3/A4 共用）**，全部取自冻结 records 列，不新增派生列：
`log1p(area)`、`extent`、`circularity`、`dist`、`pmax`、`margin`、`border`、`class` one-hot(5)
——共 12 维。遇非有限值：记录并中止（不静默填补）。

**支持度（仅 A4）**：support = jury 中与候选 IoU ≥ 0.3 的成员数（排除来源 run；口径同 E1-C）。
- fold1 jury：split2_seed{19,1,2} × {x1,x2}（6 席；逐候选排除其来源 run，有效 ≤5）
  **+ HoVer-Net split2 的 fold1（val）推理 1 席**（唯一新增 GPU 项，见 §7）。
- fold2 jury：E1-C 13 席（`per_candidate.npz` 已算，来源 run 已除）。
- 敏感性列：`support_frac`（support / 该候选有效席数）。
- **结构性限制（如实报告）**：fold1 jury 单一训练折、CellViT+HoVer-Net；fold2 jury 双训练折
  同构。训练/应用的 support 分布偏移用 reliability 对比单列，不据此调整特征。

## 3. 模型（冻结）

- 主模型：逻辑回归（sklearn lbfgs，max_iter=5000，特征标准化）。超参 C ∈ {0.1, 1, 10} 由
  **fold1 内部** GroupKFold（按 image 分组）CV 的 AUC 选择；fold2 不参与。
- 一致性检查：torch 线性头种子 {0,1}（同 T 轮惯例）方向一致；进判定的点估计仍用 sklearn 主模型。
- 校准（描述性）：fold2 上按十分箱 reliability + AUC；按 support 分箱的 p̂ 校准表。

## 4. 臂与选择（fold2 重放；菜单随本 spec 冻结）

| 臂 | 规则 | 菜单 |
|---|---|---|
| A0 | identity（无添加） | — |
| A1 | 冻结 EF-P2 重放（现任）= stage-1 P2（max_area 200 ∧ p≥.9 ∧ interior 任意）+ EF 地板 a30 | 固定一档 |
| A2 | 纯面积（仅面积区间，去掉 p/interior 条件） | (floor, cap) ∈ {0,30} × {100,200,400}，共 6 档 |
| A3 | 学习质量（低维特征） | p̂ ≥ {0.3, 0.4, 0.5, 0.6, 0.7} |
| A4 | 学习质量 + support | 同 A3 阈值 |

每臂内部选择规则**逐字沿用 EF 轮**（select_fusion 语义，参照 = A0 identity 行）：gate
mPQ、bPQ ≥ identity − .002、strict Dead PQ ≥ identity、interior Dead matched > identity；
然后 max val mPQ → 更少添加 → 名称字典序。评测 = 官方 `pannuke_eval`，fold2，输入预测
npz 与 SHA 绑定。

## 5. 判定（dev 初筛值，随本 spec 冻结；不是确认性结论）

对每臂取其选中行，与 **A1 选中行** 比较：

- **主门（Dead 增益路线）**：ΔDead PQ ≥ +.002 且 ΔbPQ ≥ −.0005 且 Δstrict Dead PQ ≥ −.002。
- **次门（FP 精简路线）**：|ΔDead PQ| ≤ .001 且 ΔbPQ ≥ +.0005 且选中添加中 unmatched
  （无 GT 匹配）数 ≤ A1 选中添加 unmatched 数的 75%。
- A3 或 A4 任一门通过 ⇒ "候选效用可学习"成立，进入确认轮 spec（3 split × 3 匹配种子；
  split3 缺 OOF 生成器的问题届时单独决策并预注册）。
- 两门都不过 ⇒ 停线：负结果收档，不加菜单、不换模型挽救；收获记为"候选效用不可由
  廉价特征+支持度预测"（对照 E1-C 的单调性，定位失败层级：特征/标签/分布偏移）。
- **汇报（无论过否）**：全臂×菜单端点表；tissue 分层配对图像 bootstrap（2000 重复，种子
  20261011）；候选级机制诊断 P(match)·E[IoU|match] 对 q/2（按 10-06 §路线B 的推导口径，
  用验证折实际 evaluator 定义）；添加的类别构成（Dead 占比单列）。

## 6. 护栏

- fold-guard：训练数据构造拒绝 fold2/fold3 行；端点运行拒绝 fold3 与其它 split。
- 全部输入/输出 SHA 记录；A1 重放对冻结 EF-P2 选择的硬断言不过即中止。
- 不回改任何已冻结的 EF-P2/P2/matched-seed 结果；E3 是新增 dev 证据。
- 代码入 `src/nucseg/candidate_quality/` 与 `scripts/`（TDD，先测后码），不放 `runs/`。

## 7. 成本

- CPU 为主：特征/训练分钟级；重放 + 官方评测 ≈ 15 个端点 × fold2（并行 workers，数小时内）。
- GPU（默认执行，1 项）：HoVer-Net split2 fold1（**val 折**）推理 1 次，量级同 E1-C 单席
  （约 10–20 min A100）；不触任何 test 折。无任何训练。

## 8. 决策记录

- [x] 用户批准方向（E3 + E1-T 并行）｜2026-10-11
- [x] 用户批准本 spec（"请继续推进"，2026-10-11）。批准后、实现前修正 §4 A2 语义：
  P2 菜单的 a200 为 max_area 上限（frozen selection.json 核实），A2 由单向面积阈值
  改为 (floor, cap) 网格——控制结构不变，语义纠正记录在案。
