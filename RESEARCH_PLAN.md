# 研究计划：融合多模态与生成式世界模型的 PanNuke 细胞核实例分割

> 制定日期 2026-09-24。文献调研由 4 路并行检索完成（WebSearch 被限流，主要通过 arXiv / HF / GitHub API 获取原文与表格）。标 **[未核实]** 的条目需要你在引用前自行确认；2026 年的 arXiv 预印本大多尚无正式 venue。

---

## 0. 一句话定位

**题目（暂定）**：*Language-Anchored, Generation-Stress-Tested Nuclei Segmentation: Fixing Touching and Rare Nuclei on PanNuke*

**核心论点**：PanNuke 上 backbone 换来换去只差 1–2 个 mPQ 点（CellViT-HIPT 0.485 → SAM-H 0.498 → SOTA ~0.51），真正的短板有两个：
1. **稀有类**：Dead 类 PQ 普遍只有 0.14–0.19（全场最差），Inflammatory/Connective 也只有 ~0.42；
2. **粘连核**：merge/split 错误。

我们用两种"新范式"各打一个短板，并用第三种做评测：

| 支柱 | 范式 | 针对的问题 | 风险 |
|---|---|---|---|
| **A. 语言锚定的类型头** (主方法) | 多模态 / 病理 VLM (CONCH 文本空间 + LLM 生成的形态学描述) | 稀有类 (Dead)、类间混淆 | 低 |
| **B. 失败驱动的反事实合成** | 生成式"组织世界模型"（掩码/实例条件扩散，作为可干预的模拟器） | 粘连核 merge/split + 稀有类样本 | 中 |
| **C. 反事实压力测试 + 真实跨域** | 同一生成模型作为评测用模拟器 | 跨组织/染色鲁棒性 | 低 |

课程要求对应：(a) HoVer-Net 可复现基线 ✔；(b) 按 tissue × class 用 PQ/AJI 评测 ✔；(c) merge/split/miss 分析 + 针对难组织/稀有类的改进 ✔（A、B 两个改进，至少一个必成）。

> **关于"世界模型"的诚实说明**：调研中没有找到任何组织病理学的真正世界模型（动作条件的动力学模型只在放射/手术视频里出现：CheXWorld、Cosmos-H-Surgical 等）。本计划采用可辩护的定义：**可控生成模型 = 组织外观的模拟器**，可以对"布局 / 类别 / 染色 / 组织上下文"做干预并观察分割器的反应（训练用于增强，评测用于反事实压力测试）。写报告时建议用 "generative tissue simulator / counterfactual" 措辞，避免过度宣称。

---

## 1. 调研结论要点

### 1.1 PanNuke 官方 3-fold 排行（均为论文自报，协议：train/val/test = 1/2/3, 2/1/3, 3/2/1）

| 方法 | mPQ | bPQ | Dead PQ | 代码 / 按折权重 |
|---|---|---|---|---|
| HoVer-Net | 0.463 | 0.660 | 0.139 | vqdang/hover_net；公开权重**用了全量数据，不能用于测试** |
| HoVer-NeXt-T (MIDL'24) | 0.477 | 0.656 | 0.154 | ✔ Zenodo 10635618 **按折权重** |
| CPP-Net | 0.48 | 0.68 | 0.131 | ✔ 代码 |
| CellViT-256 / SAM-H (MedIA'24) | 0.485 / 0.498 | 0.670 / 0.679 | 0.149 | ✔ 代码；公开权重用 90% 全折数据（泄漏） |
| CellViT-UNI / Virchow2 (CellViT++) | 0.492 / 0.493 | 0.664 / 0.665 | 0.154 | ✔ |
| LKCell-L | 0.508 | 0.685 | 0.172 | ✔ (基于 CellViT) |
| CellVTA (UNI + CNN adapter) | 0.506 | 0.668 | **0.185** | ✔ 按折权重 |
| **PromptNucSeg-H (ECCV'24)** | **0.512** | 0.692 | 0.161 | ✔ **按折权重**，3090 可训 |
| CFR-SAM-H (2026 预印本) | 0.512 | 0.696 | 0.153 | ✗ |

- 可验证 SOTA 天花板 ≈ **mPQ 0.51 / bPQ 0.69–0.70**，前几名差距 < 跨折标准差。拼 SOTA 不现实也没必要；**讲清楚 Dead / 粘连 / 跨组织**才是贡献点。
- 他人复现普遍低于原文（HoVer-Net 复现常见 0.42–0.44）。**所有对比基线必须自己在同一 pipeline 下重跑**。

### 1.2 多模态（支柱 A 的依据）
- 冻结病理 FM 不一定赢 ImageNet CNN（*Mind the Gap*, 2502.02471：Swin-V2 0.503 > UNI2 0.476 mPQ+）；**浅层特征 + CNN adapter** 是关键（CellVTA）。
- **CONCH（病理 VLM）在稠密任务上与纯视觉 FM 持平**（PFM-DenseBench 2602.03887：LoRA mIoU 0.518 vs 最好 0.539），意味着我们几乎"免费"获得一个对齐的病理文本空间。
- **纯文本提示路线失败**：SAM3 用类名/LLM 改写提示在 PanNuke 上 mIoU < 10%（2604.18225）。⇒ 语言应作为**训练模型内部的先验/正则**，而非提示接口。
- 最近邻工作：**PromptNu**（TMI'25，通用 OpenAI CLIP + GPT-4V 描述，repo 无 PanNuke 配置）、**MONCH**（2412.02978，CLIP，PanNuke 仅语义分割 + F1，其消融显示 Dead 类对文本模块最敏感）。
- **空白**：尚未找到"病理 VLM 文本空间 + 实例分割模型 + PanNuke 官方 mPQ"的工作（受检索限制，表述为"未找到"）。

### 1.3 生成模型（支柱 B/C 的依据）
- **没有任何工作在官方 3-fold 协议上报告合成数据对 PanNuke mPQ 的提升** —— 现有 PanNuke 合成工作（Oh&Jeong MICCAI'24、Co-synthesis ECCV'24、PathDiff ICCV'25）都用非标准 80/20 划分，只报 Dice/AJI/F1。
- 全数据场景下合成增益小（PQ +0.01–0.02），但**稀有类 F1 增益大**（DiffMix CoNSeP misc F1 0.20→0.36；HistoSmith CoNIC 中性粒 PQ 22→29；NucleiMix +0.06）。
- 可用生成底座：**PixCell-256 + Cell-ControlNet**（2506.05127，HF 权重，ControlNet 单卡 A5000 可训；但 20×、仅二值掩码、依赖 UNI2-h 门控）；NucleiMix（有代码，稀有类插入 + 扩散 inpainting）。
- 2608.03990：合成数据的**多样性**比保真度更能预测下游 AJI+ —— 支持"失败驱动采样"而非"随机采样"。
- 反事实压力测试（2605.10894）只在胸片/钼靶做过；**细胞核实例分割尚无**。

---

## 2. 方法设计

### 2.0 统一骨架（所有实验共享）
- **框架**：TIO-IKIM/CellViT 代码库（PanNuke 预处理、按组织/类别 mPQ、组织×细胞加权采样、可换 backbone 都现成；LKCell/CellVTA/NuLite 都是它的 fork，对比公平）。
- **主 backbone**：UNI（ViT-L/16）或 CONCH-ViT-B/16 **+ CellVTA 式 CNN 高分辨率 adapter**（解决 16×16 token 对小核太粗的问题）。
- **后处理**：HoVer 式 NP/HV/TP 三头 + watershed（与 HoVer-Net 基线同构，便于归因）。

### 2.1 支柱 A：语言锚定类型头（Text-Anchored Type Head, TATH）
1. **类别原型**：对 5 类 × 19 组织，用 LLM（本地 Qwen / MedGemma-4B 或 API）生成 K≈8 条形态学描述，例如 Dead: "apoptotic body; pyknotic, karyorrhectic nucleus; hypereosinophilic cytoplasm; shrunken dense chromatin"。用**冻结的 CONCH 文本编码器**编码 → 组织条件化的类原型 `t_{c,tissue}`。
2. **类型头**：把 TP 分支的线性分类器换成 `logit = τ · cos(W·f_pixel, t_c) + b_c`，外加 CoOp 式可学习上下文 token；DenseCLIP 式 pixel-text 对比损失 + **logit adjustment**（按类频率）。
3. **实例级聚合**：在 watershed 得到的实例内做 mask pooling，再与原型做一次实例级对比（per-nucleus 而非 per-pixel，降低边界噪声）。
4. **组织条件**：PanNuke 提供 tissue 标签；训练时用真值，测试时用模型自带的 tissue 分类头预测（CellViT 已有），避免测试泄漏。

**必须的消融**（证明是"语言"而非"重加权"带来的增益）：
- 线性头 + 同等 logit adjustment / focal / 过采样（**最关键的对照**）
- 文本空间：OpenAI CLIP vs PLIP vs QuiltNet vs CONCH vs KEEP
- 原型：仅类名 vs LLM 描述 vs 组织条件描述 vs 随机向量（同维度）
- 冻结 vs 可学习上下文

**预期**：Dead PQ / F1 与 Inflammatory↔Connective 混淆改善；整体 mPQ +0.5–1.5 点（推测）。

### 2.2 支柱 B：失败驱动的反事实合成（Failure-Targeted Counterfactual Synthesis, FTCS）
1. **生成器**：在 **训练折** 上把 PixCell-256 Cell-ControlNet 扩展为**实例+类型条件**（条件输入 = 类型图 one-hot + 实例边界图 + HV 图），LoRA 适配到 40×。
   - 备选 A：NucleiMix 代码（稀有类插入 + inpainting），无需 UNI2-h。
   - 备选 B：自训小型 LDM（SD-VAE + 小 UNet，~1 天 A100）。
2. **失败挖掘**：用基线模型在训练折上做 out-of-fold 预测，统计 merge/split/miss 发生的**局部布局**（接触核对数、核间距、核大小、类别对，如 Dead 紧贴 Neoplastic）。
3. **布局采样器**：从真实实例掩码出发做可控扰动 —— 平移使核接触、复制稀有类核（Dead）插入、形变 —— 按失败分布加权采样，生成 ~10–20k 图像-标签对。
4. **标签忠实性过滤**：用一个强分割器（在训练折训的 PromptNucSeg/CellViT）预测合成图，与条件掩码 PQ < 阈值的丢弃（避免"图不对题"）。
5. **训练**：真实 : 合成 = 1 : r（r ∈ {0.25, 0.5, 1}），合成样本仅在前若干 epoch 或以课程方式加入。

**必须的对照**（等样本量）：简单 copy-paste（GradMix 式）、组织×类过采样、随机布局合成（非失败驱动）。**合成方法若不能打过 copy-paste，就不算贡献** —— 这点要在报告中正面回答。

### 2.3 支柱 C：评测 —— 真实跨域 + 反事实压力测试
- **真实跨域**（零样本，类别映射到 PanNuke）：
  - Lizard/CoNIC（HF `MedOtter/CoNIC2022`，**剔除 112 张来自 PanNuke 的 patch**，20×→上采样 2×）
  - MoNuSAC（HF `RationAI/MoNuSAC`）、PUMA（黑色素瘤，含 apoptotic→Dead，Zenodo 15050523 取 0.65GB ROI 包；注意其标签由 HoVer-Net 初始化）
  - MoNuSeg、NuInsSeg（仅 bPQ/AJI）
- **PanNuke-CF**：固定测试集实例掩码，用生成器重新渲染外观（换组织条件 / 换染色 / 换中心 embedding），得到"标签不变、外观变化"的反事实测试集。验证其有效性：CF 上的性能排序是否与真实跨域排序一致（Spearman）。

---

## 3. 评测协议（统一、严格）
- 官方 3-fold 协议；每折 **last checkpoint**（不做测试集早停）；报告 3 折 mean ± std。
- 官方 `PanNuke-metrics/run.py`：**tissue-averaged mPQ / bPQ**（注意其 `class_stats.csv` 把 conn 写进了 inflam 位置，以终端输出为准）。
- 每类 PQ（Dead 单列）、每组织 mPQ/bPQ 热力图、AJI/AJI+、检测 F_d 与分类 F_c（HoVer-Net 定义，12px 质心匹配）。
- **Strict mPQ**：额外报告一个对"图中不存在的类的 FP"也计罚的版本（官方 mPQ 对此置 NaN，偏乐观，尤其 Dead）。
- **错误分解**（IoU 0.5）：matched / merge（1 pred 覆盖 ≥2 GT 各 ≥50%）/ split / missed（再分 no-overlap 与 poor-shape）/ FP / 类别混淆矩阵，全部按组织拆分；可复用 deepcell-toolbox `metrics.py`（单文件）。
- TTA（8 个二面体变换，**HV 通道需按变换重映射符号/交换**）与 watershed 调参结果**单独**报告。
- 显著性：按图像 bootstrap（组织内分层）给出 Δ 的 95% CI。

---

## 4. 算力与存储分配

| 资源 | 现状（2026-09-24 实测） | 用途 |
|---|---|---|
| **ugradx + ugradv**（共 4×L4-24G，空闲） | home 显示 1.4T 可用，quota 按 15G 规划 | HoVer-Net 基线 3 折训练；全部评测与错误分析；LLM 生成描述 / CONCH 文本编码；PanNuke 压缩版（uint8/uint16 ≈ 2–3GB） |
| **LM2**（8×A100-80G，GPU2 近空闲，其它卡 13–60G 空余） | `/data3` 剩 143G | **主力**：FM-backbone 训练、支柱 A 消融、生成器微调与采样；存放模型权重与合成数据 |
| **LM1 本机**（8×A100-80G，多数卡满载） | `/data7` **仅剩 24G** | 代码、日志、小文件；GPU 只做机会性补位（利用率 100% 会慢） |

**存储红线**：PanNuke 官方 npy 为 float64，解压 ~37GB —— **不要在 /data7 解压**。在 LM2 `/data3` 解压后立即转成 uint8 图像 + uint16/int32 掩码（≈2–3GB），再同步到其它机器；或直接用 HF `RationAI/PanNuke`（~834MB parquet）。

**GPU 预算估计**（推测，按 A100 计；L4 约慢 2–3×）：

| 项目 | GPU·h |
|---|---|
| HoVer-Net 3 折（L4 上） | 20–30 (L4) |
| CellViT-UNI+adapter 基线 3 折 | 20–30 |
| 支柱 A 消融（fold-1 上 ~8 配置）+ 最终 3 折 | 60–90 |
| 支柱 B：生成器微调 + 采样 20k + 3 个 r 值重训 | 80–120 |
| 支柱 C：CF 渲染 + 跨域推理 | 10–20 |
| **合计** | **~200–300 A100·h**（与调研中的可行性估计一致） |

策略：**所有消融只在 fold-1 跑，最终配置再跑 3 折**；AMP + 冻结 backbone 前 25 epoch（CellViT 设置）省显存，适配 20–50G 的碎片显存。

---

## 5. 时间线（按 10 周规划；截止日期未知，可等比压缩）

| 周 | 里程碑 | 交付 / Go-No-Go |
|---|---|---|
| **W0（立即）** | 申请门控权重：CONCH、UNI、UNI2-h（Mahmood Lab，需机构邮箱，用 JHU 邮箱）、PixCell；下载 PanNuke 到 LM2 并压缩 | 权限到位 |
| **W1** | 数据管线 + 官方评测脚本跑通；用 HoVer-NeXt / PromptNucSeg **按折权重**直接评测，验证评测管线（应复现其论文数字 ±0.01） | 评测数字对得上 → Go |
| **W2–3** | **HoVer-Net 基线**（vqdang 官方，fast 模式，Py3.9 + torch2 + numpy 1.23.5，输入反射填充到 348，自写 PanNuke loader）3 折；CellViT-UNI+adapter 3 折；错误分解分析 | 目标 HoVer-Net mPQ ≈ 0.44–0.46；**课程 (a) 部分完成** |
| **W4–5** | 支柱 A 实现 + fold-1 消融 | 相对"线性头+同等重加权"Dead PQ 有提升 → Go；否则转为以 B 为主 |
| **W5–7** | 支柱 B：生成器条件扩展与微调（并行）、失败挖掘、布局采样、过滤、重训 | 合成 > copy-paste（fold-1）→ Go |
| **W8** | 最终配置 3 折；A+B 组合 | 主表 |
| **W9** | 支柱 C：跨域零样本 + PanNuke-CF | 鲁棒性表 |
| **W10** | 报告、图（组织×类 PQ 热力图、错误分解条形图、合成样例、CF 样例） | 终稿 |

**最小可交付（保底）**：HoVer-Net 基线 + 完整评测/错误分解 + 支柱 A。支柱 B/C 是加分项。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 门控权重（UNI2-h/CONCH）审批慢 | 许可宽松替代：KEEP（MIT，ViT-L + 文本塔）、QuiltNet（MIT）、H-optimus-0（Apache-2.0）、Hibou-L；生成器用 NucleiMix / 自训 LDM |
| CONCH v1.5 **未公开文本塔** | 文本用 CONCH v1（0.8GB，含文本编码器） |
| PixCell 为 20×、二值掩码 | LoRA 适配 40× + 重训 ControlNet 条件分支；或下采样 PanNuke 生成后上采样（需对比） |
| 权重泄漏 | 不在 PanNuke 测试折上评测任何"全量训练"的公开权重（HoVer-Net 官方、CellViT、NuLite、BiomedParse v1）；生成器只在训练折微调 |
| 增益 < 跨折方差 | bootstrap CI；重点报告 Dead/稀有类与 merge/split 计数等"针对性指标"，而不只看 mPQ |
| 共享 GPU 抢占 | 训练脚本支持断点续训；消融限 fold-1；L4 跑基线与评测 |
| 旧代码环境（hover_net 锁 torch1.6/Py3.6，A100 不支持） | Py3.9 + 新 torch + `numpy<1.24`（imgaug 依赖）；或替换 imgaug 为 albumentations |

---

## 7. 立即可做的事（下一步）
1. 提交 CONCH / UNI / UNI2-h / PixCell 的 HF 访问申请。
2. LM2 `/data3` 下载 PanNuke 三折 → 转 uint8/uint16 → 同步到 ugradx。
3. clone `TIO-IKIM/CellViT`、`vqdang/hover_net`、`TissueImageAnalytics/PanNuke-metrics`、`windygoo/PromptNucSeg`、`digitalpathologybern/hover_next_train`；下载 HoVer-NeXt / PromptNucSeg 的按折权重做评测管线校验。
   - 调研代理已临时下载：`/tmp/hn`（hover_net clone）、`/tmp/hnres/hn_pannuke.tar`（HoVer-Net PanNuke 全量权重，**仅可作 sanity check**）。

---

## 附录：关键文献（arXiv ID）
- 分割：HoVer-Net (MedIA'19)；CellViT 2306.15350；CellViT++ 2501.05269；HoVer-NeXt (MIDL'24)；PromptNucSeg 2311.15939；LKCell 2407.18054；CellVTA 2504.00784；NuLite 2408.01797；CPP-Net 2102.06867；KongNet 2510.23559；Mind the Gap 2502.02471
- 多模态：CONCH 2307.12914；UNI 2308.15474；Virchow2 2408.00738；KEEP 2412.13126；QuiltNet 2306.11207；PromptNu (TMI'25, doi 10.1109/TMI.2025.3579214)；MONCH 2412.02978；SAM3-pathology 2604.18225；PFM-DenseBench 2602.03887；DenseCLIP；CLIP-Driven Universal Model (ICCV'23)
- 生成：PixCell 2506.05127；NucleiMix 2410.16671；DiffMix 2306.14132；HistoSmith 2502.08754；PathDiff 2506.23440；Co-synthesis 2407.14434；NuDiff 2310.14197；合成多样性 2608.03990；StainFuser 2403.09302；反事实压力测试 2605.10894
- 评测批评：Foucart et al. Sci Rep 2023 (doi 10.1038/s41598-023-35605-7)
- 可选扩展（分子监督）：GigaTIME-Flash 2607.18218（虚拟 mIF，Apache-2.0）；CytoFormer 2608.16718；STHELAR（Xenium，CC-BY-4.0）
