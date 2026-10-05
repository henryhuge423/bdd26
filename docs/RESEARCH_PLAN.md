# 研究计划：融合多模态与生成式世界模型的 PanNuke 细胞核实例分割

> **2026-10-02 合并前复核更新（v2）：** [P0–P3](superpowers/plans/2026-10-02-p0-p3-research.md) 已重新执行受影响的 CPU 阶段。
> 在类别索引/周长修复之外，补上共享边界跨方向累加与官方 GT 通道修正，重跑 P0 验证选择/评测、P1、P2 和 bootstrap；原 v1 保留为被取代的历史版本。
> 原九份 x2 结果不构成完整三不同种子网格（split2 实为 `{19,1,1}`，base 覆盖 3/1/1）。
> 修正版 base+M/S 为 .4998 mPQ / .6656 bPQ；验证门控候选添加为 .4999/.6652，
> Dead PQ .1767→.1835、strict Dead PQ .1340→.1371，仅属待多种子验证的候选。
> [完整结果、置信区间与来源](P0_P3_RESULTS_2026-10-02.md) 优先于下面的历史解释。
> P3 仅为 split1 的 128 张验证图尺度交叉，不与完整测试均值混列。
> 已尝试的合成方案无效不等于所有数据方法无效；当前 CF 排序失败不等于固定标签测试无法施加检测压力。
> 本轮没有新训练或 GPU 推理；复用原始预测，v2 只重跑 CPU 阶段，没有补训种子。

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

## 进展记录（详细数据见 [docs/findings.md](findings.md)）

> 2026-10-05 后续复核已纠正近期条目的统计口径；未改变预注册菜单或历史结果。
> EF identity 测试回退及存在性审计边界分箱错误已修复，现有九组 EF 分数不受影响。
> 旧存在性审计的 `added_by_border` 全零表无效，不能引用；已从冻结验证预测重建 v2 表。
> 贴边候选匹配率 52.86%（5273/9975），内部候选 26.15%（1931/7384）；这是描述性分组，
> 不据此追加过滤规则。37,915 个候选的边界标记均经原预测图四边独立核验，详情见 findings。

**2026-09-25**
- 基线：HoVer-Net（官方 fast 模式，全 patch 监督）3 折训练中；**CellViT-UNI 复现**（`scripts/train_cellvit.py`，
  CellViT 论文配方）split 1 测试折 3：mPQ **0.4951** / bPQ 0.6676（TTA 0.5013 / 0.6728），与 CellViT++ 报告的
  CellViT-UNI 0.492 一致；split 2/3 训练中。
- 支柱 A 可行性（CONCH）：紧裁剪零样本≈随机；**单次前向 + 半径受限注意力池化**（R=32px）零样本平衡准确率
  0.695（描述 > 类名 8–11 个点；组织条件化反而下降），Dead 召回 0.88，但自然分布下 Dead 精度仅 ~0.10。
- 文本锚定分类头只在极少样本（~25 核/类）时优于随机初始化；全量标注时无差别、强锚定有害。
- **负结果**：在 CellViT-UNI 上做 CONCH 实例重分类（零样本融合 / 线性头融合 / 偏置校准，全部在验证折调参）
  测试 mPQ 无提升（0.4951 → ≤0.4953）。
- **关键诊断**：Dead 的损失主要在**检测**而非分类——38.5% 的 Dead GT 核完全漏检（其它类 8–15%），
  匹配到的 Dead 有 69% 分类正确；漏检的 Dead 核内 NP 概率均值仅 0.16，而 TP 分支给出 P(Dead)≈0.26。
  漏检的 Inflammatory 核约一半 NP 已激活 → 在后处理中丢失。

**2026-09-25（晚）— 基线与支柱 A 第一轮全部结束**
- **两个基线的官方 3 折全部完成**：HoVer-Net mPQ .4564 / bPQ .6613（TTA .4666/.6714，论文 .463/.660）；
  CellViT-UNI .4995 / .6654（TTA .5047/.6706，CellViT++ 报告 .492）。课程要求 (a) 完成。
- 免训练找回漏检核（L4 上 96 配置扫描 × 3 splits）：**负结果**，最好的找回方案与只调阈值的对照无差别
  （±.0004 test mPQ）→ 检测缺陷必须训练时解决。
- 训练侧三个消融（split 1）：M1 Dead 像素加权 ×11 ✗；C1 完全平衡采样 ✗；**M2 小核（<100px）加权 ×6
  是唯一候选**（test mPQ +.0042，CI 不含 0；val 无退化）但 ~2× 种子噪声，未确认。
- **种子噪声（3 seeds）**：mPQ ±.0021，Dead PQ ±.0074；Dead 的检验噪声（图像 bootstrap）±.02-.03。
  ⇒ 单种子 Dead 提升 <.01 不可信；Dead 结论必须多种子 × 3 折。
- 下一步：M2 补 2 个种子验证（~2×2h A100）；支柱 B（PixCell 合成，目标改为小核/Dead 检测）为主攻方向。

**2026-09-26（凌晨）— M2 三种子判定：负结果；支柱 A 第一轮关闭，转攻支柱 B**
- M2（小核 <100px 前景加权 ×5）补齐种子 20/21：test ΔmPQ **-.0010** [-.0043,+.0022]——单种子的 +.0042
  是种子运气；ΔbPQ **-.0024** [-.0048,-.0002] 显著小幅下降；ΔDead **+.0108** [-.0021,+.0238] 方向一致
  （3/3 种子高于 baseline 均值）但不显著。⇒ M2 不作为贡献（用 bPQ 换不显著的 Dead）。
- **支柱 A 第一轮全部负结果**（重分类 ✗、免训练找回 ✗、M1 ✗、C1 ✗、M2 ✗）；收获 = Dead 瓶颈在检测的
  诊断 + 种子噪声量化规则（已验证）。检测缺陷必须靠数据（支柱 B）或架构，不是损失重加权/后处理。
- 支柱 B 启动（PixCell-256 + Cell-ControlNet 权重已就绪）：第一步 (1) **失败挖掘**——用 split2 模型对
  fold1 做 out-of-fold 预测（fold1 在 split2 中是 val，从未参与训练），统计漏检/merge 的局部布局
  （接触邻居数、类别对、核大小）；(2) **PixCell 条件采样 smoke test**——真实 fold1 掩码 → 生成图，
  检查 20× 模型在 40× PanNuke 上的外观差距（决定 LoRA 适配 40× 还是降采样对比）。
  **两项均已完成**：失败挖掘结论（孤立小核 + 同类簇，见 §2.2 修订与 findings.md）；PixCell 冒烟测试
  440 对象定量评估——**放置忠实（可用作标签）但外观差距结构性**（平色块染色质、粉→紫偏移、高频
  -24%、大掩码欠渲染 322px vs 137px）⇒ **决策：LoRA 适配 40×（训练折）**，适配目标含核内纹理与
  边界真实感；合成时按大小分层 + fill-rate 过滤。下一步：LoRA 训练管线（fold1 图+掩码对）；
  扩量定量（几百 patch，Dead/大核分层）；并行做 copy-paste 对照。

**2026-09-26（上午）— 支柱 B 第一轮并行推进：CP1 对照训练 + PixCell LoRA 管线**
- **CP1 失败驱动 copy-paste**（`src/nucseg/augment/copy_paste.py`，对照实验也是假设检验：Dead 检测
  缺陷是否"数据可修"）：donor 取自训练折（50-400px，Dead 权重 0.4），插入空基质、8px 间距（对应
  "漏检 Dead 均为孤立核"），prob 0.5 × Poisson(3)；**在几何/色彩增强之前**粘贴使粘贴核同步增强。
  视觉审查（sonnet 子代理）抓到单测漏掉的真 bug：v1 有边界裁剪碎片、增强顺序漏洞（粘贴核未增强=
  可学习捷径）、与未标注核碰撞、色调失配；v2 已修（边界 donor 剔除、基质色偏移、Otsu 暗区防护、
  先粘贴后增强）。run `split1_cp1`（seed 19）已启动训练。
- **PixCell LoRA 管线**（`scripts/train_pixcell_lora.py` + `nucseg.pixcell`）：DiT attention LoRA
  8.3M 参数，ControlNet/VAE 冻结，条件协议与推理完全一致（CFG dropout=0.1，dropped rows 零贡献，
  加性注入已核实）；fold1 条件缓存；5000 步约 40 分钟。定量评估器 `scripts/pixcell_eval.py`
  （200 分层 patch、OOF split2 检测器评渲染率、Laplacian/OD/UNI2-h cos）。LoRA 前基线在跑
  （`runs/pixcell/eval_base`）。
- 待办：CP1 结果（多种子判定规则同前）；LoRA before/after；布局采样扩量合成 → 合成训练对照 copy-paste。

**2026-09-26（下午）— CP1 判定（负结果）；LoRA 放弃，改 Reinhard 后处理**
- **CP1（copy-paste 对照）被拒绝**：val ΔmPQ +.0048（CI 恰不含 0）但 test ΔmPQ +.0009（不显著）、
  **test ΔDead -.031 [-.058,-.010]**（TTA 一致，>4x 种子噪声）。损伤主要在**类型边界**而非检测
  （匹配 Dead 类型正确率 .690→.609、精度 .512→.484；混淆双向增多）。val/test 的 Dead 方向相反
  （+.024 vs -.031）再次验证单种子单折 Dead 不可信。⇒ 真实外观移植扰乱类型边界；数据侧干预只剩
  **完整合成**（外观在上下文中生成）一条路。
- **LoRA 第二轮（r8/5e-5 快照扫描）所有检查点均差于基座**（od_mean_l1 .60-1.10 vs .116）→ 路线
  放弃；改用 **Reinhard LAB 后处理**（od_mean_l1 .072→.009，饱和度与真实对齐）。合成配方锁定：
  基座模型 + 配对上下文 + Reinhard（`synth_pannuke.py` 默认值）。
- SYN-v2 冒烟（32 图）：keep rate .901、0 整图丢弃、2.69 插入/图；颜色与配对真实对齐（中位 RGB
  差 ~3/255、饱和度 68.5 vs 69.6）；Laplacian 比值 .70（基座纹理上限，接受）。视觉审查通过后
  扩量 3000 图 → SYN1 训练（--synth-frac 0.25/0.5，对照仍为 3 种子基线）。

**2026-09-26（夜）— SYN1 判定（无效）；支柱 B 关闭，转支柱 C**
- SYN1 视觉审查通过（外观缺陷全部修复）但发现**结构性阻断**：ControlNet 条件是二值掩码、
  上下文 embedding 不含类别信息 ⇒ 生成器无法按类控制外观，729 个保留对象中 Dead 类型为 0
  （教师 P(Dead)≤.005，非混淆模式）。改用**类别无关标签**（教师自洽类型）后启动 SYN1。
- **SYN1 = split1 基线配方（seed 19）+ 1146 合成（synth-frac .5，keep-rate .886）**：test mPQ
  .4900 / TTA .4956——落在 3 种子基线区间（.4910-.4951）底部，无效偏负；Dead .1521（TTA .1570）
  高于全部 3 个种子抽样但 ~+1pt 处于噪声边缘，且分解显示增益来自类型配对而非分割
  （PQ+/DQ+/SQ+ 全平）；Inflammatory -1.0~-1.6pt。**按预注册规则判定：无效，支柱 B 关闭。**
- 结论：Dead/小核检测缺陷**不可由数据侧干预修复**（真实外观移植 CP1 有害、上下文内合成 SYN1
  无效）；三个支柱中唯一还有正向信号的手臂是损失重加权 npwce_small5（Dead 方向 3/3 种子一致
  但不显著、以 bPQ -.0024 为代价）。
- 下一步（支柱 C，计划 §2.3）：跨域零样本（CoNIC/Lizard 剔除 PanNuke 重叠、MoNuSAC、PUMA，
  类别映射 + bPQ/AJI/PQ）+ PanNuke-CF 反事实渲染（复用已锁定配方：基座+配对上下文+Reinhard），
  验证 CF 排序与真实跨域排序的一致性（Spearman）。

**2026-09-26（深夜）— 支柱 C 启动：跨域数据转换完成并审查通过；PanNuke-CF 渲染器验证通过**
- CoNIC（剔除 112 张 PanNuke 重叠、20×→2×上采样、四分裂 256）= 19,476 tiles；MoNuSAC test
  （RGBA→RGB、Ambiguous 剔除、256 tiling）= 443 tiles；存于 `data/external/`。Sonnet 拼接图
  审查：无转换 bug（亚像素对齐、尺寸方向正确、插值平滑）。
- 评测管线打通（evaluate 可选 tissue_names；external.py 镜像 PanNukeFold、惰性 gt_channels；
  predict_external / collect_external）。修复静默仓库 bug：.gitignore 的 `data/` 吞掉了
  `src/nucseg/data/`（pannuke.py/prepare.py 从未被跟踪）。
- PanNuke-CF：固定 fold3 GT 标签，context {配对,换组织} × Reinhard {自身, donor} 因子设计
  （control/stain/ctx/tissue 四臂 + real 参照 + control 3 种子重复）。冒烟（8 patch）：control
  比真实还容易（bPQ .71→.80）但 strict 掉（外观-类型不一致）；tissue 臂强压力信号（.28）。
  审查修复：空白 patch 剔除（context 是内容通道）、donor 亮度护栏。全量 500/臂渲染中。
- 在跑：CellViT-UNI 3 splits × {conic, monusac} × {plain, TTA}；HoVer-Net 随后。

**2026-09-27 — 外部零样本表 + CF 首轮分析 + 第三评估器就绪**（详见 findings.md）
- 外部零样本（3-split mean mPQ）：CoNIC CellViT .335 / HoVer-Net .261；MoNuSAC .250 / .037
  （HoVer-Net 检测崩溃 F_d .37）；PUMA（新增黑色素瘤集，apoptotic→Dead）CellViT .452 / HoVer-Net .320。
- CF 分析（2 架构）：噪声地板 .004-.006 << 压力效应；stain -.02~-.07、ctx -.14~-.19、tissue 最强；
  但 rank agreement 在架构层面反向 → 需第三评估器定论。
- HoVer-NeXt-T 端口修复（瓶颈在 decode 不在模型）：val 调阈值 flat 解码 3 折 **mPQ .4579**
  （介于 HoVer-Net .4564 与 CellViT .4995 之间），可用作第三评估器；错误模式与二者不同
  （2026-09-28 审计更正：flat 解码 merge 率约为分水岭解码的 2 倍，.10 vs .044/.049，
  missed_bg 高于 CellViT）。

**2026-09-28 — 支柱 C 终判关闭（详见 findings.md 同日条目）**
- 第三评估器 HoVer-NeXt-T 补齐全部外部集（CoNIC .279 / MoNuSAC .112 / PUMA .381 mPQ）与 CF 两臂；
  真实跨域排序三架构一致：CellViT .152 < HoVer-NeXt-T .199 < HoVer-Net .246（domain drop）。
- **CF-vs-真实 rank agreement 在 mPQ 轴反向**（6 点 rho -.89 perm p .033；架构均值 -1.00）。
  轴分解给出机理：真实架构间差异几乎全在**检测轴**（HoVer-Net F_d 掉 .15-.18 = MoNuSAC 崩溃，
  CellViT 仅 .007），而标签固定的 CF 渲染对检测轴施加的压力三架构齐平（.08-.10）——构造上不可施压；
  在 CF 能施压的**分型轴**上排序与真实一致（typing rho +1.00 架构均值）。
- 结论：PanNuke-CF = 分型压力测试，**不是**跨域迁移代理。三个支柱全部关闭；
  剩余工作 = 最终报告与图表（组织×类热力图、错误分解、合成/CF 样例、主表）。
- **阶段报告一已出**：`docs/report/stage_report_1/stage_report_1.tex`（英文 LaTeX，12 页，6 表 4 图，
  TeXLive 2026 编译，PDF 入库；中文平行版 `stage_report_1_zh.tex`，11 页；报告一所有文件
  （含 figs/）集中在 `docs/report/stage_report_1/`，后续报告各占一个子目录）。覆盖三支柱
  全部结论；最终报告还需补组织×类热力图、错误分解图、以及可选的 MoNuSeg/NuInsSeg 外部集。

**据此对 §2.1（支柱 A）的修订**：原"语言锚定类型头"针对分类，而瓶颈在检测，故改为以**检测**为目标——
(1) 类别感知前景：利用 TP 分支/CONCH 稠密先验召回 NP 漏检的核（先做免训练的后处理验证，在验证折调参，
单独报告）；(2) 训练侧：对小核/Dead 的前景损失加权或类别条件的前景监督；(3) CONCH 半径受限池化保留为
少样本/跨数据集（新类别）场景的方法与分析工具。支柱 B（合成）的首要目标相应改为"难检测的小核/Dead"。
所有改进仍须与"同等重加权/后处理调参"的对照比较。

**2026-09-28 → 10-01 — 第二阶段（Dead 检测瓶颈的两个假设检验：分辨率 vs 检测优先架构）**
- **A 线（KongNet，检测优先 CenterNet 式）负结果收案**：按折权重 + 仅在验证折调解码阈值，
  3 折 test mPQ .346 / bPQ .471 / Dead PQ .053——比最弱稠密解码器还低 11 分；融合率 .12-.18
  （稠密 .044-.049），内部 Dead 漏检 .496（CellViT .354）。论文 Dead F_c .59 在严格协议下不复现。
  ⇒ Dead 缺陷跨解码器家族共有，"检测优先"架构不解决问题；karyorrhexis 碎片群的失败是实例分离问题。
- **LKCell-L 严格复评（A1 外部池首条，2026-10-01）**：发布按折权重（HF `xiazhi/LKCell-L`）经共享
  管线复评——比论文均匀低 ~.015（mPQ .4923 vs .508；协议转换而非管线问题）。同等条件（无 TTA、
  last ckpt）下 mPQ -.0072 / Dead PQ -.0218 逊于 CellViT-UNI 基线，但 bPQ +.0076 / mPQ+ +.0057：
  **分割更好、分型更差**——Dead F_c .382 > .359（找得到 Dead）但检测到的 Dead 无法通过类型配对
  转化为 Dead PQ。与 KongNet 判定同向：架构整体优势在匹配协议下不转化为 Dead/分型增益。
  Pillar-C rank pool 扩至 4 架构。
- **B 线（B1，2× 工作分辨率 512）**：split1 全部预注册 Dead 端点正向（内部 Dead 漏检 .35→.25，
  Uterus 剔除 Dead PQ +.02）但 bPQ -.026 全面税。深挖（含视觉审查）定位机制：**解码器像素单位
  常数在 512 下放松**（min_size 10、Sobel ksize 21 不随分辨率缩放）→ 大核碎裂 + 微碎片洪泛。
  预注册修复 `--decode-u 2`（du2）：bPQ 税收回 61-70%，mPQ+ 差距收回 76-88%，Dead 增益保留放大。
- **杠杆 M/S（同类合并 + 边缘细条丢弃）**：仅验证折选择 (frac .25, a_min 40)，四个扩展验证折
  扫描独立同选；**test 判定 PASS**：mPQ +.0014~+.0020、bPQ +.0021~+.0030（全部 bootstrap CI>0），
  Dead 不动。B1 终局（3 折均值）：mPQ .4910 / bPQ .6608（vs 基线 .4995/.6653）换 Dead F_c +.034、
  Dead DQ+ +.026、内部 Dead 漏检 -.070——**按预注册联合端点 x2 不是默认胜利，是明码标价的权衡**；
  mPQ+ 已打平（.5171 vs .5178）。mk2（marker-open 核缩放）验证折更差，未采纳。
- 3×3 种子网格（x2+du2）**9/9 收齐**（split3_seed2 10-01 14:38 落地；canonical 中心复算与 watcher
  位相同，第三链验证；lever PASS 9/9）。**预注册种子噪声判定**：代价稳健——dmPQ -.0098、
  dbPQ -.0069 为最大分裂种子 std（.0021/.0013）的 4.7×/5.3×，逐 run 同号；Dead 增益脆弱——
  +.0049 仅为单种子 Dead std（.0068-.0112）的 0.4-0.7×，成对种子 4/5 胜（split3 s19 翻负 -.0032），
  分裂级种子均值 +.0111（split1，3/3 种子）/ -.0015（split2 翻负）/ +.0051（split3，2/3）。
  3 分裂均值（9 run）：du2 mPQ .4889 bPQ .6579 Dead .1799，du2lev .4905/.6603/.1801；较 seed19-only
  表几乎不动（du2lev vs base mPQ -.0083、bPQ -.0045、Dead +.0051、内部 Dead 漏检 -.069）。
- 推理内存优化不改变评测语义：输出预分配、避免不必要拷贝，并将推理与评测拆为独立进程；
  batch 2 vs 32 结果不变性已验证（<=2.8e-05），见 findings.md 同日条目。

**2026-10-01（晚）— 阶段报告二已出**
- `docs/report/stage_report_2/`：英文版 `stage_report_2.tex`（12 页，7 表 5 图）、中文平行版
  `stage_report_2_zh.tex`（10 页）、面向零背景读者的导读 `summary_plain_zh.tex`（4 页，标题
  《把图像放大一倍，AI 就能看见将死的细胞了吗？》）；TeXLive 2026 编译（pdflatex/xelatex
  各两遍，0 错误 0 overfull），PDF 已入库。figs/ 自含（6 张图表 EN+ZH 双版 + 3 张 montage 复制）。
- 全部表格数字于写前从 artifacts 重新导出核验（`scripts/report2_numbers.py` →
  `runs/analysis/report2_numbers.txt`）：架构总表（KongNet/HoVer-Net/LKCell/base/x2 手臂）、
  du2 回收、杠杆 CI、3×3 网格、B1 初始 Dead 端点电池、尺寸分箱机制、误差构成、npred、
  Uterus Dead PQ——与 findings 条目一致，仅两处以 artifact 为准修正了笔记笔误
  （npred TTA 25.5→28.4 而非"24→27"；du2 回收 noTTA bPQ 71%）。split2/3 `eval_test_fold*_x2{,_tta}`
  与 du2-TTA evals 已汇总到 runs 树。
- 新增图表脚本 `scripts/make_report_figures2.py`（dataviz 调色板同报告一；尺寸分箱/du2 回收/
  种子网格，中英双版）。

**2026-10-03 — 类别置信度与分型一致性审计（P0-P3 §7 建议的第一步，描述性、无阈值选择）**
- 消费 P0-P3 v2 缓存预测 + ugradx 重推的 3 个 x1 test 概率表（L4 顺序 batch2/无 TTA，共约
  8 GPU 分钟；重推与缓存图 0.05-0.09% 实例在 IoU>0 的贪心配对后未配对；已配对者中
  95.8-96.0% 的 IoU>0.5，IoU>0.9 配对的类型一致率为 99.96-99.98%，不据此宣称分布等价）。
  冻结 P2 规则重放精确复现验证折添加数（split2 1713 / split3 2604）。
- 核心结论：**置信度门控的是分型而非存在性**——P2 添加候选 test 匹配率仅 .39-.43（57-61%
  纯 FP），但匹配者分型正确率 .86-.88；被拒候选 .75-.77、Dead .4375-.4444（仅测试折；.60 来自验证折），置信度确实排序
  分型风险（Dead 最陡）。split2 的 p≥0.9 添加 Dead 分型 val .850 → test .667 未迁移；
  split3 p≥0.7 保持 .821。
- **du2 对已匹配 Dead 的分型反而更好**（test .5788 vs x1 .5539；val .6144 vs .5735）；P1
  类型交换优势不在 Dead 分型也不在稳定配对分歧群（base 对 1101 vs x2 对 1038，均为逐 split 计数的均值，近对称）。
  像素多数票 vs 概率表 argmax 不一致（测试三 split 均值：x1 1.23%、x2 1.55%）时，
  匹配且类型正确的比例为 .16-.20（分母包含未匹配预测），可作为待验证的风险信号。
- 匹配种子轮候选（仅记录未测试）：存在性侧过滤、map-vs-row 一致性弃权、按类 Dead 阈值。
  代码/产物：`src/nucseg/postproc/typing_audit.py`、`scripts/analyze_typing_confidence.py`、
  `scripts/typing_audit_numbers.py`、`runs/analysis/typing_audit_20261003/`（含 commands.md）。

**2026-10-05 — 匹配种子验证判定（预注册）；存在性审计与 EF-P2 预注册轮启动**
- 9 个 (split, seed) 匹配对全部执行完毕：18/18 val 选择在读任何测试折前冻结；
  **判定 robust_improvement = false** —— Dead 两臂通过（种子均值 +.00742 ≥ 1× 最大种子
  std .00312；9/9 对 ≥0）但 ΔbPQ −.00087 < −.00064 护栏失败。门控添加把原种子脆弱的
  Dead 增益变成稳健（raw du2 仅 4/5 对），但不能去除 bPQ 税（split3 Dead +.0163 与
  bPQ −.0018 的 bootstrap CI 均不含 0）——**Dead-for-bPQ 交易而非免费胜利**。M/S 次要
  端点 mPQ/bPQ/Dead PQ 的总均值为正，各有 7/9 对非负，并非逐对或逐 split 都改善。过程修复：seed19 硬校验 bug（da3b934）、bootstrap 索引 bug
  （447fd88，含非常量装置回归测试）。全程 142 CPU 测试通过。详见 findings.md 同日条目。
- **存在性审计（仅 val 折，`runs/analysis/existence_audit_20261005`）**：17,359 个冻结 P2
  val 添加，7204 个匹配（41.50%）。面积揭示低精度的小候选子集（<30px 匹配率 20%、n=2294；30–60px 44%）；
  到 base 的有限距离分箱匹配率为 39–44%，不支持仅按距离筛选，不能排除所有碎片机制；
  圆度和填充率也有边际差异，尚未证明独立预测价值；
  添加从不重复 base 已检测的 GT（0/17359）。val 上所选配置 ΔbPQ≈0（−.0007~+.0020），
  测试侧 −.00087 的税属边际量。
- **EF-P2 预注册轮**（[计划](superpowers/plans/2026-10-05-existence-filter-p2.md)，菜单
  冻结于任何 sweep 之前）：在冻结的 P2 添加上加面积下限菜单 {identity, off, a30, a30d,
  a60}，选择规则与判定规则逐字沿用 P2/匹配种子轮；CPU-only，第三次预注册读这些测试折
  （多样性限制如实报告）。
- **EF-P2 判定（同日完成）：robust_improvement = TRUE** —— 30px 面积下限把 bPQ 税
  的均值损失从 −.00087 减小到 −.000009，同时保留 94% 的 Dead 增益（+.00742→+.00699，8 对正增益、1 对 identity 为零，
  约 2.1× 最大 split 内配对差值总体标准差），mPQ 还小升 +.0007。val 选择 7×a30/1×a30d/1×identity；off 行在
  9 对上与冻结 stage-1 逐位一致。EF-P2 取代 P2 成为报告的候选手臂；多样性限制（第三次
  读测试折）与边际性（mPQ/bPQ CI 跨零）如实记录于 findings.md。
  通过规则不等于统计学非劣性证明；两套模型预测的部署代价未测，strict mPQ 仍略降。

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
2. **失败挖掘**：用基线模型在训练折上做 out-of-fold 预测，统计 merge/split/miss 发生的**局部布局**（接触核对数、核间距、核大小、类别对）。
   **已做（2026-09-26，fold1 OOF + fold3 test 双折一致）**：漏检 Dead 几乎全是**孤立**小核
   （0 接触邻居的 Dead miss 率 .39-.42，missed Dead 与任何核接触的 <5%）；merge 集中在**同类簇**
   （merged Dead 的 73-82% 与 Dead 接触 = karyorrhexis 碎片群）；PanNuke 40× 上跨类接触本身罕见。
   ⇒ 布局采样器优先级改为：(1) 孤立小核/Dead 插入空隙基质；(2) 同类密簇（Dead 链、Connective 场）
   训练分离；(3) 跨类接触降优先（原"Dead 紧贴 Neoplastic"假设不成立）。详见 findings.md。
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

## 4. 复现资源与存储

机器清单、资源分配和同步方式不属于项目协议；具体部署记录仅在本地维护。
训练需要足够的 GPU 显存，推理与 CPU 评测可分开运行；根据可用资源设置 batch size，
改变批大小后核对结果不变性，不改变训练/验证/测试划分。

**存储红线**：PanNuke 官方 npy 为 float64，完整解压约 37GB。**不要解压到磁盘**；
使用 `scripts/prepare_pannuke.py` 从 zip 流式转换为 uint8 图像与 uint16 掩码（约 2–3GB）。
数据默认放在 `data/pannuke/`，可通过 `PANNUKE_ROOT` 指定其它位置。

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
| **W0（立即）** | 申请门控权重：CONCH、UNI、UNI2-h（Mahmood Lab，需机构邮箱）、PixCell；下载并流式转换 PanNuke | 权限到位 |
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

> 2026-09-24 制定时的原始清单，三项均已完成；当前状态与下一步见顶部进展记录。

1. 提交 CONCH / UNI / UNI2-h / PixCell 的 HF 访问申请。
2. 下载 PanNuke 三折 zip → 用 `scripts/prepare_pannuke.py` 流式转为 uint8/uint16。
3. clone `TIO-IKIM/CellViT`、`vqdang/hover_net`、`TissueImageAnalytics/PanNuke-metrics`、`windygoo/PromptNucSeg`、`digitalpathologybern/hover_next_train`；下载 HoVer-NeXt / PromptNucSeg 的按折权重做评测管线校验。
   - HoVer-Net PanNuke 全量权重**仅可作 sanity check**，不能用于测试结论。

---

## 附录：关键文献（arXiv ID）
- 分割：HoVer-Net (MedIA'19)；CellViT 2306.15350；CellViT++ 2501.05269；HoVer-NeXt (MIDL'24)；PromptNucSeg 2311.15939；LKCell 2407.18054；CellVTA 2504.00784；NuLite 2408.01797；CPP-Net 2102.06867；KongNet 2510.23559；Mind the Gap 2502.02471
- 多模态：CONCH 2307.12914；UNI 2308.15474；Virchow2 2408.00738；KEEP 2412.13126；QuiltNet 2306.11207；PromptNu (TMI'25, doi 10.1109/TMI.2025.3579214)；MONCH 2412.02978；SAM3-pathology 2604.18225；PFM-DenseBench 2602.03887；DenseCLIP；CLIP-Driven Universal Model (ICCV'23)
- 生成：PixCell 2506.05127；NucleiMix 2410.16671；DiffMix 2306.14132；HistoSmith 2502.08754；PathDiff 2506.23440；Co-synthesis 2407.14434；NuDiff 2310.14197；合成多样性 2608.03990；StainFuser 2403.09302；反事实压力测试 2605.10894
- 评测批评：Foucart et al. Sci Rep 2023 (doi 10.1038/s41598-023-35605-7)
- 可选扩展（分子监督）：GigaTIME-Flash 2607.18218（虚拟 mIF，Apache-2.0）；CytoFormer 2608.16718；STHELAR（Xenium，CC-BY-4.0）
