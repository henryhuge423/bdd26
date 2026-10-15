# 研究进度与收尾清单（2026-10-14）

[Mechanism closeout](../results/VISIBILITY_SUPERVISION_MECHANISM_2026-10-15.md).

> **2026-10-15 update (America/New_York):** All eight visibility-supervision runs and original-image fold2 evaluations are complete. Frozen dev admission fails on original-image clauses 2 and 3: A2−A0 mPQ/bPQ losses exceed .002 in both seeds; seed1 strict Dead falls .004372; mean A2−A1 strict Dead falls .009781. All eight fixed-C2 mechanism evaluations are now complete. A1−A2 mean miss-rate improvement is −.008621 (95% CI [−.040918,+.017699]); the frozen mechanism clause also fails. No added seeds or confirmation. No additional seeds or three-split confirmation; PM-Net remains a proposal. See [full results](../results/VISIBILITY_SUPERVISION_DEV_2026-10-15.md). Earlier dated execution statements below are historical snapshots.


LM1 为权威工作区。本次代码基点 `research/visibility-e4` / `988c0cd`。
本次独立核验聚焦 E4 修复后工件；其他路线状态来自当前研究计划、findings 和已有报告，不声称全部历史实验重新复现。

## 当前状态

| 工作 | 当前状态 | 本次处理 / 下一步 |
|---|---|---|
| 官方评测、基线和阶段 I–V | 已形成代码与报告 | 保留历史成绩与协议边界 |
| 冻结特征事后重分型 T0–T3 | 按门关闭 | 不扩大分类器菜单 |
| DSB 专家分支 | Gate B 停线 | 无新专家训练 |
| E3 候选质量学习 | dev 准入失败、关闭 | 图像级缺类风险属于新假设，不自动重开 |
| E1-C / E1-T | 检测器共鸣证据已形成 | 纳入标签质量讨论；不作生物学确证 |
| E1 专家盲审 | 资源不可得，未执行收案 | 保留材料包与义务；不可写“人类验证完成” |
| M 合并类诊断 | 已执行，描述性 | 五类成绩不改；合并机械效应不等于可恢复性能 |
| E4 可见性探针 | 修复后全量运行已完成 | 本次核验指纹、窗口、C0与冻结门；同步旧文档 |
| 可见片段监督 | 当前用户已批准 dev 实现和启动训练 | 全量回归与smoke已通过；A0–A3 seed19与A0 seed1训练和fold2评估完成，A1 seed1训练中，余2次排队 |
| 最终综合报告 | 阶段报告齐备，综合稿仍需整合 | 按以下框架整合，不将草案写成已执行方法 |

## 本次确认的 E4

有效工件 `runs/analysis/visibility_probe_20261014/probe.json`，对应修复代码 `988c0cd`。
H1 面积分层 Dead 贴边−内部漏检差 +.1738 [.1060,.2528]；H2 DiD(C2) +.1474 [.1392,.1557]，
插值对照 −.0006；H3 碎片面积档差 +.8175 [.7968,.8393]。三门仍通过。
旧10-13工件的 train-mode推理与粗H1门已失效，保留历史，不再引用为正式结果。
本次没有重复GPU推理，没有新训练；只审计已有修复后工件。

检查点、缓存预测、GT实例图、冻结eval、runner/module SHA256一致；192/160窗口各2523个
逐位复现；C0 official mPQ .483486065526703对冻结eval逐位一致；门按冻结阈值独立复算一致。

## 最终报告分析框架

1. 先说明严格协议、发布权重与自训练来源、last-checkpoint规则、三split和重复测试使用限制。
2. 把分型与检测拆开：oracle是GT辅助上限；T/CONCH负结果不否定全部联合训练路线。
3. 标签质量证据分存在性和分型两轴，E1-C/E1-T仅提供相关检测器的共鸣证据；专家缺失单列。
4. M只说明合并标签空间下的度量响应，不能把+.0519称为标签错误导致的可相加性能损失。
5. Dead检测部分串联尺度、HV门槛、DSB、EF-P2/E3和E4；每条假设按自身门收案，不把不同轮次
   的端点混成一个“改进方法”。EF-P2的成本约6.1倍与strict轴限制应和official增益同时呈现。
6. E4的准入通过支持设计下一轮，不证明特定HV机制，也不保证训练成功。窗边位置、上下文、
   zoom残余混杂和单检查点限制与主结论放在一起。

## 剩余工作与执行顺序

- 审阅可见片段监督设计，冻结四臂和准入规则，再制定实现计划与smoke/完整dev算力预算；
  当前批准已覆盖实现和8次dev训练；三split确认仍需新的研究立项。
- 综合稿按上述框架整合现有阶段材料，并统一官方/strict、描述性/方法结果和独立性措辞。
- 如综合稿引用旧typing-oracle/T轮具体CI，先按已修复bootstrap重新计算；现有findings已标注
  旧CI失准。点估计与停线判定未因该修复翻转，不把未重算的旧CI当作当前有效区间。
- 专家评审仅在真实资源出现时恢复原冻结协议；模型预审不能替代专家证据。

验证、提交和进程记录见LM1 ignored `ops/progress_20261014_codex/`；本次会话全量pytest
最终退出码以 `test_status.json`、测试数以 `pytest.log` 为准。

## 前次E4收尾验证

指定环境全量 `python -m pytest -q`：**367 passed，19074 warnings，434.61秒，退出码0**。
首次控制请求120秒超时已明确终止；后台进程3970414及其pytest子进程3970416已正常结束。
大量warnings来自合成探针用例空分组的NumPy非零索引/均值操作；没有据此隐瞒测试噪声或改动统计实现。
文档相对链接与新门数字校验、git diff --check均通过。

## 当前训练实现验证

可见片段监督四臂实现已完成；独立评审的A3验证HV口径问题以2RED→55GREEN修复。
最终全量回归 **403 passed，19074 warnings，996.77秒，退出码0**；warnings主要来自既有
E4合成用例空分组NumPy操作。GPU smoke已通过、队列已启动；尚无新方法成绩。
文档分类迁移10文件；独立校验172链接无缺失，既有报告和数值快照不变。

## 当前 dev 训练执行（2026-10-14，美国东部）

用户已授权的四臂双种子 dev 队列持续执行：`A0_seed19`、`A1_seed19`、`A2_seed19`、`A3_seed19`、`A0_seed1` 已完成130轮和2523张fold2验证，`A1_seed1` 训练中，剩余2次排队（seed1的A2/A3）。
A0验证official mPQ **.4840002**、bPQ **.6536900**、Dead PQ **.1610816**、strict Dead PQ **.1174554**。
A1验证official mPQ **.4767094**、bPQ **.6464515**、Dead PQ **.1934476**、strict Dead PQ **.1253827**；A2验证official mPQ **.4748876**、bPQ **.6474460**、Dead PQ **.1833232**、strict Dead PQ **.1199311**。
A3验证official mPQ **.4789208**、bPQ **.6468693**、Dead PQ **.1976473**、strict Dead PQ **.1182505**；A0_seed1验证official mPQ **.4784470**、bPQ **.6499801**、Dead PQ **.1575460**、strict Dead PQ **.1239126**。
以上仅为本轮重训基线的dev验证结果；尚无干预臂完整结果，不与历史测试均值混比，也不判断方法收益。
上一次检查（2026-10-14 23:40美东）核验144个冻结源码文件、当时已启动两run的config哈希、A0最终模型哈希与130轮完整有限loss记录，均与队列清单一致。
固定E4 C2的2523窗口仍在，最终机制复评及准入判定待八run完成后执行。
最终全量回归403 passed/19074 warnings；真实fold1上冻结与解冻batch16 smoke通过。
训练科学源码固定为 `eee3248`，144个源码文件和9个权重/数据输入已保存指纹，
八个命令与完整config哈希冻结；首run实际config与冻结值一致。
只训练fold1、验证fold2、130epochs、最后checkpoint、x1无TTA；每run完成后自动原图fold2评估。
输出根 `runs/visibility_supervision_20261014/`。训练/验证失败即停，不自动恢复或追加菜单。
当前尚无新dev性能判定；八run完成后仍须固定C2机制复评与原预注册门分析。
当前训练与运维记录在ignored `ops/visibility_supervision_20261014/`；PID/资源细节不放入公开文档。

## 续接检查与扩大标签审计

本次固定fold1前1024张、每张一次确定性增强的CPU审计通过：555张执行裁剪，68张含79个符合A2条件的片段，79个均产生HV差异，共1096个像素。
A1/A2图像、NP/TP标签、计数和增强RNG一致；所有HV修改严格限于符合条件的片段。
在该1024张子集中符合条件对象里Dead为0（整个训练折并非如此，见下节）；因此该子集审计只验证标签修改范围，不能估计Dead干预暴露量，更不是方法收益证据。
实际采样分布与训练全程暴露未由此估计；仍需完成八run及冻结C2机制比较后才能判定。

### 全fold1标签暴露审计（2026-10-14，第二次续接检查）

为量化整个训练折而非前1024张子集的干预暴露，对fold1全部2656张训练图各做一次确定性增强（种子19+序号，A1/A2配对），
断言A1/A2的图像、NP/TP标签、计数与增强RNG逐位一致、越界像素HV不变、A2仅在符合条件片段上取min_size=1目标，全部通过；实际裁剪1352张。
A2修改范围：227个符合条件对象（184张图），按类为Neoplastic 64、Inflammatory 55、Connective 61、**Dead 7**、Epithelial 40，227个对象全部产生HV差异，共3086像素。
因此“前1024张中Dead为0”是子集现象：整个训练折中A2可触及7个Dead对象，占该折751个Dead对象的0.93%，占全部符合条件对象的3.1%。
A3修改范围（`hv_targets(min_size=1)`与`min_size=30`的差异，代码事实的直接测量）：只落在最终工作面积小于30像素的对象上，
全折共1965个对象、25433像素，按类为Neoplastic 714、Inflammatory 378、Connective 529、**Dead 70**、Epithelial 274，占该折Dead对象的9.3%；
符合条件对象全部落在A3范围内（eligible_outside_a3_scope=0）。
以上为标签修改范围的机械测量，不含模型、GPU、验证/测试折或指标，既不是干预暴露的训练分布估计，也不是方法收益证据；八run与固定C2机制比较完成后才能判定。
工件：ignored `ops/visibility_supervision_20261014/exposure_audit_fullfold1.py`、`.json`、`.log`与对应launch记录。

## 第二次续接检查（2026-10-14，美国东部）

本次为只读检查，未新建SSH连接（复用既有两条master连接），未改动科学源码、队列、既有run或既有产物。
队列进程 `run_visibility_batch.py`（PID1423016，父进程1）存活。A0/A1/A2 seed19 已完成130轮与2523张fold2验证；
A3 seed19 训练中（检查时已记录epoch 83）；seed1的A0/A1/A2/A3 仍排队。A0至A3四个日志的traceback计数均为0。
各run的config哈希、final模型有限性与SHA256、恢复检查点释放均由队列自身在run边界核验，逐run记录见ignored `queue_events.jsonl`；本检查未重复执行该核验。

A3有效干预范围（代码事实，需写入最终机制分析）：`visibility_hv_targets` 在A3下取 `hv_targets(min_size=1)`，
而最终工作面积不小于30像素的对象在min_size=30与min_size=1下目标相同，因此A3相对A1的实际目标差异只落在
最终工作面积小于30像素的对象上，并非字面上的“全部对象”。A2的修改范围则是同一面积口径下**且**被裁窗截断的对象。

本轮已知且不修复的限制：训练引擎缺少增强worker状态检查点，八run均无断点续训；这是冻结轮次的既定设计
（manifest固定 `eee3248`），改动引擎会使已完成run失去出处一致性。下一轮若需续训，须在启动前加入该能力并重新冻结。

下一步（计划规定，尚未执行）：八run与各自fold2评估完成后，实现并运行固定C2配对机制评估，再按预注册门判定。
现有E4 runner只接受其历史checkpoint，不能直接指向新臂，故该评估需要新的实现与独立审查。

## 第三次续接检查（2026-10-14 19:30，美国东部）

本次为只读检查（复用既有master连接，未新建SSH连接），未改动科学源码、队列、既有run或既有产物。
队列进程 `run_visibility_batch.py`（PID1423016，父进程1）存活；`A3_seed19` 与 `A0_seed1` 已完成130轮与2523张fold2验证，
`A1_seed1` 训练中（检查时epoch 76/130），`A2_seed1`、`A3_seed1` 排队；本轮run日志traceback计数为0（保留的smoke.log属于启动前已处置的失败尝试）。
本次以JSON解析逐位核对五run的 `eval_fold2/summary.json`：A0/A1 seed19与上节记录一致；A2 seed19的Dead PQ与strict Dead PQ第7位小数与上节记录不符
（上节 .1833230/.1199305，工件 .1833232/.1199311），已在上文按工件更正；`A3_seed19` 与 `A0_seed1` 的指标本次直接取自各自summary.json。
磁盘 /data7 余约22G，仍高于队列14G启动门槛；`smoke_result.json` status=passed（A2、fold1）。
本检查同时刷新上文“当前 dev 训练执行”与README/计划/findings中的当前状态行，并按项目惯例提交；其余内容不变。

## 2026-10-15 synthesis

[Evidence chain and PM-Net pre-training decisions](../report/RESEARCH_SYNTHESIS_2026-10-15.md).
