# E6 结果复盘与 E7 研究推进 — 2026-10-16

E6 四次正式训练、最后检查点评估和成本测试已完成。冻结质量门失败，原配方停止扩展。E7 是独立的学习率日程对照，不覆盖 E6 结论。当前所有方法数字均为 split1/fold2 开发验证，绝非三 split 测试成绩。

## 已核验的结果

|run|official mPQ|bPQ|Dead PQ|strict Dead PQ|merge率|
|---|---:|---:|---:|---:|---:|
|B_seed19|0.482774|0.651880|0.187975|0.125317|0.045213|
|Q_seed19|0.090481|0.178044|0.006351|0.006090|0.252405|
|B_seed1|0.480090|0.649665|0.171294|0.131765|0.045514|
|Q_seed1|0.114916|0.207183|0.005550|0.004918|0.319365|

两种子均值：B mPQ=0.481432、bPQ=0.650773、Dead=0.179635；Q 分别为 0.102699、0.192614、0.005951。
几何匹配率从 79.00% 降至 29.00%，合并率从 4.54% 升至 28.59%，检测 F1 从 0.8129 降至 0.3938。两种子中五类 PQ 及全部19组织 mPQ 均下降。整体实例表示/学习/解码链条失败，不能归为仅 Dead 分类不足或小幅随机波动。

内部 Dead 主端点：固定 741 对象/60 图，Q−B 匹配率变化 -0.507422，1000次固定种子组织分层图像聚类 bootstrap95%CI [-0.563164,-0.451449]。逐种子变化 {"19": -0.5330634278002699, "1": -0.4817813765182186}。真实漏检改善 B_miss−Q_miss=-0.051282，不使用1−matched替代 max-IoU<.1。

冻结判定：{"interior_dead_matched": false, "official_dead_pq": false, "quality_floors": false, "strict_mpq_nondecrease": false, "merge_guard": false, "latency_each_seed": true, "finite": true}；总准入=False。
本段来自原冻结队列已完成的全对象、全类别/组织分析（SHA256 `2943799bcfaf6fa27b3b300f806e5ab13b56523850c8a8062a25fc0790ef01cc`）。独立目标子集复算已复核B19内部匹配/漏检率完全一致；原全量分析完成后，仅停止本轮新建且经实时归属核验的冗余子集进程，部分日志保留。未改动原队列或产物。

更关键的错误分解：内部Dead完全漏检(max-IoU<.1)仅恶化5.13个百分点，但IoU落在[.1,.5]中间区间的比例由 9.99% 增至 55.60%。约45.6个百分点的匹配损失转为中等重叠而非全无重叠。结合Q预测实例数83146/78976高于B的59442/59131及显著合并上升，当前问题应描述为定位、轮廓和实例分离链条退化，不能笼统写成“查询数不够”或“只需降低置信度”。这是错误构成描述，不能在没有干预对照时单独归因于损失、日程或重叠竞争。

## 成本与来源

同卡无外来 GPU 进程、batch1、固定66训练图、16图预热、3次重复，Q/B均值延迟比 seed19=0.7712、seed1=0.7955；load_comparable=True。包含归一化/H2D/前向/完整解码/重叠处理/CPU输出，排除模型加载与磁盘I/O。成本门通过不构成质量胜利。

四次均130完整epochs、21580步，训练/预测/评估退出码均0。回收的配置、来源、130轮日志、评估摘要均逐文件对照 completion SHA256；B/Q每种子全部130轮 data_digest一致。E6 来源43f47edca0d8adea1b2d506deb8490d892fc9ecc；小型凭证位于 `ops/e6_review_20261016/`。大检查点和预测保留LM2，冻结分析原进程未被修改。

## 为什么现在优先检查优化

原配方 ExponentialLR gamma=.85，每epoch衰减：初始3e-4，解冻前25轮后约5.16e-6；日志epoch25记录的下一轮学习率约4.39e-6，epoch129约2.00e-13。学习率求和中解冻后的105轮只占约1.72%。这是精确日程算术，不等同于优化器实际更新量或中介因果效应。

新查询头在原日程中训练掩码Dice未充分收敛；低损失的B/Q数值也不可跨不同目标直接比较。Q的配对原始掩码召回高于最终解码输出，提示类别筛选/像素竞争/目标配对同样应保留为候选机制，不能把一切归为学习率。

## 已执行的训练集机制探针

用户批准固定fold1同批16图、seed19、冻结编码器、两臂各300步、最多20分钟。两臂从同一个Q19最终模型和AdamW状态开始，保留同样初始随机状态；控制臂继续约2e-13终末学习率，干预臂重置3e-4恒定。图像和增强仅抽取一次，不读取fold2/3，不保存探针训练权重。两臂共同存在train模式和BN状态更新；对照不是严格权重/BN零变化。

|固定训练批次端点|共同起点|终末LR控制300步|恢复LR300步|
|---|---:|---:|---:|
|eval loss|2.9285|2.9397|1.8671|
|配对原始掩码 recall IoU>.5|54.38%|53.78%|75.53%|
|配对最终解码 recall IoU>.5|18.73%|20.24%|36.86%|

这里的召回只按训练匹配查询配对计算，不是对所有预测取最佳IoU，不是official检测指标。包含331个增强后训练目标；不外推全训练集或验证集。耗时301.7s，峰值保留显存9.85GiB。此前冻结的分配门：loss下降≥20%、配对解码recall较起点和对照均≥+.10；实测 36.24%、+0.1813/+0.1662，通过。

结论限于：原Q最终状态仍有可学习空间，恢复有效更新可以改善这个固定训练批次。不能证明完整训练退化由日程造成，不能证明gamma=.99正确，也不能证明泛化改善；仍远未完美拟合，不排除匹配/表征/输出竞争问题。

## 已冻结的后续 E7

唯一科学变量为 gamma .85→.99。B/Q×seed19/1均重新从匹配的原始初始化开始；其他训练、标签、分辨率、损失、增强、最终检查点与质量/成本门全部保持。gamma=.99使解冻后学习率求和约占69.53%，但该数值仍不是更新量或预期收益。

观察三个问题：Q是否恢复基本分割能力；B是否同样受益；架构差值是否随日程改变。E7−E6的日程效应及两臂交互仅作开发描述；用新的匹配B/Q对决定质量门。没有额外种子、阈值菜单、fold3、最佳checkpoint选择或SOTA主张。

LM1独立分支 `research/set-mask-e7-schedule`，源码 `c6d9b58ff080a3686ea760c05c533a2f62f51ac8`；LM2新快照 `ops/staging/e7_schedule_20261016`，219文件/40vendor文件，归档SHA256 `169b445c410caf6ad402ae033a0b8e3030452ac03945bb95aa9874da93697e6a`。原分支未提交改动全部保留。当前启动状态以 `ops/e7_schedule_20261016/HANDOFF.md` 为准。

资源约束：串行四run，单run10h、累计40GPUh、32GiB输出、20GiB磁盘余量，失败停止，不重试，不覆盖原产物。LM2共享盘97%使用但尚有约1.8TiB可用；显存与活动任务在实际启动前重新核查。用户本轮明确委托所有研究审定；该授权不放宽连接的一次预算、科学数据隔离或他人任务保护。

## 局限与决策

fold2多次复用；固定两种子；缺少患者聚类信息；不得将图像bootstrap解释为病人或种子总体推断。E6只否定已测最小配方，不是否定所有集合模型，也不证明HV是必须的。E5和可见片段监督的失败结论保留，不在E7偷偷加入这些模块。

本轮不改指标或HoVer-Net，不升级依赖，不发布Git。专项测试LM1/LM2均70passed，原E6基线61passed；新增配置6RED→GREEN、分析2RED→GREEN，队列生命周期4passed。完整套件及新GPU smoke的实际结果写入交接，不以历史测试数代替。

## Verified launch and validation update

E7 formally launched on LM2 GPU6: queue2629445, current B19 child2630256. First epoch0 is committed (1/130), finite loss6.882130760744394,166steps, next LR.000297; this first-epoch loss equals E6 B19 before the schedule diverges. Full CPU487passed/3skipped/19073warnings, exit0; LM1/LM2 focused70passed each. Initial full-suite overwrite-guard failure was caused by a missing worktree runs/config path; after linking the existing asset, the targeted check and complete CPU suite pass. Fresh GPU smoke admitted in75.81s, measured peak15.16/13.18GiB for B/Q, storage estimate28.93GiB; per-run extrapolations2.03/3.69h are not promises under shared load. Source remains c6d9b58; no E7 quality result yet.
