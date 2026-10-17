# 可见片段监督 dev 轮实施计划

> **For agentic workers:** Use superpowers:executing-plans for native execution and a final independent review. Steps use checkbox syntax.

**Goal:** 实现四臂、双种子 split1 dev 对照，在完成回归和GPU smoke后启动串行训练，并整理研究文档。
**Architecture:** 在既有增强和HV目标生成管线上增加可选的visibility_arm，不增加模型头；运维队列位于ignored ops，实验代码仍在src/scripts。
**Tech Stack:** Python3.10、torch2.5.1+cu124、numpy1.23.5、albumentations1.3、pytest，均不升级。
**Spec:** [2026-10-14-visibility-supervision-design.md](../specs/2026-10-14-visibility-supervision-design.md)。用户已明确授权实现和启动训练。

## Global Constraints

- 训练fold1、验证fold2，split1；seed19与1；四臂A0/A1/A2/A3，禁止fold3、测试推理和自动三split扩展。
- 每次从相同UNI预训练初始化，独立输出目录；130epochs、unfreeze25、batch16、lr3e-4、AdamW betas(.85,.95)、weight_decay1e-4、gamma.85、sampling_gamma.85、workers8、val_every5、x1、无TTA。
- A0 crop概率.1且hv30；A1 crop概率.5且hv30；A2同A1，仅新截断且最终工作面积<30保留hv1；A3同A1，全部对象hv1。
- 未显式设置visibility_arm的既有运行行为不改变；所有新轮次显式臂共同设置Python/NumPy/torch种子。
- 仅复用现有broker；不连接worker、不升级依赖、不读取凭据、不覆盖旧run；每次计算前核验资源。
- 此轮新产生的恢复检查点last.pth在失败/未完成时保留；成功且final.pth验证保存完整模型后，可释放此轮恢复优化器文件。所有预先存在的checkpoint不动。
- 全量回归结束且代码审查无重要问题后才运行smoke；smoke不是调参，任何失败停止队列。

## Review Focus

1. 原图贴边对象、裁窗恰好贴边但未丢像素对象，不能误判新截断。
2. resize/elastic后的工作面积与裁窗原生面积不同；最终阈值用最终工作像素。
3. A1/A2/A3的增强与抽样RNG逐位一致，新增追踪不得消耗随机数。
4. resumed config不匹配、新实验与旧专家/合成/尺度开关混用、非法fold输入须在读取数据前拒绝。
5. 单像素、单轴、消失对象和无截断图，HV有限且不改NP/TP标签。

## Task1: 目标和增强身份追踪

Files: src/nucseg/cellvit/data.py、辅助visibility监督模块、tests/test_visibility_training.py。
Interfaces: original_border_ids(inst)；VisibilityCrop.truncated_ids；visibility_hv_targets(inst,arm,truncated_ids)->(hv,counts)。

- [ ] 写合成掩码测试：原完整对象裁去像素与仅触边未裁去像素区分；原边界对象排除；A2仅修改对应实例HV，A1与既有hv_targets等价，A3等于min_size1。
- [ ] 跑RED并记录失败：`python -m pytest tests/test_visibility_training.py -q`。
- [ ] 裁剪前采集ID像素支持，裁剪后resize前比较丢失像素；追踪状态不调用随机函数。最终几何变换后生成HV；保留原背景和非干预对象的目标。
- [ ] 添加真实增强RNG比较、旋转/翻转、弹性形变、退化对象及数据集集成测试；GREEN并审阅修改范围。

## Task2: config/CLI保护和训练集成

Files: src/nucseg/cellvit/engine.py、scripts/train_cellvit.py、相关CLI/config测试。
Interfaces: TrainConfig.visibility_arm: str|None；`--visibility-arm {A0,A1,A2,A3}`。

- [ ] 写RED测试要求四臂flag需要split1/train-only；引擎直接调用也要求train_folds=[1],val_folds=[2]。
- [ ] 将字段传给训练数据集；验证数据集保持原图和统一目标口径。仅新轮次seed Python random，保留旧运行默认行为。
- [ ] 拒绝与upscale、非默认hv_min_size、copy-paste、synth、额外NP权重、dead_expert/widen等组合；配置恢复不允许换臂。
- [ ] 跑专用测试和全量`~/.conda/envs/nuclei/bin/python -m pytest -q`；保存日志和退出码。

## Task3: 规格批准记录、目录规范与独立审查

- [ ] 将设计状态记录为用户批准dev实现/训练，保留既有门和输入规则。目录分类结果/results、审计/audits、旧建议/proposals、背景/background、文献/literature、参考/reference；根入口、报告自含布局和superpowers规格/计划保持稳定。
- [ ] 重写搬迁文件的全部入链/出链，检查Markdown/TeX引用；冻结JSON/PDF不改变。
- [ ] 独立评审生产代码、测试和规格一致性；重要发现修复后重新跑必要回归。不暂存其他并行任务的内容。

## Task4: smoke、预算和正式训练队列

唯一新round目录：`runs/visibility_supervision_20261014/`；每run名`A0_seed19`等。运行命令模板：

```bash
~/.conda/envs/nuclei/bin/python scripts/train_cellvit.py --split 1 \
  --visibility-arm A0 --out runs/visibility_supervision_20261014/A0_seed19 \
  --epochs 130 --unfreeze-epoch 25 --batch 16 --lr 0.0003 \
  --workers 8 --val-every 5 --seed 19 --train-only
```

- [ ] 无旧round和重复进程后，在实时可用A100上以batch16做冻结/解冻各一次真实forward/backward/optimizer步骤，检查loss、梯度有限和峰值显存，保存smoke.json。仅读fold1。
- [ ] 预算：8个最终模型约10.8GiB；一次运行原子替换last临时峰值约8–10GiB。每run开始至少保留14GiB可写空间；新完成run仅保留final模型与日志，失败last保留。GPU按smoke峰值加4GiB余量检查。
- [ ] 忽略ops保存完整训练命令、冻结配置/数据/权重/代码SHA、commit、GPU UUID、PID和日志；源码快照+共享只读数据和权重供长期训练，避免后续编辑改变运行代码。
- [ ] 串行顺序A0/A1/A2/A3 seed19，然后seed1；每run前重查资源。资源不足时仅本机等待并记录状态；程序失败不重试、不跳过；输出存在且未有队列证据时拒绝启动。
- [ ] 正式队列启动后检查真实训练PID和至少首个epoch进度，再更新研究入口。不可把排队或running写成训练完成。

## Task5: dev评估交接

- [ ] 最后checkpoint在fold2原图及固定E4 C2窗口推理，nucseg.metrics.pannuke_eval评估；无TTA、同解码、配对图像聚类bootstrap，门只按原spec判断。正式队列先在每个run后执行原图fold2预测/评估，C2机制比较与最终门判定留给八run完成后。
- [ ] 训练启动交接记录清楚：本次完成实现/启动，8次训练与最终端点仍在执行或待执行。队列不自动启动确认轮，不读fold3。

## Task5 completion — 2026-10-15

All8 original-image and fixed-C2 evaluations complete. Mechanism clause1 and original-image clauses2/3 fail; see `docs/results/VISIBILITY_SUPERVISION_MECHANISM_2026-10-15.md`. No confirmation or extra training. New runner52 focused checks pass, independent review and raw-record/hash recomputation complete. Earlier unchecked steps are the original execution checklist, not current queue status.
