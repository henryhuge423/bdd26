# 门控多尺度融合 dev 轮：实施与执行计划（E5）

对应[冻结规格](../specs/2026-10-15-fusion-gating-design.md)。规格先于结果固定；本计划只安排实现、验证与执行顺序。

## Task 1 — 门控模块与解码路径（TDD）

- `src/nucseg/cellvit/model.py`：新增 `FUSION_MODES = ("plain", "gate", "adapter")` 与 `SkipGate`；
  `CellViTUNI(fusion=...)` 为 `"plain"` 时 `self.gates` 为空 `ModuleList`（**不新增任何参数名或 state_dict 键**，历史 run 逐位可复现）；
  `gate`/`adapter` 时按 skip0..skip3 的 64/128/256/512 通道各加一个 1×1 卷积并零初始化。
  `forward` 中在构建 skips 后、送入各分支前统一施加一次变换（四个分支共享同一批已变换 skips，保持原"共享 skip"语义）。
- 新增 `fusion_param_count` / `total_param_count`，`branch_param_count` 在非 plain 下追加 `gates` 组。
- 变换定义（冻结）：`gate: s' = s * (C * softmax(W(s), dim=1))`；`adapter: s' = s + W(s)`。
  零初始化时两者都精确等于恒等（`C*softmax(0)=1`，C 为 2 的幂；`W(0)=0`）。

**红灯**：`tests/test_fusion_gating.py` 先写，先跑出 3 个失败（`SkipGate` 未定义、`fusion` 入参不存在、`fusion_param_count` 不存在）。
**绿灯**：实现后全部通过。

## Task 2 — 配置、断点与 CLI 守卫

- `engine.py`：`TrainConfig.fusion`（默认 `"plain"`）；`__post_init__` 拒绝未知取值，并把 `fusion != "plain"` 加入可见性臂的"拒绝额外科学干预"条件；
  `build_model` 透传 `fusion`；`train()` 对非 plain 强制 fold1/fold2；新增 `run_fusion(run)` 供推理读取 run 的 config。
- `train_cellvit.py`：`--fusion {plain,gate,adapter}`；非 plain 必须 `--split 1 --train-only`，且不得与 `--visibility-arm` 并用。
- `predict_cellvit.py`：cfg shim 增加 `fusion=run_fusion(run)`，使 fold2 推理按 run 的 config 重建门控结构。
- 兼容性：既有 run 的 `config.json` 无 `fusion` 键 → 以默认 `"plain"` 比较，不触发 mismatch；新增测试覆盖。

## Task 3 — 验证

1. `~/.conda/envs/nuclei/bin/python -m pytest -q tests/test_fusion_gating.py`（先）
2. `~/.conda/envs/nuclei/bin/python -m pytest -q`（全量回归；改动触及训练/模型路径，必须全量）
3. 单次短 smoke（真实 fold1、batch16、冻结/解冻各一步）：有限损失、有限梯度、峰值显存、步时；不用于筛选超参。
4. 参数预算核对：新增 349,120 参数，占总参数 362,953,920 的 0.096%（<0.1%）。

## Task 4 — 执行（阶段 1）

队列（在**LM2** 执行，代码经 `ops/scripts/push_lm2.sh` 同步；LM1 只保留代码、文档与小体积端点产物）：

| run | 命令要点 | 输出 |
|---|---|---|
| `G_seed19` | `train_cellvit.py --split 1 --out <LM2 runs>/G_seed19 --fusion gate --epochs 130 --unfreeze-epoch 25 --batch 16 --lr 3e-4 --workers 8 --val-every 5 --seed 19 --train-only` | 训练 + `final.pth` |
| `G_seed19` 评估 | `predict_cellvit.py --run <...>/G_seed19 --fold 2 --batch-size 16 --no-inst-probs` | `pred_fold2.npz`、`eval_fold2/` |
| `G_seed1` | 同上，`--seed 1` | 同上 |

- 每 run 开始前复核：所选 GPU 空闲显存 ≥ smoke 峰值 + 4GiB，磁盘 ≥ 14GiB；失败即停，不自动重试或追加菜单。
- **GPU 约束**：不使用 LM1 GPU2（用户 2026-10-15 要求）；LM1 只用 GPU0/1/4/6；不使用 LM2 上用户正在运行的卡（当前 3–6）。
- 每 run 成功且 `final.pth` 校验为完整有限权重后释放 `last.pth` 优化器状态；`final.pth` 保留在 LM2。
- 回传 LM1（权威工作区）：`config.json`、`log.jsonl`、`pred_fold2.npz`、`eval_fold2/`、checkpoint SHA256 与 LM2 路径。

## Task 5 — 端点判定

- 新脚本 `scripts/analyze_fusion_arms.py`：对任意 run 目录计算 fold2 原图端点
  （official mPQ/bPQ/Dead PQ/strict Dead PQ 来自各自 `eval_fold2/summary.json`；
  内部 Dead 匹配率/缺失率复用 `nucseg.visibility_probe.max_iou_table` + `original_full` 与
  `scripts/probe_visibility.gt_class_map`，与可见性机制报告同一实现与同一 741 对象集合）。
- 配对 bootstrap：组织分层图像聚类，1,000 次抽样，固定种子 20261015，G/C 与 A0 共用同一批重抽样索引；
  按规格 §4 的三条疗效条件与四条质量护栏判定，输出 JSON 与中文结果文档。
- 阶段 2（`adapter` 臂）仅在疗效条件满足时运行，命令与阶段 1 相同但 `--fusion adapter`。

## Task 6 — 归档与文档

- 结果文档 `docs/results/FUSION_GATING_DEV_2026-10-1x.md`（含 commit、config/输入哈希、命令、端点表、判定与边界）。
- 更新 `docs/findings.md`、`docs/results/README.md`、`README.md` 当前状态行、`docs/RESEARCH_PLAN.md` 进展记录。
- 无论正负都如实写结论：负结果按停止规则收档；通过只授权**提出**三 split 确认轮（另立项）。
- 分支 `research/fusion-gating-e5`；按项目惯例提交，不主动推送 GitHub（除非用户要求）。

## 成本估算

单 run ≈ 2.5–4 A100 小时（与 A0 同配方、同规模）；阶段 1 两次训练 + 两次 fold2 推理 ≈ 6–9 GPU 小时；
阶段 2 另加同等量。存储：LM2 每 run ≈ 1.8GB（final 1.45GB + 预测/日志）；LM1 回传每 run ≈ 0.3GB。

## 回滚

实现为加法、默认关闭：`fusion="plain"` 时 `state_dict`、参数字典与数值路径与改动前完全一致，
既有 run 全部照常加载与推理。如判定失败，删除阶段 1 的 run 目录与 `--fusion` 通路即可，
不影响任何历史工件。
