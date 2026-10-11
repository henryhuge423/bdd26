# E1-T 跨检测器分型共识审计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结缓存预测上量化"基模型分错类的已匹配实例"上跨折/跨架构 jury 的类型投票方向，产出 E1-T spec §2 的全部统计量与解读键判定。

**Architecture:** 纯逻辑模块 `src/nucseg/e1_typing.py`（配对/投票/多数类/聚类 bootstrap，不碰磁盘）+ 薄 CLI `scripts/e1t_typing_consensus.py`（载入 base+12 jury npz 与 fold2 GT，逐图调逻辑，JSON+Markdown 输出，fold-guard）。复用 `nucseg.metrics.instance.overlap` 与 `scripts/fold_guard.ensure_dev_fold`。

**Tech Stack:** Python 3.10, numpy 1.23.5（pinned）, pytest。

**Spec:** `docs/superpowers/specs/2026-10-11-e1t-typing-consensus-design.md`（统计量与解读键以 spec 为准）。

## Global Constraints

- numpy 1.23.5 / torch 2.5.1+cu124 环境为 `~/.conda/envs/nuclei`；不新增依赖。
- `np.load` 的 NpzFile 必须先物化整个成员数组再逐图索引（2026-10-08 E0 教训）；实现中用
  `with np.load(p) as z: inst = z["inst"]; typ = z["type"]`（inst/type 是 fold 级堆叠数组，
  索引 [j] 得单图 2D int 数组——与 `run_existence_ef.py` 相同布局）。
- 只读 fold2（split1 val 视角）：`ensure_dev_fold(2, 1)`；不读 fold3。
- 无新推理、无训练、不改任何冻结结果。
- 代码进 `src/`/`scripts/`，工件进 `runs/analysis/e1t_typing_20261011/`（本地不跟踪）。
- git：research 分支 `research/e3-e1t`，逐任务提交。

## Review Focus

- 基-GT 配对歧义（一个 GT 与两个 base 实例 IoU 并列最大）→ 应排除并单列计数，不得悄悄取先者。
- jury 成员对某 GT 的"票"必须是该成员与之**最大 IoU** 的实例且 IoU≥0.3；低于门槛不计票、不弃权整个成员。
- 多数类定义必须与基/GT 无关（G0 不退化）；平票/总票<3/缺训练折 → "无多数"，比例单列。
- 同图多实例非独立 → bootstrap 必须按图像整簇重采样。
- 输出不得含生物学断言措辞；所有率带分子分母。
  - 对应测试：Task 1 test_ambiguous_pair_excluded / test_vote_threshold / test_majority_clauses /
    test_bootstrap_clusters_by_image / Task 2 端到端 fixture 校验分组计数。

---

### Task 1: 纯逻辑模块 `src/nucseg/e1_typing.py`

**Files:**
- Create: `src/nucseg/e1_typing.py`
- Test: `tests/test_e1_typing.py`

**Interfaces:**
- Consumes: `nucseg.metrics.instance.overlap`（iou 矩阵）。
- Produces:
  - `pair_base(gt, base, thr=0.5) -> dict[int, tuple[int, float] | None]`：GT id → (base id, IoU)
    或 None（无 IoU>thr 或并列最大）。并列 = 两个 base 实例 IoU 差 < 1e-12 且为最大。
  - `jury_votes(gt, member_inst, member_type, thr=0.3) -> dict[int, tuple[int, int] | None]`：
    GT id → (该成员最佳重叠实例 id, 其 type)，仅当最大 IoU ≥ thr；否则该 GT 无此成员票。
  - `majority_type(votes: dict[int, int], folds: dict[int, int], min_votes=3) -> tuple[int | None, dict]`：
    votes = {member_idx: type}，folds = {member_idx: train_fold}。返回 (多数类或 None, 诊断
    dict{n_votes, counts, both_folds_ok})。多数类须：总票≥3、最高严格多于次高、最高类收到
    两个不同 train_fold 的票。
  - `clustered_rate_bootstrap(indicator, image_ids, n_boot=2000, seed=20261011, stat="mean") -> (lo, hi)`：
    indicator 为逐实例 0/1（或 NaN=不计入分母），按 image 整簇重采样的百分位 95% CI。

- [ ] **Step 1: 写失败测试**（覆盖 Review Focus 全部 5 项）

```python
import numpy as np
import pytest
from nucseg.e1_typing import pair_base, jury_votes, majority_type, clustered_rate_bootstrap


def _rect(inst, i, y, x, h, w):
    inst[y:y+h, x:x+w] = i


def test_pair_base_ambiguous_excluded():
    gt = np.zeros((16, 16), np.int32); _rect(gt, 1, 2, 2, 6, 6)
    base = np.zeros((16, 16), np.int32); _rect(base, 5, 2, 2, 6, 3); _rect(base, 7, 2, 5, 6, 3)
    out = pair_base(gt, base)
    assert out[1] is None  # IoU 并列最大 → 歧义排除


def test_pair_base_plain():
    gt = np.zeros((16, 16), np.int32); _rect(gt, 1, 2, 2, 6, 6); _rect(gt, 2, 9, 9, 4, 4)
    base = np.zeros((16, 16), np.int32); _rect(base, 5, 2, 2, 6, 6); _rect(base, 6, 10, 10, 3, 3)
    out = pair_base(gt, base)
    assert out[1] == (5, 1.0) and out[2] is None  # GT2 与 base6 IoU=9/16<0.5


def test_jury_vote_threshold_and_best_instance():
    gt = np.zeros((16, 16), np.int32); _rect(gt, 1, 2, 2, 6, 6)
    m = np.zeros((16, 16), np.int32); _rect(m, 3, 2, 2, 6, 6)
    mt = np.zeros((16, 16), np.uint8); mt[m == 3] = 2
    assert jury_votes(gt, m, mt)[1] == (3, 2)
    assert jury_votes(gt, m, mt, thr=0.95)[1] == (3, 2)
    far = np.zeros((16, 16), np.int32); _rect(far, 4, 10, 10, 4, 4)
    ft = np.zeros((16, 16), np.uint8); ft[far == 4] = 1
    assert jury_votes(gt, far, ft)[1] is None  # IoU=0 < 0.3


def test_majority_clauses():
    folds = {0: 1, 1: 1, 2: 3, 3: 3, 4: 3}
    maj, d = majority_type({0: 2, 1: 2, 2: 2, 3: 1}, folds)
    assert maj == 2 and d["n_votes"] == 4 and d["both_folds_ok"]
    assert majority_type({0: 2, 1: 2, 3: 1}, folds)[0] is None        # 两个训练折缺一
    assert majority_type({0: 2, 1: 2, 2: 1, 3: 1}, folds)[0] is None  # 平票
    assert majority_type({0: 2, 1: 2}, folds)[0] is None              # 总票<3


def test_bootstrap_clusters_by_image():
    ind = np.array([1.0, 0.0, 1.0, 1.0])
    img = np.array([0, 0, 1, 1])
    lo, hi = clustered_rate_bootstrap(ind, img, n_boot=500, seed=7)
    assert 0.0 <= lo <= hi <= 1.0
    # 全 1 → CI 收敛到 1
    lo2, hi2 = clustered_rate_bootstrap(np.ones(4), img, n_boot=500, seed=7)
    assert lo2 == hi2 == 1.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data7/jinxinhao/bdd26 && ~/.conda/envs/nuclei/bin/python -m pytest tests/test_e1_typing.py -q`
Expected: FAIL（ModuleNotFoundError: nucseg.e1_typing）

- [ ] **Step 3: 最小实现**

```python
"""E1-T pure logic: base-GT pairing, jury typing votes, majority rule, clustered bootstrap.

All functions are side-effect free and defined WITHOUT reference to the base model's own
prediction except `pair_base`, per spec 2026-10-11 §2 (G0 non-degeneracy)."""
from __future__ import annotations

import numpy as np

from nucseg.metrics.instance import overlap


def pair_base(gt, base, thr=0.5):
    ov = overlap(gt, base)
    out = {}
    for g in range(ov.iou.shape[0]):
        col = ov.iou[g]
        j = int(np.argmax(col)) if col.size else -1
        if j < 0 or col[j] <= thr:
            out[g + 1] = None
            continue
        ties = np.flatnonzero(np.isclose(col, col[j], rtol=0, atol=1e-12))
        out[g + 1] = None if len(ties) > 1 else (j + 1, float(col[j]))
    return out
```
（`jury_votes` 用同一 overlap 矩阵列最大值；`majority_type` 与
`clustered_rate_bootstrap` 按 Interfaces 签名直接实现；bootstrap 用
`np.random.default_rng(seed)`，按唯一图像索引重采样，NaN 指标剔除于分母。）

- [ ] **Step 4: 跑测试确认通过**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_e1_typing.py -q`
Expected: PASS（5 项全绿）

- [ ] **Step 5: Commit**

```bash
git add src/nucseg/e1_typing.py tests/test_e1_typing.py
git commit -m "feat(e1t): pairing, jury typing votes, base-independent majority rule, clustered bootstrap"
```

---

### Task 2: CLI `scripts/e1t_typing_consensus.py`（端到端 + 端到端测试）

**Files:**
- Create: `scripts/e1t_typing_consensus.py`
- Test: `tests/test_e1t_runner.py`

**Interfaces:**
- Consumes: Task 1 全部函数；`PanNukeFold(2)`（fold2 GT：`f.inst`/`f.type`，长度 = 图数）；
  jury 布局 = E1-C 同款（`matched_seed_20261004/p0/split{1,3}_seed{19,1,2}/val/{base,x2}/identity/pred.npz`
  11 席 − 基 run split1_seed1/val/base → 10 席 + `e1_consensus_20261010/hovernet_split{1,3}_fold2/pred.npz`
  2 席）；`scripts/fold_guard.ensure_dev_fold`。
- Produces: `runs/analysis/e1t_typing_20261011/{stats.json, tables.md}`；列结构：
  - `groups`: {"G1": n, "G0": n, "ambiguous": n, "unmatched_gt": n}
  - `overall`: G1 {maj_eq_gt, maj_eq_base, maj_other, no_majority} 计数；G0 {maj_eq_gt,
    maj_neq_gt, no_majority} 计数（分母 = 有多数类之外的实例也计入 no_majority 分母之外单列）
  - `by_pair`: {(gt_class, base_class): 同 G1 四格} × 20 个非对角格
  - `bootstrap`: 总体与 Inf↔Conn 双向、X→Dead、Dead→X 的 (lo, hi)
  - `sensitivity`: {hn_only, iou_thr_05, min_votes_5} 三套同构统计
  - `provenance`: 全部输入 SHA256、代码指纹、菜单/参数（含 seed 20261011）

- [ ] **Step 1: 写失败测试**（合成 2 图 × 3 席迷你 fixture，走 runner 主函数 `analyze()`）

```python
import json
import numpy as np
import pytest
from pathlib import Path
from scripts_helper import import_from  # 测试内已有点缀 import 约定则沿用，否则直接
from e1t_typing_consensus import analyze  # scripts/ 下模块，tests 已有 sys.path 先例


def _mini_world(tmp_path):
    # GT: img0 两个核(class 1,2)，img1 一个核(class 4=Dead)
    # base: img0 GT1 分错(说 2)，GT2 分对；img1 分错(说 1)
    # jury: 3 席 CellViT(train_fold 1) + 2 席 HN(train_fold 3)
    # 期望: G1 n=2, G0 n=1；构造一例多数=基类、一例多数=GT、一例 no_majority
    ...
```
（fixture 具体坐标在实现时按 Task 1 的 `_rect` 风格写全；断言：
`analyze()` 返回的 groups/overall 计数与手工推演一致；fold-guard 用 monkeypatch 拒绝
fold3 调用；`--out` 已存在时拒绝覆写。）

- [ ] **Step 2: 跑测试确认失败**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_e1t_runner.py -q`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 CLI**

结构（镜像 `scripts/e1_consensus.py` 的参数与护栏风格）：
- `builtin_jury(round_root)`：返回 12 席（E1-C 同款枚举**减去基 run** `split1_seed1/val/base`，
  即 split1_seed{19,2}×{x1,x2}=4 席 + split3_seed{19,1,2}×{x1,x2}=6 席）+ 2 席 HN extra-jury；
  每席带 `{"name", "path", "arch", "train_fold"}`。
- `main()`：`ensure_dev_fold(2, 1)`；逐图循环——先物化各 npz 的 `inst`/`type` 堆叠数组
  （NpzFile 教训），`pair_base` 定 G0/G1/ambiguous，对每个 matched GT 收 12 席
  `jury_votes`，`majority_type` 判多数类；按 spec §2 聚合四格计数与 by_pair 表；
  bootstrap 与敏感性三套；写 JSON+Markdown；`--out` 存在即 `SystemExit`。
- 类型注意：`f.type`/npz `type` 与类目 0..5 一致（0=背景）；GT 类取 `f.type[gt==g]`
  的唯一值（官方 gt 通道语义同 audit 代码用 `instance_classes`，这里 GT 侧直接
  `np.unique` 校验唯一后取值，冲突即中止）。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_e1t_runner.py tests/test_e1_typing.py -q && ~/.conda/envs/nuclei/bin/python -m pytest -q`
Expected: 新测试 PASS；全量无回归（此前 277 passed 基线）

- [ ] **Step 5: Commit**

```bash
git add scripts/e1t_typing_consensus.py tests/test_e1t_runner.py
git commit -m "feat(e1t): end-to-end typing-consensus runner with fold guard and provenance"
```

---

### Task 3: 真实数据执行 + 判定记录（执行任务，无新代码）

- [ ] **Step 1: 运行**

```bash
cd /data7/jinxinhao/bdd26 && ~/.conda/envs/nuclei/bin/python scripts/e1t_typing_consensus.py \
  --round runs/analysis/matched_seed_20261004 \
  --extra-jury runs/analysis/e1_consensus_20261010/hovernet_split1_fold2/pred.npz:HoverNet:hovernet:split1 \
               runs/analysis/e1_consensus_20261010/hovernet_split3_fold2/pred.npz:HoverNet:hovernet:split3 \
  --out runs/analysis/e1t_typing_20261011
```
（若基 run 排除逻辑内建则无需额外参数；预计 CPU 数十分钟。）

- [ ] **Step 2: 核验**

- 抽 3 个图像手工复核四格归类（用临时只读脚本打印该图 GT/base/jury 类型）。
- G0+G1+ambiguous+unmatched_gt 总数 == fold2 GT 实例总数（与 existence audit 的
  59,593 量级对照，差异即报警）。

- [ ] **Step 3: 按 spec §2 解读键写判定进 findings.md（新条目 2026-10-11 E1-T）**

- 主表：G1 四格（总体 + Inf↔Conn 双向 + Dead 相关格）、G0 本征翻票率、CI；
- 判定文字按预注册键二选一，Dead 单列描述；
- 记录 ambiguous/no_majority 比例与敏感性三套的一致性。

- [ ] **Step 4: Commit findings + 分支**

```bash
git add docs/findings.md
git commit -m "docs: E1-T typing-consensus verdict (pre-registered key)"
```
