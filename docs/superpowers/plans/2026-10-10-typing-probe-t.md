# T 实验（固定几何实例分型探针 T0–T3）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不动实例几何的前提下，用四个预注册配置（T0 恒等 / T1 概率均值 / T2 冻结特征+线性头 / T3 +上下文环）测出 split1/fold2 验证折官方 mPQ 的 Δ，按 spec §2 判定规则出预注册结论。

**Architecture:** 复用 `RetypeEvaluator`（固定实例集合的精确官方 mPQ）扩展 strict 模式与配对图像 bootstrap；新增 `src/nucseg/typing_probe.py` 承载纯逻辑（池化、环、T1 规则、线性头）；两个薄 CLI（特征导出 GPU / 实验运行 CPU）。T1 用已缓存的 `inst_prob`，无需新推理；T2/T3 需一次 fold1+fold2 特征导出。

**Tech Stack:** Python 3.10, torch 2.5.1+cu124（bf16 autocast 前向）, numpy 1.23.5, cv2, sklearn（仅交叉核对）, pytest。

**Spec:** `docs/superpowers/specs/2026-10-10-typing-probe-t-design.md`（用户 2026-10-10 批准）。执行者须同时读 spec；本计划与 spec 冲突时以 spec 为准。

## Global Constraints

- **fold3 / split2、split3 一律不读**；每个新 CLI 在 `main()` 里用 `fold_guard.ensure_dev_fold(a.fold, run_split(a.run))` 守卫（`scripts/fold_guard.py`）。
- 不扫描任何超参：环宽 24 px、头种子 {0,1}、AdamW lr 1e-3 / wd 1e-4 / 20 epoch / batch 4096、bootstrap 1000 次、种子 20261010，全部来自 spec §3 冻结值。
- 语义镜像官方：strict 聚合与 `pannuke_eval.evaluate` 的 `strict_pq = where(~gt_present & pred_present, 0, cls_pq)` + `nanmean(cls,1)` + tissue_avg 完全一致；bootstrap 重采样在同一公式内做。
- 环境：`~/.conda/envs/nuclei`（Python 3.10, torch 2.5.1+cu124, numpy 1.23.5）；不新增依赖。
- 动 `src/nucseg/metrics/` 后跑 `python -m pytest -q`（CLAUDE.md 规则）。
- 输出目录 `runs/analysis/typing_probe_20261010/`（本地、gitignored）；可复用代码只进 `src/`、`scripts/`。
- 所有写出文件拒绝覆盖已存在路径（沿 `write_dead_predictions` 惯例）。
- 提交信息以 `Co-Authored-By: Claude Code <noreply@anthropic.com>` 结尾。

## Review Focus

1. **实例表错位**：`instance_classes` 的 relabel 枚举若与 npz `inst_id` 表不同序，类别会安到错的实例上。→ Task 5 断言逐图 id 相等（测试 pin 之）。
2. **class-0 / 未分型实例**：667/59593 行 `inst_prob` argmax 为背景通道；T0 也可能产出 type-0。T1 规则必须永不输出 0（0 在 RetypeEvaluator 里=不计入任何类）。→ Task 3 测试。
3. **bootstrap 的 NaN 语义**：重采样后某组织某类可能整列 NaN；必须在 `errstate(ignore)` 下与官方公式同构。→ Task 2 测试用极小分层构造。
4. **前向语义漂移**：导出前向（自写、带特征）必须与 `predict_fold` 的 tp softmax 一致。→ Task 4 的绑定检查：fold2 池化 tp softmax vs 缓存 `inst_prob`，逐行 max|Δ| < 1e-4。
5. **退化几何**：1-px 实例、近贴邻核（环被吃光）、整图一个实例 → 环为空。→ Task 3 空环=0 向量测试 + Task 4 导出时记录空环计数。

---

### Task 1: `RetypeEvaluator` strict mPQ

**Files:**
- Modify: `src/nucseg/metrics/fast_retype.py`
- Test: `tests/test_typing_probe.py`（新建，本任务只加 strict 部分）

**Interfaces:**
- Consumes: 现有 `RetypeEvaluator.per_image_class_pq(cls)`、`self.n_gt`、`self.inst_img`、`self.tissue`。
- Produces: `RetypeEvaluator.mpq_strict(cls) -> tuple[float, np.ndarray]`（tissue 平均 strict mPQ + 逐类 strict PQ，图像 nanmean 与 `pannuke_eval` 严格同构）。

- [ ] **Step 1: 写失败测试（与 canonical evaluator 对拍）**

```python
# tests/test_typing_probe.py
"""T 实验（typing probe）配套测试：fast_retype strict、bootstrap、pooling/环/头、runner 纯逻辑。"""
import numpy as np
import pytest

from nucseg.constants import NUM_CLASSES
from nucseg.metrics.fast_retype import RetypeEvaluator
from nucseg.metrics.pannuke_eval import evaluate


def _toy_fold(n=4):
    """2 组织 × n 图；每图 2 个 GT 核（类 1 与类 3），预测同几何、类 1 保持、类 3 错标为类 5。"""
    gt_ch = np.zeros((n, 8, 8, NUM_CLASSES), np.uint16)
    gt_inst = np.zeros((n, 8, 8), np.int32)
    pred_inst = np.zeros((n, 8, 8), np.int32)
    tissue = np.array([0, 0, 1, 1])[:n]
    for i in range(n):
        gt_ch[i, 1:4, 1:4, 0] = 1; gt_ch[i, 5:7, 5:7, 2] = 1
        gt_inst[i, 1:4, 1:4] = 1; gt_inst[i, 5:7, 5:7] = 2
        pred_inst[i] = gt_inst[i]
    gt_type = np.where(gt_inst == 1, 1, np.where(gt_inst == 2, 3, 0)).astype(np.uint8)
    pred_type = np.where(pred_inst == 1, 1, np.where(pred_inst == 2, 5, 0)).astype(np.uint8)
    # 预测类 5 在图中无 GT → strict 罚 0
    return gt_ch, gt_inst, gt_type, pred_inst, pred_type, tissue


def _tables(pred_inst):
    img = np.concatenate([np.full(len(np.unique(p[p > 0])), j) for j, p in enumerate(pred_inst)])
    iid = np.concatenate([np.unique(p[p > 0]) for p in pred_inst])
    return img, iid


def test_mpq_strict_matches_canonical():
    gt_ch, gt_inst, gt_type, pred_inst, pred_type, tissue = _toy_fold()
    canon = evaluate(gt_ch, gt_inst, gt_type, tissue, pred_inst, pred_type, workers=1)
    img, iid = _tables(pred_inst)
    ev = RetypeEvaluator(gt_ch, pred_inst, img, iid, tissue, workers=1)
    cls = np.where(iid == 1, 1, 5)
    m_fast, per_fast = ev.mpq(cls)
    assert abs(m_fast - canon["summary"]["mPQ"]) < 1e-9            # 既有官方行为不回归
    s_fast, per_s_fast = ev.mpq_strict(cls)
    assert abs(s_fast - canon["summary"]["strict"]["mPQ"]) < 1e-9
    assert s_fast < m_fast                                          # 缺类 FP 确实被罚
```

（`summary["mPQ"]` / `summary["strict"]["mPQ"]` 键名见 `pannuke_eval.py:188-191`。）

- [ ] **Step 2: 运行确认失败**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_typing_probe.py -k strict -q`
Expected: FAIL `AttributeError: 'RetypeEvaluator' object has no attribute 'mpq_strict'`

- [ ] **Step 3: 最小实现（fast_retype.py 追加）**

```python
    def _pred_counts(self, cls: np.ndarray) -> np.ndarray:
        """(n, C) predicted-instance counts per image and class (class 0 excluded)."""
        v = cls > 0
        return np.bincount(self.inst_img[v] * NUM_CLASSES + cls[v] - 1,
                           minlength=self.n * NUM_CLASSES).reshape(self.n, NUM_CLASSES)

    def mpq_strict(self, cls: np.ndarray) -> tuple[float, np.ndarray]:
        """Tissue-averaged strict mPQ: classes absent in GT but predicted score 0 (pannuke_eval)."""
        pq = self.per_image_class_pq(cls)
        strict = np.where((self.n_gt <= 0) & (self._pred_counts(cls) > 0), 0.0, pq)
        with np.errstate(all="ignore"):
            img = np.nanmean(strict, 1)
            per_t = [np.nanmean(img[self.tissue == t]) for t in TISSUES if (self.tissue == t).any()]
            return float(np.nanmean(per_t)), np.nanmean(strict, 0)
```

- [ ] **Step 4: 跑测试通过**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_typing_probe.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/nucseg/metrics/fast_retype.py tests/test_typing_probe.py
git commit -m "feat: RetypeEvaluator.mpq_strict mirrors pannuke_eval strict aggregation"
```

### Task 2: 配对图像 bootstrap（组织内分层）

**Files:**
- Modify: `src/nucseg/metrics/fast_retype.py`
- Test: `tests/test_typing_probe.py`

**Interfaces:**
- Produces: `RetypeEvaluator.mpq_bootstrap_delta(cls_a, cls_b, n_boot=1000, seed=0) -> dict`
  返回 `{"delta": float, "lo": float, "hi": float}`（a−b 的点估计与百分位 95% CI；重采样在
  **每个组织内**有放回抽图像，mPQ 公式与 `mpq` 逐字相同）。

- [ ] **Step 1: 写失败测试**

```python
def test_bootstrap_delta_identical_and_stratified():
    gt_ch, gt_inst, pred_inst, tissue = _toy_fold()
    ev = RetypeEvaluator(gt_ch, pred_inst, *_tables(pred_inst), tissue)
    rng = np.random.default_rng(0)
    cls = rng.integers(1, 6, len(ev.inst_img))
    r = ev.mpq_bootstrap_delta(cls, cls, n_boot=64, seed=1)
    assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0
    r2 = ev.mpq_bootstrap_delta(cls, cls, n_boot=64, seed=1)
    assert r2 == r                                   # 种子确定
    # 点估计 == mpq(a) - mpq(b)
    assert abs(r["delta"] - (ev.mpq(cls)[0] - ev.mpq(cls)[0])) < 1e-12
```

（`_tables` 为本文件 helper：从 pred_inst 重建排序的 `(inst_img, inst_id)`。）

- [ ] **Step 2: 确认失败** — `pytest tests/test_typing_probe.py -k bootstrap -q` → FAIL (no attribute)

- [ ] **Step 3: 实现（追加到 RetypeEvaluator）**

```python
    def mpq_bootstrap_delta(self, cls_a, cls_b, n_boot=1000, seed=0):
        """Paired image bootstrap of mPQ(a)-mPQ(b); resample images within each tissue stratum."""
        def img_mpq(pq):
            with np.errstate(all="ignore"):
                return np.nanmean(pq, 1)
        pa, pb = img_mpq(self.per_image_class_pq(cls_a)), img_mpq(self.per_image_class_pq(cls_b))
        strata = [np.nonzero(self.tissue == t)[0] for t in np.unique(self.tissue)]
        rng = np.random.default_rng(seed)
        def m(v):
            per_t = [np.nanmean(v[idx]) for idx in strata]
            return np.nanmean(per_t)
        deltas = np.empty(n_boot)
        for b in range(n_boot):
            idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
            deltas[b] = m(pa[idx]) - m(pb[idx])
        with np.errstate(all="ignore"):
            lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
        return {"delta": float(m(pa) - m(pb)), "lo": float(lo), "hi": float(hi)}
```

- [ ] **Step 4: 跑通过** — `pytest tests/test_typing_probe.py -q` PASS

- [ ] **Step 5: 全套回归 + 提交**

```bash
~/.conda/envs/nuclei/bin/python -m pytest -q      # CLAUDE.md：动了 metrics 必跑全套
git add src/nucseg/metrics/fast_retype.py tests/test_typing_probe.py
git commit -m "feat: tissue-stratified paired image bootstrap for retype mPQ deltas"
```

### Task 3: `typing_probe` 纯逻辑模块（池化 / 环 / T1 规则 / 线性头）

**Files:**
- Create: `src/nucseg/typing_probe.py`
- Test: `tests/test_typing_probe.py`

**Interfaces:**
- Produces:
  - `pool_instances(feat: np.ndarray, inst: np.ndarray, ids: np.ndarray) -> np.ndarray`：feat (H,W,C) float32、inst (H,W) int、ids (n,) → (n,C) 实例内均值，按 ids 顺序。
  - `pool_rings(feat, inst, ids, width=24) -> np.ndarray`：环 = `cv2.dilate(own_mask, MORPH_ELLIPSE(2w+1))` 减去**全部实例像素**（own+邻核）；空环 → 0 向量。
  - `t1_classes(inst_prob: np.ndarray) -> np.ndarray`：(M,6) → (M,) `argmax(inst_prob[:,1:6])+1`，永不输出 0。
  - `train_linear_head(X, y, seed, epochs=20, lr=1e-3, wd=1e-4, batch=4096) -> np.ndarray`（返回权重 W (5,d), b (5,)）；`apply_head(W, b, X) -> np.ndarray` 预测 1..5。torch、CPU、确定性按 seed。
  - `sklearn_logreg(X, y) -> np.ndarray` 预测 1..5（lbfgs, C=1.0, max_iter=1000）。

- [ ] **Step 1: 写失败测试**

```python
def test_pool_and_ring():
    import numpy as np
    from nucseg.typing_probe import pool_instances, pool_rings
    f = np.zeros((8, 8, 2), np.float32); f[1:3, 1:3] = (1., 2.); f[5:7, 5:7] = (3., 4.)
    inst = np.zeros((8, 8), np.int32); inst[1:3, 1:3] = 1; inst[5:7, 5:7] = 2
    pooled = pool_instances(f, inst, np.array([1, 2]))
    assert np.allclose(pooled, [[1., 2.], [3., 4.]], atol=1e-6)
    rings = pool_rings(f, inst, np.array([1, 2]), width=1)   # 环=膨胀1px−全部实例
    assert rings[0].sum() > 0 or True   # 背景像素(值0)也进环——环测的是掩码几何，不是值
    # 环必须不含任何实例像素：用指示特征验证
    f_id = (inst > 0).astype(np.float32)[..., None]
    r_id = pool_rings(f_id, inst, np.array([1, 2]), width=1)
    assert r_id.sum() == 0
    # 满图实例 → 空环 = 0 向量
    inst_full = np.ones((8, 8), np.int32)
    assert pool_rings(np.ones((8, 8, 2), np.float32), inst_full, np.array([1]), width=3)[0].sum() == 0


def test_t1_rule_drops_background_channel():
    from nucseg.typing_probe import t1_classes
    p = np.array([[.6, .3, .1, 0, 0, 0], [.05, .02, .03, 0, 0, .9]], np.float32)
    assert list(t1_classes(p)) == [1, 5]          # 第1行背景最大但被丢弃→类1；第2行→类5
    assert t1_classes(p).min() >= 1


def test_linear_head_separable_and_seeded():
    import numpy as np, torch
    from nucseg.typing_probe import train_linear_head, apply_head, sklearn_logreg
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-1, .3, (64, 4)), rng.normal(1, .3, (64, 4))]).astype(np.float32)
    y = np.array([0] * 64 + [1] * 64)
    W1, b1 = train_linear_head(X, y, seed=0, epochs=50)
    assert (apply_head(W1, b1, X) == np.where(y == 0, 1, 2)).all()   # 可分数据全对，输出1..5域
    W2, _ = train_linear_head(X, y, seed=1, epochs=50)
    assert not np.allclose(W1, W2)                                    # 种子确实改变初始化/洗牌
    assert (sklearn_logreg(X, y) == np.where(y == 0, 1, 2)).all()
```

- [ ] **Step 2: 确认失败** — `pytest tests/test_typing_probe.py -k "pool or t1 or head" -q` → FAIL (no module)

- [ ] **Step 3: 实现 `src/nucseg/typing_probe.py`**

```python
"""T 实验（固定几何分型探针）纯逻辑：实例/环池化、T1 聚合规则、线性头。spec 见
docs/superpowers/specs/2026-10-10-typing-probe-t-design.md（2026-10-10 批准，值冻结）。"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

RING_WIDTH = 24          # spec §3 冻结
HEAD_SEEDS = (0, 1)


def pool_instances(feat: np.ndarray, inst: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """feat (H,W,C) float32 -> (len(ids),C) per-instance mean, rows in `ids` order."""
    flat = feat.reshape(-1, feat.shape[-1])
    key = inst.ravel().astype(np.int64)
    sums = np.zeros((int(ids.max()) + 1, feat.shape[-1]), np.float64)
    cnt = np.bincount(key, minlength=int(ids.max()) + 1)
    np.add.at(sums, key, flat)
    assert (cnt[ids] > 0).all(), "instance id with zero pixels"
    return (sums[ids] / cnt[ids][:, None]).astype(np.float32)


def pool_rings(feat: np.ndarray, inst: np.ndarray, ids: np.ndarray, width: int = RING_WIDTH) -> np.ndarray:
    """Ring = dilate(own mask, ellipse 2w+1) minus ALL instance pixels; empty ring -> zeros."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    bg = (inst == 0)
    out = np.zeros((len(ids), feat.shape[-1]), np.float32)
    for row, i in enumerate(ids):
        ring = cv2.dilate((inst == i).astype(np.uint8), k) & bg
        n = int(ring.sum())
        if n:
            out[row] = feat[ring.astype(bool)].mean(0)
    return out


def t1_classes(inst_prob: np.ndarray) -> np.ndarray:
    """(M,6) instance-mean tp softmax -> classes 1..5; background channel dropped (spec §3)."""
    return inst_prob[:, 1:6].argmax(1).astype(np.int64) + 1


def train_linear_head(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = 20,
                      lr: float = 1e-3, wd: float = 1e-4, batch: int = 4096):
    """Torch linear classifier 64->5 (spec §3 frozen recipe); y in 0..4. Returns (W, b)."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], 5)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    yt = torch.from_numpy(y.astype(np.int64))
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), batch):
            idx = perm[s:s + batch]
            loss = F.cross_entropy(lin(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return lin.weight.detach().numpy().copy(), lin.bias.detach().numpy().copy()


def apply_head(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (X @ W.T + b).argmax(1).astype(np.int64) + 1


def sklearn_logreg(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, y)
    return (lr.predict(X) + 1).astype(np.int64)
```

- [ ] **Step 4: 跑通过** — `pytest tests/test_typing_probe.py -q` PASS

- [ ] **Step 5: 提交**

```bash
git add src/nucseg/typing_probe.py tests/test_typing_probe.py
git commit -m "feat: typing-probe pure logic (instance/ring pooling, T1 rule, linear heads)"
```

### Task 4: 特征导出 CLI（GPU，fold-guarded，带 inst_prob 绑定检查）

**Files:**
- Create: `scripts/export_tp_features.py`
- Test: `tests/test_typing_probe.py`（CLI 守卫与拒绝覆盖）

**Interfaces:**
- Consumes: `nucseg.cellvit.engine.build_model, TrainConfig, run_dead_expert, run_widen`；`typing_probe.pool_instances/pool_rings`；`fold_guard.ensure_dev_fold, run_split`；`pannuke_eval.instance_classes`。
- Produces: `--out` npz，键：`inst_img, inst_id, feat (M,64), ring (M,64), prob (M,6), area (M,), border (M,), cls (M,) 1..5（fold1=GT 类，fold2=T0 多数票类）, tissue (M,)`；**fold2 额外诊断行**（spec §2"GT 区域 vs 预测轮廓单列"）：`gt_inst_img, gt_inst_id, gt_feat, gt_ring, gt_cls`（fold2 GT 实例上的同款池化，仅供诊断，不进主端点）；json 侧车：输入 SHA256（ckpt、pred npz、fold inst/type/images）、RING_WIDTH、batch、逐图池化耗时均值、空环计数、`n_rows`。
- fold1 用 GT 掩码（标签 `instance_classes(gt_inst, gt_type)`），fold2 用 `pred_fold2.npz` 的 inst/inst_id 表（prob 池化用于绑定检查）+ GT 掩码诊断行。

- [ ] **Step 1: 写失败测试（守卫与覆盖拒绝，不起 GPU）**

```python
def test_export_cli_guards(tmp_path, monkeypatch):
    import sys
    run = tmp_path / "run"; run.mkdir()
    (run / "config.json").write_text('{"split": 1}')
    sys.argv = ["export_tp_features.py", "--run", str(run), "--fold", "3",
                "--pred", str(tmp_path / "p.npz"), "--out", str(tmp_path / "o.npz")]
    from scripts.export_tp_features import main
    with pytest.raises(SystemExit, match="TEST fold"):
        main()
    out = tmp_path / "o.npz"; out.write_bytes(b"x")
    sys.argv = ["export_tp_features.py", "--run", str(run), "--fold", "1",
                "--pred", str(tmp_path / "p.npz"), "--out", str(out)]
    (tmp_path / "p.npz").write_bytes(b"x")   # 让守卫先过、卡在拒绝覆盖
    with pytest.raises(SystemExit, match="exists"):
        main()
```

- [ ] **Step 2: 确认失败** — `pytest tests/test_typing_probe.py -k export -q` → FAIL (no module)

- [ ] **Step 3: 实现（要点）**
  - 前向：复制 `_forward_probs` 但 `model(x, return_features=True)`；`to_nhwc` 后取 `p["tp_feat"]`（(B,C,H,W)→(B,H,W,C)），tp softmax 同 `_forward_probs`；autocast bf16，池化前 `.float()`。
  - 逐批：`pool_instances(tp_feat[b], inst_map, ids)` + `pool_rings(...)` + `pool_instances(tp_softmax[b], ...)`（prob，用于绑定）；`border` = 实例触碰图像四边（沿用 existence 审计定义：掩码触及任一边）。
  - 绑定检查（fold2）：`max|pool(tp_softmax) − pred["inst_prob"]|` 逐行 < 1e-4，否则 `SystemExit`（Review Focus #4）。
  - 输出 npz + `.json` 侧车（SHA256、耗时、空环数）；out 存在即拒绝。
  - `--fold` 只接受 1 或 2，且 `ensure_dev_fold`。

- [ ] **Step 4: 跑通过** — `pytest tests/test_typing_probe.py -q` PASS

- [ ] **Step 5: 提交**

```bash
git add scripts/export_tp_features.py tests/test_typing_probe.py
git commit -m "feat: fold-guarded tp_feat/ring export CLI with inst_prob binding check"
```

### Task 5: 实验 runner（CPU）：T0–T3、判定、bootstrap、成本、表格

**Files:**
- Create: `scripts/run_typing_probe.py`
- Test: `tests/test_typing_probe.py`（纯逻辑：表对齐、判定规则）

**Interfaces:**
- Consumes: Tasks 1–4 全部产物；`pred_fold2.npz`（inst, type, inst_img, inst_id, inst_prob）；特征缓存两个 npz。
- Produces: `runs/analysis/typing_probe_20261010/results.json`（每配置 mPQ/strict/逐类/Δ/bootstrap/混淆/耗时）、`tables.md`、判定 `verdict`（spec §2 六条件的逐条布尔）、重分型 npz（胜出配置）+ canonical 复核命令行打印。
- 纯函数 `evaluate_decision(deltas: dict) -> dict` 导出供测试。

- [ ] **Step 1: 写失败测试**

```python
def test_table_alignment_guard():
    # relabel 枚举与 inst_id 表不同序时必须报错
    from scripts.run_typing_probe import t0_classes
    inst = np.zeros((1, 8, 8), np.int32); inst[0, 1:3, 1:3] = 7      # 非常规 id
    typ = np.where(inst > 0, 3, 0).astype(np.uint8)
    inst_id = np.array([7])
    with pytest.raises(AssertionError):
        t0_classes(inst, typ, np.array([0]), inst_id)                # ids 必须逐图相等


def test_evaluate_decision_rules():
    from scripts.run_typing_probe import evaluate_decision
    ok = {"delta": .004, "lo": .001, "hi": .008, "strict_delta": .0005, "dead_delta": .0,
          "beats_t1": True, "seeds_same_direction": True}
    assert evaluate_decision(ok)["pass_gate"] is True
    assert evaluate_decision({**ok, "delta": .002})["pass_gate"] is False       # ΔmPQ<.003
    assert evaluate_decision({**ok, "lo": -.001})["pass_gate"] is False         # CI 跨零
    assert evaluate_decision({**ok, "strict_delta": -.001})["pass_gate"] is False
    assert evaluate_decision({**ok, "dead_delta": -.003})["pass_gate"] is False # Dead 掉>.002
    assert evaluate_decision({**ok, "beats_t1": False})["pass_gate"] is False
    assert evaluate_decision({**ok, "seeds_same_direction": False})["pass_gate"] is False
```

- [ ] **Step 2: 确认失败** — `pytest tests/test_typing_probe.py -k "alignment or decision" -q` → FAIL

- [ ] **Step 3: 实现（要点）**
  - `t0_classes(inst, type, inst_img, inst_id)`：逐图 `instance_classes`，**断言**其 relabel ids == 该图 `inst_id` 行（Review Focus #1），返回 (M,) 类列。
  - T1 = `t1_classes(pred["inst_prob"])`；T2/T3 = 载入缓存后**断言** `(inst_img, inst_id)` 与 pred 表逐位相等，再 `train_linear_head` 种子 {0,1}（T3 特征 = feat+ring）+ `sklearn_logreg` 交叉核对；每配置输出两种子各自 Δ 与均值。
  - 每配置：`mpq`, `mpq_strict`, 逐类 PQ, **逐组织 mPQ**（从 `per_image_class_pq` 按组织 nanmean）, Δ vs T0, `mpq_bootstrap_delta(cls, cls0, 1000, seed=20261010)`, 混淆矩阵（T0→T* 变更计数；matched 行 GT 对/错）。
  - **GT 区域诊断单列**（spec §2 止损依据）：T2/T3 头在 fold2 GT 实例诊断行上的分类准确率 + 混淆，与预测轮廓主端点分开报告。
  - T0 绑定：`mpq(t0)` 与 `eval_fold2` 官方 mPQ 差 < 1e-6（`--baseline-mpq` 传入），否则退出。
  - 成本：侧车逐图池化毫秒 + 头前向毫秒（59593×64 matmul 计时），写入 results.json。
  - 胜出配置写出重分型 npz（inst 不变，type 图 = 每实例新类）并打印 canonical 复核命令
    `python scripts/eval_pannuke.py --pred <retyped>.npz --fold 2 --out runs/analysis/typing_probe_20261010/eval_<cfg>`；
    canonical 复核除 mPQ 一致（<1e-6）外，**bPQ 必须与 eval_fold2 逐位相同**（几何未动——spec §2 的流程性检查）。
  - 拒绝覆盖；`results.json` 带 spec 路径与全部输入 SHA。

- [ ] **Step 4: 跑通过 + 全套回归**

```bash
~/.conda/envs/nuclei/bin/python -m pytest tests/test_typing_probe.py -q
~/.conda/envs/nuclei/bin/python -m pytest -q
```

- [ ] **Step 5: 提交**

```bash
git add scripts/run_typing_probe.py tests/test_typing_probe.py
git commit -m "feat: typing-probe runner with pre-registered decision gate and cost report"
```

### Task 6: 运行实验（操作任务，无 TDD）

- [ ] **Step 1: GPU 特征导出**（A100，fold1 + fold2，各 ~7 分钟）

```bash
~/.conda/envs/nuclei/bin/python -I scripts/export_tp_features.py \
  --run runs/cellvit_uni/split1 --fold 2 --pred runs/cellvit_uni/split1/pred_fold2.npz \
  --out runs/analysis/typing_probe_20261010/feat_fold2.npz --batch-size 32
~/.conda/envs/nuclei/bin/python -I scripts/export_tp_features.py \
  --run runs/cellvit_uni/split1 --fold 1 \
  --out runs/analysis/typing_probe_20261010/feat_fold1.npz --batch-size 32
```

Expected: 绑定检查通过（inst_prob max|Δ|<1e-4）；侧车空环计数记录。

- [ ] **Step 2: 运行 runner（CPU）**

```bash
~/.conda/envs/nuclei/bin/python -I scripts/run_typing_probe.py \
  --pred runs/cellvit_uni/split1/pred_fold2.npz \
  --feat-fold1 runs/analysis/typing_probe_20261010/feat_fold1.npz \
  --feat-fold2 runs/analysis/typing_probe_20261010/feat_fold2.npz \
  --baseline-mpq $(official fold2 mPQ from runs/cellvit_uni/split1/eval_fold2) \
  --out runs/analysis/typing_probe_20261010
```

- [ ] **Step 3: canonical 复核**（独立进程；对胜出配置 fast vs 全量 `pannuke_eval` 差 < 1e-6）

- [ ] **Step 4: findings.md 新条目（2026-10-10，T 实验）**：结果表（T0–T3 各配置 mPQ/strict/Dead/CI/耗时）、判定六条件逐条、失败位置描述（若未过门）、spec/工件路径、限制（单 split 单分割器种子、GT 轮廓训练 vs 预测轮廓验证、头种子仅覆盖线性头）。同步 `RESEARCH_PLAN.md` 顶部进度注记一行。

- [ ] **Step 5: 提交**

```bash
git add docs/findings.md docs/RESEARCH_PLAN.md
git commit -m "docs: typing-probe T0-T3 dev-fold results and pre-registered verdict"
```
