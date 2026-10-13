# M 合并类分型模糊度量诊断 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结的 3-split 测试预测上，把 GT 与预测共同的类空间做 10 个类对合并重映射（含 M0 恒等），重算官方/strict mPQ 与逐类 PQ，并在 {2,3} 合并空间重算 typing-oracle 三干预——量化 Inf/Conn 标签边界模糊在官方指标上的权重（纯描述、无门）。

**Architecture:** 纯逻辑模块 `src/nucseg/metrics/merged_class.py`（类映射 + 泛化 C 类的 fast 重评分 + 配对 bootstrap，消费 `RetypeEvaluator` 的匹配表，不碰磁盘）+ 薄 CLI `scripts/merged_class_diag.py`（allowlist fold guard、SHA256 provenance、M0/不变量绑定断言、JSON+Markdown 输出）。合并语义的关键：GT 通道与预测实例的 IoU>0.5 配对与类无关，合并 = 对配对两侧与计数同时过 `cmap` 紧凑重映射，C=4 的聚合口径同构（图 nanmean × 组织 nanmean × 组织均值）。

**Tech Stack:** Python 3.10, numpy 1.23.5（pinned）, pytest；无新依赖。

**Spec:** `docs/superpowers/specs/2026-10-13-merged-class-m-design.md`（干预菜单、不变量、措辞边界以 spec 为准）。

## Global Constraints

- numpy 1.23.5 / torch 2.5.1 环境为 `~/.conda/envs/nuclei`；不新增依赖。
- NpzFile 必须先物化成员数组再逐图索引（2026-10-08 E0 教训）。
- 预测输入 allowlist（spec §5，路径外一律拒绝）：
  `runs/cellvit_uni/split1/pred_test_fold3.npz`、`runs/cellvit_uni/split2/pred_test_fold3.npz`、
  `runs/cellvit_uni/split3/pred_test_fold1.npz`；对应冻结 eval：
  `runs/cellvit_uni/split{1,2}/eval_test_fold3/summary.json`、`runs/cellvit_uni/split3/eval_test_fold1/summary.json`。
- 测试产物只读诊断：结论措辞 = "合并类空间下的度量变化"，不得写成"模糊损失"或份额分解；
  不进入方法成绩表（spec §8）。
- 无训练、无调参、无选择；不动任何冻结结果。
- 代码进 `src/`/`scripts/`，工件进 `runs/analysis/merged_class_20261013/`（不跟踪）。
- git：master 上逐任务提交（纯诊断线，无 research 分支必要；与 typing_oracle_20261009 先例一致）。
- 每步 pytest 用 `~/.conda/envs/nuclei/bin/python -m pytest`。

## Review Focus

- 合并必须同时作用于**配对两侧**（GT 类 `m_cls` 与预测实例类 `cls`）与**两侧计数**
  （`n_gt` 列求和、预测实例类重分桶）——只改一侧会把 FP/FN 搞错。
  → Task 1 test_merge_matches_naive_official_eval（对官方 `evaluate` 全量对照）。
- 恒等映射必须与 `fast_retype`/冻结 eval **逐位一致**（bincount 行序不变）——任何重排都是 bug。
  → Task 1 test_identity_map_bitexact + Task 3 运行时绑定断言。
- 官方 NaN 约定：合并类在该图 GT 缺失 → NaN；strict 则"GT 缺但预测有"记 0（`pannuke_eval` 同款）。
  → Task 1 test_nan_and_strict_conventions。
- 类 0（untyped）实例在任何映射下不得计入任何类（`cmap[0] == 0` 恒成立，且 tp 选择天然排除）。
  → Task 1 test_untyped_excluded（构造含 cls=0 实例的世界）。
- bPQ 类无关：从配对表独立重算的 bPQ 必须等于冻结 eval bPQ（验证配对表本身就是类无关匹配），
  且所有合并共用同一值。
  → Task 1 test_bpq_from_pairs_matches + Task 3 运行时不变量 1。

---

### Task 1: 纯逻辑模块 `src/nucseg/metrics/merged_class.py`

**Files:**
- Create: `src/nucseg/metrics/merged_class.py`
- Test: `tests/test_merged_class.py`

**Interfaces:**
- Consumes: `nucseg.metrics.fast_retype.RetypeEvaluator`（只读其表：
  `m_img, m_cls, m_row, m_iou, n_gt (n,5), inst_img, tissue`）；测试用
  `nucseg.metrics.pannuke_eval.evaluate / instance_classes` 与 `nucseg.text.retype.paint_types`。
- Produces:
  - `class_map(pair: tuple[int, int] | None) -> np.ndarray`：(6,) int64 紧凑合并类 id。
    `None` → `[0,1,2,3,4,5]`；`(a,b)` a<b → a,b 同 id，幸存类压缩为 1..4
    （例 `(2,3)` → `[0,1,2,2,3,4]`，`(1,5)` → `[0,1,2,3,4,1]`）。恒有 `cmap[0] == 0`。
  - `per_image_class_pq(ev, cls, cmap) -> np.ndarray`：(n, C) 合并类 PQ，GT 缺类处 NaN（官方口径）。
  - `mpq(ev, cls, cmap) -> tuple[float, np.ndarray]`：组织平均官方 mPQ + 逐类 PQ（image nanmean）。
  - `mpq_strict(ev, cls, cmap) -> tuple[float, np.ndarray]`：strict 口径（GT 缺类且预测有 → 0）。
  - `bootstrap_delta(ev, cls, cmap_a, cmap_b, n_boot=1000, seed=0) -> dict`：
    配对图像 bootstrap of mPQ(a)-mPQ(b)，组织内分层、百分位 95% CI（`mpq_bootstrap_delta` 同款，
    但两侧 cmap 可不同）。
  - `bpq_from_pairs(ev) -> tuple[float, np.ndarray]`：从类无关 IoU>0.5 配对表独立重算的
    （组织平均 bPQ, 逐图 bPQ）；GT 无实例的图 NaN（`pannuke_eval` 同款）。

- [ ] **Step 1: 写失败测试**（覆盖 Review Focus 全部 5 项；真实 fold1 子集 + 合成世界双轨）

```python
"""M merged-class diagnostic tests: identity binding, merge vs naive official eval, invariants."""
import numpy as np
import pytest

from nucseg.constants import NUM_CLASSES
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.fast_retype import RetypeEvaluator
from nucseg.metrics.merged_class import (bpq_from_pairs, bootstrap_delta, class_map,
                                         mpq, mpq_strict, per_image_class_pq)
from nucseg.metrics.pannuke_eval import evaluate, instance_classes
from nucseg.text.retype import paint_types

N = 24  # fold1 前 24 图，覆盖多组织


@pytest.fixture(scope="module")
def world():
    f = PanNukeFold(1)
    if not (f.dir / "inst.npy").exists():
        pytest.skip("PanNuke fold1 not prepared")
    rng = np.random.default_rng(0)
    inst, typ = [], []
    for i in range(N):
        lab = np.asarray(f.inst[i]).copy()
        t = np.asarray(f.type[i]).copy()
        # 随机把 ~1/6 实例整体改类 + ~1/10 抹成 untyped，制造跨类错分与 cls=0
        ids = np.unique(lab); ids = ids[ids > 0]
        for g in ids:
            m = lab == g
            if rng.random() < 1 / 6:
                t[m] = rng.integers(1, NUM_CLASSES + 1)
            elif rng.random() < 0.1:
                t[m] = 0
        inst.append(lab); typ.append(t)
    inst, typ = np.stack(inst), np.stack(typ)
    gt = np.asarray(f.gt_channels[:N])
    tissue = np.asarray(f.tissue[:N])
    rows = [(j, i) for j in range(N) for i in np.unique(inst[j]) if i]
    ii = np.array([r[0] for r in rows]); iid = np.array([r[1] for r in rows])
    ev = RetypeEvaluator(gt, inst, ii, iid, tissue, workers=4)
    cls = np.concatenate([instance_classes(inst[j], typ[j])[1] for j in range(N)])
    return f, ev, inst, typ, ii, iid, cls, gt, tissue


def _merged_gt_channels(gt, cmap):
    """官方路径的合并 GT：(n,H,W,4)，a+b 通道像素质加（PanNuke 实例跨通道不相交）。"""
    c = int(cmap.max())
    out = np.zeros(gt.shape[:3] + (c,), gt.dtype)
    for k in range(NUM_CLASSES):
        out[..., cmap[k] - 1] += gt[..., k]
    return out


def test_class_map():
    assert list(class_map(None)) == [0, 1, 2, 3, 4, 5]
    assert list(class_map((2, 3))) == [0, 1, 2, 2, 3, 4]
    assert list(class_map((1, 5))) == [0, 1, 2, 3, 4, 1]
    for a in range(1, 6):
        for b in range(a + 1, 6):
            m = class_map((a, b))
            assert m[0] == 0 and sorted(set(m[1:])) == [1, 2, 3, 4] and m[a] == m[b]


def test_identity_map_bitexact(world):
    _, ev, _, _, _, _, cls, _, _ = world
    ident = class_map(None)
    m_m, per_m = mpq(ev, cls, ident)
    m_r, per_r = ev.mpq(cls)
    assert m_m == m_r and np.array_equal(per_m, per_r, equal_nan=True)
    s_m, sper_m = mpq_strict(ev, cls, ident)
    s_r, sper_r = ev.mpq_strict(cls)
    assert s_m == s_r and np.array_equal(sper_m, sper_r, equal_nan=True)


def test_merge_matches_naive_official_eval(world):
    f, ev, inst, typ, ii, iid, cls, gt, tissue = world
    for pair in [(2, 3), (1, 5), (4, 5)]:
        cmap = class_map(pair)
        m, per = mpq(ev, cls, cmap)
        s, sper = mpq_strict(ev, cls, cmap)
        full = evaluate(_merged_gt_channels(gt, cmap), np.asarray(f.inst[:N]),
                        np.asarray(f.type[:N]), tissue, inst,
                        paint_types(inst, ii, iid, cmap[cls]), workers=4)["summary"]
        assert m == pytest.approx(full["official"]["mPQ"], abs=1e-10)
        assert s == pytest.approx(full["strict"]["mPQ"], abs=1e-10)
        got = list(full["per_class_PQ"].values())
        assert per == pytest.approx(got, abs=1e-10, nan_ok=True)
        assert sper == pytest.approx(list(full["per_class_PQ_strict"].values()),
                                     abs=1e-10, nan_ok=True)


def test_nan_and_strict_conventions(world):
    _, ev, inst, typ, ii, iid, cls, gt, tissue = world
    cmap = class_map((2, 3))
    pq = per_image_class_pq(ev, cls, cmap)
    # 合并类 2 的 GT 计数 = 原 2+3 通道计数和；GT 缺 → NaN（official），strict 只在"有预测"时记 0
    n_gt_m = ev.n_gt[:, 1] + ev.n_gt[:, 2]
    assert np.isnan(pq[n_gt_m == 0, 1]).all()
    strict = mpq_strict(ev, cls, cmap)
    assert strict[0] <= mpq(ev, cls, cmap)[0]


def test_untyped_excluded(world):
    _, ev, inst, typ, ii, iid, cls, gt, tissue = world
    cmap = class_map((2, 3))
    pq0 = per_image_class_pq(ev, cls, cmap)
    cls0 = cls.copy(); cls0[:] = 0  # 全部 untyped：任何合并类无 TP 且无预测计入
    pq1 = per_image_class_pq(ev, cls0, cmap)
    assert np.isnan(pq1).all() or (pq1[~np.isnan(pq1)] == 0).all()
    # 与恒等空间一致性：cls=0 在两种 cmap 下逐位同值
    assert np.array_equal(pq1, per_image_class_pq(ev, cls0, class_map(None)),
                          equal_nan=True)


def test_bpq_from_pairs_matches(world):
    f, ev, inst, typ, ii, iid, cls, gt, tissue = world
    b_m, per_img = bpq_from_pairs(ev)
    full = evaluate(gt, np.asarray(f.inst[:N]), np.asarray(f.type[:N]), tissue,
                    inst, typ, workers=4)["summary"]
    assert b_m == pytest.approx(full["official"]["bPQ"], abs=1e-12)
    assert per_img.shape == (N,)


def test_bootstrap_delta_same_map_is_zero(world):
    _, ev, _, _, _, _, cls, _, _ = world
    cmap = class_map((2, 3))
    d = bootstrap_delta(ev, cls, cmap, cmap, n_boot=50, seed=1)
    assert d["delta"] == 0.0 and abs(d["lo"]) < 1e-12 and abs(d["hi"]) < 1e-12


def test_bootstrap_delta_point_estimate_consistent(world):
    _, ev, _, _, _, _, cls, _, _ = world
    d = bootstrap_delta(ev, cls, class_map(None), class_map((2, 3)), n_boot=50, seed=2)
    m0 = mpq(ev, cls, class_map(None))[0]
    m23 = mpq(ev, cls, class_map((2, 3)))[0]
    assert d["delta"] == pytest.approx(m23 - m0, abs=1e-12)
    assert d["lo"] <= d["delta"] <= d["hi"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data7/jinxinhao/bdd26 && ~/.conda/envs/nuclei/bin/python -m pytest tests/test_merged_class.py -q`
Expected: FAIL（ModuleNotFoundError: nucseg.metrics.merged_class）

- [ ] **Step 3: 最小实现**

```python
"""Merged-class (4-of-5) official/strict mPQ over RetypeEvaluator's class-agnostic match tables.

Merging classes a,b remaps BOTH sides of every IoU>0.5 pair (GT class via ev.m_cls, predicted
class via the instance table) and BOTH count tables (per-image GT counts ev.n_gt, predicted
instance counts); PQ matching itself is class-agnostic so the pairs never change. Aggregation
is the official one with C=4: per-image nanmean over merged classes present in GT -> tissue
nanmean -> mean over tissues. bPQ is class-independent and derived once from the pair table
(invariant 1 of spec 2026-10-13: identical under every merge)."""
from __future__ import annotations

import numpy as np

from ..constants import NUM_CLASSES, TISSUES
from .fast_retype import RetypeEvaluator


def class_map(pair: tuple[int, int] | None) -> np.ndarray:
    cmap = np.arange(NUM_CLASSES + 1, np.int64)
    if pair is not None:
        a, b = sorted(pair)
        cmap[b] = a
    uniq, new = np.unique(cmap[1:], return_inverse=True)  # compact survivors to 1..C
    out = np.zeros_like(cmap)
    out[1:] = new + 1
    return out
```

`per_image_class_pq(ev, cls, cmap)`（C = `cmap.max()`；注意保持 `fast_retype` 的行序——
直接用 `ev.m_*` 原数组做 bincount，不重排）：

```python
def per_image_class_pq(ev: RetypeEvaluator, cls: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    n, C = ev.n, int(cmap.max())
    g = cmap[ev.m_cls]                      # pair 的 GT 合并类（>=1）
    tp_sel = cmap[cls[ev.m_row]] == g       # cls==0 -> cmap[0]==0 != g，天然排除
    key = ev.m_img[tp_sel] * C + g[tp_sel] - 1
    tp = np.bincount(key, minlength=n * C).reshape(n, C).astype(np.float64)
    siou = np.bincount(key, ev.m_iou[tp_sel], minlength=n * C).reshape(n, C)
    n_gt_m = np.zeros((n, C), np.int64)     # 两侧计数同步重映射
    for c in range(1, NUM_CLASSES + 1):
        n_gt_m[:, cmap[c] - 1] += ev.n_gt[:, c - 1]
    v = cls > 0
    npred = np.bincount(ev.inst_img[v] * C + cmap[cls[v]] - 1, minlength=n * C).reshape(n, C)
    fp, fn = npred - tp, n_gt_m - tp
    with np.errstate(all="ignore"):
        d = tp + 0.5 * fp + 0.5 * fn
        pq = np.where(d > 0, (tp / d) * (siou / (tp + 1.0e-6)), np.nan)
    return np.where(n_gt_m > 0, pq, np.nan)
```

`mpq` / `mpq_strict` / `bootstrap_delta` / `bpq_from_pairs`：按 `fast_retype.mpq / mpq_strict /
mpq_bootstrap_delta` 同款聚合与分层重采样逐行移植，仅类轴由 NUM_CLASSES 换成 C；
strict 的"预测有"用合并后 `npred > 0`；`bpq_from_pairs` 按图聚合配对：
`tp = bincount(m_img)`、`siou = bincount(m_img, m_iou)`、`fn = n_gt.sum(1) - tp`、
`fp = bincount(inst_img) - tp`、GT 无实例（`n_gt.sum(1) == 0`）→ NaN。

- [ ] **Step 4: 跑测试确认通过**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_merged_class.py -q`
Expected: PASS（8 项全绿）

- [ ] **Step 5: 全量回归 + Commit**

Run: `~/.conda/envs/nuclei/bin/python -m pytest -q`（此前基线 335 passed）

```bash
git add src/nucseg/metrics/merged_class.py tests/test_merged_class.py
git commit -m "feat(m): compact class-map rescoring of official/strict mPQ over retype match tables"
```

---

### Task 2: CLI `scripts/merged_class_diag.py`（端到端 + 绑定断言）

**Files:**
- Create: `scripts/merged_class_diag.py`
- Test: `tests/test_merged_class_diag.py`

**Interfaces:**
- Consumes: Task 1 全部函数；`typing_oracle.instance_table`（重建 (img,id) 表，`sys.path`
  先例同 `e1t_typing_consensus.py` 的 `from fold_guard import ...` 写法）；
  `lever_postproc.file_provenance`；`PanNukeFold`、`instance_classes`。
- Produces: `analyze_split(pred: dict, fold, split: int, frozen: dict) -> dict`
  （pred = 已物化的 npz dict；fold = 有 `.gt_channels`/`.tissue` 的对象；frozen = 该 split
  冻结 eval summary dict，用于绑定断言）与 `main()`；输出目录
  `runs/analysis/merged_class_20261013/`：
  - `results.json`：`provenance`（3 pred npz + 3 冻结 eval summary + gt_channels.npz 的
    sha256、script/merged_class.py/fast_retype.py 指纹、seed 20261013）、
    `splits: {1|2|3: {m0: {mpq, strict, per_class[5], bpq_recomputed, bpq_frozen},
    merges: {"a,b": {mpq, strict, per_class[4], delta_official{delta,lo,hi},
    delta_strict{...}}}, oracle_under_merge: {current, oracle, inf_conn}(各含 mpq+delta),
    invariants: {bpq_matches_frozen, dead_pq_unchanged}}}`、`mean`（3-split 头条数字）、
    `pairs`（10 对清单）。
  - `tables.md`：M0 + 10 合并 × {official, strict, Δ(+CI)} 主表 + 逐类 PQ 表 + oracle 表。
  - `commands.md`：运行命令、主机、env、时长。
- 绑定断言（运行时，违反即 SystemExit，不出任何数；实现在 `analyze_split` 内，供测试直接驱动）：
  1. M0 mpq == 冻结 summary.json `official.mPQ`（**逐位 `==`**，typing_oracle 先例已验证可达）；
     M0 逐类 PQ 与冻结 `per_class_PQ` **`abs <= 1e-12`**（冻结值出自 pannuke_eval 的矩阵求和
     顺序，快速路径为 bincount 逐行累加，逐类存在 1-2 ulp 固有差 —— 2026-10-13 已实测
     typing_oracle per-class 对冻结 eval 亦然；mPQ 恰好逐位一致。此容差在 commands.md 披露）；
  2. `bpq_from_pairs(ev)` 组织均值 vs 冻结 bPQ（`abs=1e-12`，同一浮点顺序原因）；
  3. merge(2,3) 的合并类 3（原 Dead）逐类 PQ == 本模块恒等空间的 per_class[3]（**逐位**，
     同路径同 bincount 行序，可达）；
  4. allowlist 外的预测路径 / `--out` 已存在 → SystemExit。

- [ ] **Step 1: 写失败测试**（合成迷你世界走 `analyze_split`，验证计数结构与绑定拒绝路径）

```python
"""Runner tests: mini-world end-to-end, allowlist guard, overwrite guard."""
import numpy as np
import pytest
from dataclasses import dataclass
from pathlib import Path

from scripts.merged_class_diag import ALLOWED_PREDS, analyze_split, main


def _rect(a, i, y, x, h, w):
    a[y:y + h, x:x + w] = i


@dataclass
class _Fold:
    gt_channels: np.ndarray
    tissue: np.ndarray


def _mini_world():
    # 2 图 × 5 类 GT 通道；pred 含跨类错分（Inf<->Conn）、漏检 Dead、拆半 Conn、1 个 untyped(cls=0)
    H = W = 32
    gt = np.zeros((2, H, W, 5), np.int32)
    _rect(gt[0, :, :, 0], 1, 2, 2, 6, 6)      # img0 Neo
    _rect(gt[0, :, :, 1], 1, 12, 4, 5, 5)     # img0 Inf
    _rect(gt[0, :, :, 2], 1, 4, 20, 6, 6)     # img0 Conn
    _rect(gt[0, :, :, 3], 1, 20, 20, 4, 4)    # img0 Dead（pred 漏检）
    _rect(gt[0, :, :, 4], 1, 22, 2, 5, 5)     # img0 Epi
    _rect(gt[1, :, :, 1], 1, 3, 3, 7, 7)      # img1 Inf
    _rect(gt[1, :, :, 2], 1, 14, 14, 8, 8)    # img1 Conn（pred 拆两半，各 IoU=.5 不匹配）
    inst = np.zeros((2, H, W), np.int32); typ = np.zeros((2, H, W), np.uint8)
    _rect(inst[0], 1, 2, 2, 6, 6); typ[0][inst[0] == 1] = 1   # Neo 对
    _rect(inst[0], 2, 12, 4, 5, 5); typ[0][inst[0] == 2] = 2  # Inf 对
    _rect(inst[0], 3, 4, 20, 6, 6); typ[0][inst[0] == 3] = 2  # Conn 说成 Inf（merge(2,3) 修复）
    _rect(inst[0], 4, 22, 2, 5, 5); typ[0][inst[0] == 4] = 5  # Epi 对
    _rect(inst[1], 1, 3, 3, 7, 7); typ[1][inst[1] == 1] = 3   # Inf 说成 Conn（merge(2,3) 修复）
    _rect(inst[1], 2, 14, 14, 4, 8); _rect(inst[1], 3, 18, 14, 4, 8)
    typ[1][np.isin(inst[1], (2, 3))] = 2                     # 拆半的 Conn
    _rect(inst[1], 4, 24, 2, 4, 4)                           # spurious，全 0 type -> cls=0
    pred = {"inst": inst, "type": typ}
    fold = _Fold(gt, np.array(["Breast", "Lung"]))
    # frozen=None 时 analyze_split 跳过冻结比对，只做内部不变量（Dead 列逐位、bPQ 同值）。
    return pred, fold


def test_analyze_split_mini():
    pred, fold = _mini_world()
    out = analyze_split(pred, fold, split=1, frozen=None)
    assert set(out) >= {"m0", "merges", "oracle_under_merge", "invariants"}
    assert out["invariants"]["dead_pq_unchanged"]
    assert len(out["merges"]) == 10 and "(2,3)" in out["merges"]
    # 装配层不破坏恒等性：M0 逐类 == RetypeEvaluator 直算（同 Task 1 口径）
    from nucseg.metrics.fast_retype import RetypeEvaluator
    from nucseg.metrics.pannuke_eval import instance_classes
    ii, iid = [], []
    for j in range(2):
        ids = np.unique(pred["inst"][j]); ids = ids[ids > 0]
        ii.append(np.full(len(ids), j)); iid.append(ids)
    ev = RetypeEvaluator(fold.gt_channels, pred["inst"], np.concatenate(ii),
                         np.concatenate(iid), fold.tissue, workers=1)
    cls = np.concatenate([instance_classes(pred["inst"][j], pred["type"][j])[1] for j in range(2)])
    assert out["m0"]["mpq"] == ev.mpq(cls)[0]


def test_bindings_reject_mismatch():
    pred, fold = _mini_world()
    out = analyze_split(pred, fold, split=1, frozen=None)
    names = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    wrong = {"official": {"mPQ": out["m0"]["mpq"] + 0.01, "bPQ": 0.5},
             "per_class_PQ": {k: v + 0.01 for k, v in zip(names, out["m0"]["per_class"])}}
    with pytest.raises(SystemExit):
        analyze_split(pred, fold, split=1, frozen=wrong)
    right = {"official": {"mPQ": out["m0"]["mpq"], "bPQ": out["m0"]["bpq_recomputed"]},
             "per_class_PQ": dict(zip(names, out["m0"]["per_class"]))}
    analyze_split(pred, fold, split=1, frozen=right)  # 不抛


def test_allowlist_guard(tmp_path):
    bogus = tmp_path / "pred_evil.npz"
    bogus.write_bytes(b"")
    with pytest.raises(SystemExit):
        main(["--pred", f"1={bogus}", "--out", str(tmp_path / "o")])


def test_refuses_overwrite(tmp_path):
    out = tmp_path / "already"; out.mkdir(); (out / "results.json").write_text("{}")
    with pytest.raises(SystemExit):
        main(["--out", str(out)])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_merged_class_diag.py -q`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 CLI**

结构（镜像 `scripts/typing_oracle.py` + e1t 的 provenance 风格）：
- 常量：`ALLOWED_PREDS`（3 个绝对路径，`Path(...).resolve()` 后的精确匹配表）、`FROZEN_EVAL`、
  `TEST_FOLD = {1:3, 2:3, 3:1}`、`PAIRS = [(a,b) for a in range(1,6) for b in range(a+1,6)]`、
  `BOOT_SEED = 20261013`。
- `load_split(split) -> (pred_dict, fold_obj, frozen_dict)`：预测路径 `resolve()` 后必须等于
  allowlist 项（否则 SystemExit），`dict(np.load(...))` 一次性物化；frozen = json.load(对应
  eval summary.json)。
- `analyze_split(pred, fold, split, frozen) -> dict`：建 `RetypeEvaluator`（workers 16；
  instance_table 重建 (img,id) 表，同 typing_oracle）→ `cls`（instance_classes 多数票）→
  M0（mpq / strict / per_class(5) / `bpq_recomputed`）+ 绑定断言 1/2（frozen 非 None 时）→
  10 合并（official/strict/逐类(4) + `delta_official`/`delta_strict` 两个
  `bootstrap_delta(..., n_boot=1000, seed=BOOT_SEED + split)`）→ 断言 3 → oracle 三干预
  （current/oracle/inf_conn 构造同 typing_oracle，评分过 `class_map((2,3))`，含对 current-merged
  的 delta + bootstrap CI）。
- `main(argv=None)`：逐 split `load_split` + `analyze_split`、3-split mean、写 results.json /
  tables.md / commands.md；`--out` 存在即 SystemExit；`--pred S=P` 仅接受 allowlist 路径；
  打印主表到 stdout。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_merged_class_diag.py tests/test_merged_class.py -q && ~/.conda/envs/nuclei/bin/python -m pytest -q`
Expected: 新测试 PASS；全量无回归

- [ ] **Step 5: Commit**

```bash
git add scripts/merged_class_diag.py tests/test_merged_class_diag.py
git commit -m "feat(m): merged-class diagnostic runner with allowlist guard, frozen-eval binding and provenance"
```

---

### Task 3: 真实数据执行 + findings 条目（执行任务，无新代码）

- [ ] **Step 1: 运行**

```bash
cd /data7/jinxinhao/bdd26 && ~/.conda/envs/nuclei/bin/python scripts/merged_class_diag.py \
  --out runs/analysis/merged_class_20261013
```
（CPU；每 split 建 RetypeEvaluator ~分钟级 + 10 合并重评分与 20 个 bootstrap，总计 <1 h。）

- [ ] **Step 2: 核验**

- 绑定断言全过（M0 逐位 == 冻结 eval；bPQ 一致；Dead 列不变）——由脚本自身断言，失败即停。
- 从 results.json 手工复算：3-split mean == 各 split 均值；merge(2,3) 的
  `delta_official.delta` == `merges["(2,3)"].mpq - m0.mpq`（逐 split）。
- 假设 M 三个描述性观察（(a) {2,3} 增益在 10 对分布中的排名；(b) 合并类 PQ vs 原两类 PQ 均值；
  (c) oracle-under-merge 差距 vs 5 类空间）各写明数字，**不做门判定、不做份额分解措辞**。

- [ ] **Step 3: findings.md 新条目（2026-10-13 M）+ 提交**

- 数字：M0 复核、10 对 Δofficial(+CI) 表（或指向 tables.md）、{2,3} 排名、oracle-under-merge。
- 措辞边界（spec §8）：只支持"度量侧与 E1-T 共鸣证据一致/不一致"；合并增益 ≠ 模糊损失；
  不改官方 5 类成绩表。

```bash
git add docs/findings.md
git commit -m "docs: M merged-class diagnostic verdict (descriptive, no gates)"
```
