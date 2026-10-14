# E4 视野裁剪可见性机制探针 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结的 split1 CellViT-UNI 检查点上做受控 FOV 探针（C0 基线 / C1 相位平移 / C-int 插值对照 / C2、C3 裁剪放大），以 GT 身份对应表量化截断对检出（miss）的因果效应，按预注册门 H1/H2/H3 判定是否允许起草可见性监督训练轮。

**Architecture:** 纯逻辑模块 `src/nucseg/visibility_probe.py`（窗口采样确定性、条件构造、裁剪身份分组、miss 表、DiD 与组织内分层聚类 bootstrap——不碰磁盘与 torch）+ 薄 GPU runner `scripts/probe_visibility.py`（复用 `cellvit.engine` 的 `_forward_probs`/`_post` 前向与解码、`build_model`+`final.pth` 加载先例、`pannuke_eval.evaluate` 做 C0 绑定与 crop-frame mPQ/bPQ），逐核记录表 → JSON 工件，门判定从 JSON 直接复算。

**Tech Stack:** Python 3.10, numpy 1.23.5, torch 2.5.1+cu124（pinned）, pytest；无新依赖。

**Spec:** `docs/superpowers/specs/2026-10-13-visibility-probe-e4-design.md`（条件、门数字、对象分组、统计口径以 spec 为准）。

## Global Constraints

- 环境 `~/.conda/envs/nuclei`（numpy 1.23.5 / torch 2.5.1+cu124）；不新增依赖。
- 检查点 allowlist：`runs/cellvit_uni/split1/final.pth`（seed19，x1，无 TTA；SHA256 运行时记录）；
  新推理同样无 TTA；batch 32 / workers 16（predict 先例）；**解码常量不动**（decode_u=1、
  marker_u=1、min_size 10 工作尺度 256 原生）。
- fold guard：只读 fold2（split1 验证折，2523 图）；拒绝 fold3 / split2 / split3 任何输入
  （run 目录 config.json 的 split 必须为 1，--fold 必须为 2）。
- C0 绑定：`runs/cellvit_uni/split1/pred_fold2.npz`（SHA256 记录）恒等复评 ==
  `runs/cellvit_uni/split1/eval_fold2/summary.json` 官方 mPQ（**逐位**）。
- miss 定义：GT 核与全部预测实例的最大 IoU < .1（any 类）。门的数字（+.05/+.03/+.05）
  运行前已冻结于 spec，运行后不放宽；门判定写入 JSON，三值（H1/H2/H3）可从工件复算。
- 无训练、无调参、无选择；GPU 仅推理。种子：窗口抖动 rng(20261013)，bootstrap seed 20261013。
- 代码进 `src/`/`scripts/`，工件进 `runs/analysis/visibility_probe_20261013/`（不跟踪）。
- git：分支 `research/visibility-e4`（先例 research/merged-class-m），逐任务提交。
- 每步 pytest 用 `~/.conda/envs/nuclei/bin/python -m pytest`；GPU 任务在 LM1 空闲 A100 上串行。

## Review Focus

- 窗口抖动必须**逐位确定**（同 seed 同 offset），且 192/160 的合法范围是 {0..64}²/{0..96}²
  （错成 {0..63} 会让边缘窗口系统性缺席）。
  → Task 1 test_window_sampling_deterministic_and_range。
- C2/C3 的坐标回映必须无错位：模型输出 256 → 裁窗原生坐标（nearest resize）；合成圆盘的
  可见面积/触边标记可用手工值核对。
  → Task 1 test_crop_visibility_exact（半切圆盘面积 = floor 公式核对）。
- 分组不变量：原完整核（不贴原图外圈）在窗口内 = 新截断 ∪ 保持完整；完全落窗外剔除；
  只有原完整核进 H2/H3。
  → Task 1 test_grouping_partition。
- miss 的 IoU 阈值是 **< .1 严格式**；配对差率（DiD）分子分母必须同核同条件构造，
  对照组（保持完整）与新截断组共享同一窗口。
  → Task 1 test_miss_threshold / test_did_construction。
- bootstrap 必须按图像整簇、组织内分层；每核记录带 image id 与 tissue，缺失（NaN）核
  不进分母。次端点 HV 误差只作诊断，不进门。
  → Task 1 test_bootstrap_stratified_clusters；Task 2 端到端 fixture 校验分组计数。

---

### Task 1: 纯逻辑模块 `src/nucseg/visibility_probe.py`

**Files:**
- Create: `src/nucseg/visibility_probe.py`
- Test: `tests/test_visibility_probe.py`

**Interfaces:**
- Consumes: `numpy`；测试用 `nucseg.metrics.instance.overlap`。
- Produces:
  - `sample_windows(n: int, size: int, seed: int) -> np.ndarray`：(n, 2) int32 (dy,dx)，
    均匀于 {0..256-size}²，`np.random.default_rng(seed).integers` 逐位确定。
  - `shift_pad(arr: np.ndarray, d: int, axis01=(0, 1)) -> np.ndarray`：C1 的 replicate-pad
    平移（内容向 +y+x 移 d，edge 填充），兼容 (H,W) 与 (H,W,3)。
  - `crop_group(inst: np.ndarray, win: tuple[int, int, int]) -> dict[int, dict]`：
    `win=(y,x,size)`；返回每个原 GT id → {visible: (size,size) 掩码帧内 bool,
    area: int 裁窗原生像素数, touches_crop: bool, group: "truncated"|"full"|"outside"}。
    visible 掩码以裁窗坐标 (size,size) 布尔给出（identity 对应的核心数据结构）。
  - `original_full(inst: np.ndarray) -> np.ndarray`：每 GT id 的"不贴 256 外圈"布尔
    （含 id 0 为 False 的对齐数组，或返回 id->bool dict——以测试可读为准，选 dict）。
  - `miss_table(gt_lab: np.ndarray, pred_inst: np.ndarray, thr: float = 0.1) -> dict[int, bool]`：
    每 GT id → max-IoU(all pred) < thr。
  - `bin_area(a: int) -> str`："<30" / "30-100" / ">=100"（裁窗原生 px）；
    `bin_area_native(a: int) -> str`："<100" / "100-400" / ">=400"（原图原生 px，H1 分层）。
  - `rate_gap_bootstrap(miss_a, miss_b, img, tissue, n_boot=1000, seed=0) -> dict`：
    P(miss|a) − P(miss|b) 的图像聚类、组织内分层百分位 95% CI（a/b 为逐核 0/1 数组，
    长度相等且同核配对；NaN 任一侧剔除于分母）。
  - `did(trunc_miss_ck, trunc_miss_c0, full_miss_ck, full_miss_c0) -> float`：
    DiD = [P̄(miss|新截断,Ck)−P̄(miss|同核,C0)] − [P̄(miss|保持完整,Ck)−P̄(miss|同核,C0)]；
    以及 `did_bootstrap(records_ck, records_c0, img, tissue, n_boot, seed)`（对逐核配对差
    做同一聚类重采样，返回 {did, lo, hi}）。

- [ ] **Step 1: 写失败测试**（合成圆盘世界，覆盖 Review Focus 全部 5 项）

```python
"""E4 visibility probe pure-logic tests: windows, crop grouping, miss, DiD, bootstrap."""
import numpy as np
import pytest

from nucseg.visibility_probe import (bin_area, bin_area_native, crop_group, did,
                                     miss_table, original_full, rate_gap_bootstrap,
                                     sample_windows, shift_pad)


def _disc(a, i, cy, cx, r):
    yy, xx = np.ogrid[: a.shape[0], : a.shape[1]]
    a[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = i


def test_window_sampling_deterministic_and_range():
    w1 = sample_windows(2523, 192, 20261013)
    w2 = sample_windows(2523, 192, 20261013)
    assert np.array_equal(w1, w2)                       # 逐位确定
    assert w1.shape == (2523, 2) and w1.dtype == np.int32
    assert w1.min() >= 0 and w1.max() <= 64             # 192: {0..64}
    w3 = sample_windows(100, 160, 20261013)
    assert w3.min() >= 0 and w3.max() <= 96             # 160: {0..96}


def test_shift_pad_identity_and_edge():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    s = shift_pad(img, 2)
    assert s.shape == img.shape
    assert s[2:, 2:] == pytest.approx(img[:2, :2])       # 内容移 +y+x
    assert np.all(s[0] == img[0]) and np.all(s[:, 0] == s[:, 1])  # edge 复制
    assert np.array_equal(shift_pad(img, 0), img)


def test_crop_visibility_exact():
    # 256 帧内圆盘 id=1 半径 10 圆心 (128,128)；合法窗偏移 ≤64：(64,64,192) 完整包含
    inst = np.zeros((256, 256), np.int32); _disc(inst, 1, 128, 128, 10)
    g = crop_group(inst, (64, 64, 192))
    a_full = int((inst == 1).sum())
    assert g[1]["group"] == "full" and g[1]["area"] == a_full and not g[1]["touches_crop"]
    assert np.array_equal(g[1]["visible"], inst[64:256, 64:256] == 1)
    # 圆心 (60,60)（窗心外侧）r=10：窗 (64,64,192) 的上/左边切过圆盘 → truncated、触两边、
    # 可见 = 圆盘 ∩ 窗内区域，裁窗坐标手工构造核对
    inst3 = np.zeros((256, 256), np.int32); _disc(inst3, 3, 60, 60, 10)
    g3 = crop_group(inst3, (64, 64, 192))
    visible = np.zeros((192, 192), bool)
    visible[(inst3[64:256, 64:256] == 3)] = True
    assert g3[3]["group"] == "truncated" and g3[3]["touches_crop"]
    assert g3[3]["area"] == int(visible.sum()) and g3[3]["area"] > 0
    assert np.array_equal(g3[3]["visible"], visible)
    # 完全落窗外
    inst2 = np.zeros((256, 256), np.int32); _disc(inst2, 2, 10, 10, 3)
    assert crop_group(inst2, (64, 64, 192))[2]["group"] == "outside"


def test_grouping_partition():
    inst = np.zeros((256, 256), np.int32)
    _disc(inst, 1, 128, 128, 8)   # 窗内完整
    _disc(inst, 2, 64, 64, 8)     # 窗边截断
    _disc(inst, 3, 250, 250, 6)   # 贴原图外圈且窗外
    g = crop_group(inst, (100, 100, 192))
    of = original_full(inst)
    inside = [i for i, r in g.items() if r["group"] != "outside"]
    assert set(inside) == {1, 2}
    assert of == {1: True, 2: True, 3: False}
    # 原完整 ∩ 窗内 = truncated ∪ full（互斥完备）
    assert {g[i]["group"] for i in inside if of[i]} == {"truncated", "full"}


def test_miss_threshold():
    gt = np.zeros((32, 32), np.int32); _disc(gt, 1, 16, 16, 6)
    # IoU 恰为 .1 边界两侧的两个预测：外环近似 IoU≈.1 —— 用面积构造精确控制
    pred = np.zeros((32, 32), np.int32); _disc(pred, 5, 16, 16, 19)  # IoU = 36/(361)≈.0997 <.1
    assert miss_table(gt, pred)[1] is True
    _disc(pred, 5, 16, 16, 17)                                    # IoU = 36/289≈.1246 ≥.1
    assert miss_table(gt, pred)[1] is False


def test_area_bins():
    assert bin_area(29) == "<30" and bin_area(30) == "30-100" and bin_area(100) == ">=100"
    assert bin_area_native(99) == "<100" and bin_area_native(100) == "100-400"
    assert bin_area_native(400) == ">=400"


def test_did_construction():
    # 手工 4 组概率：截断 .5→.8，完整 .2→.4 ⇒ DiD = .3−.2 = 0.1
    d = did(np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1]),   # Ck 截断 miss 率 .8
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),   # C0 截断 miss 率 .5
            np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),   # Ck 完整 miss 率 .4
            np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]))   # C0 完整 miss 率 .2
    assert d == pytest.approx((0.8 - 0.5) - (0.4 - 0.2), abs=1e-12)
    assert d == pytest.approx(0.1, abs=1e-12)


def test_rate_gap_and_bootstrap_stratified():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 2, 40)
    tissue = np.where(img == 0, "Breast", "Lung")
    miss_a = rng.random(40) < 0.5
    miss_b = miss_a.copy()                      # 同核同值 → gap 0，CI 收缩
    r = rate_gap_bootstrap(miss_a, miss_b, img, tissue, n_boot=200, seed=1)
    assert r["gap"] == 0.0 and abs(r["lo"]) < 1e-12 and abs(r["hi"]) < 1e-12
    # 全 1 vs 全 0 → gap 1，CI 覆盖 1
    r2 = rate_gap_bootstrap(np.ones(40), np.zeros(40), img, tissue, n_boot=200, seed=1)
    assert r2["gap"] == 1.0 and r2["lo"] <= 1.0 <= r2["hi"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data7/jinxinhao/bdd26 && ~/.conda/envs/nuclei/bin/python -m pytest tests/test_visibility_probe.py -q`
Expected: FAIL（ModuleNotFoundError: nucseg.visibility_probe）

- [ ] **Step 3: 最小实现**

```python
"""E4 visibility-probe pure logic: FOV-crop conditions, GT identity grouping, miss tables, DiD.

Everything here is side-effect free and torch-free: the runner feeds images/GT/predictions,
this module decides windows, groups, and statistics (spec 2026-10-13
docs/superpowers/specs/2026-10-13-visibility-probe-e4-design.md)."""
from __future__ import annotations

import numpy as np

from nucseg.metrics.instance import overlap

PATCH = 256


def sample_windows(n: int, size: int, seed: int) -> np.ndarray:
    hi = PATCH - size  # 192 -> 0..64, 160 -> 0..96
    rng = np.random.default_rng(seed)
    return rng.integers(0, hi + 1, size=(n, 2)).astype(np.int32)
```

其余函数按 Interfaces 直写：`shift_pad` 用 `np.pad(..., mode="edge")` 顶/左裁尾；
`crop_group` 逐 id 切窗 `sub = inst[y:y+size, x:x+size] == i`，`touches_crop = sub[0]|sub[-1]|sub[:,0]|sub[:,-1]`，
`group = "outside" if area==0 else ("truncated" if touches_crop else "full")`；
`original_full` 对每 id 判外圈 `inst[0]|inst[-1]|inst[:,0]|inst[:,-1]`；
`miss_table` 用 `overlap(gt_lab, pred_inst).iou` 行最大值 < thr（无预测实例 → True）；
`rate_gap_bootstrap`/`did_bootstrap` 与 `fast_retype.mpq_bootstrap_delta` 同款分层重采样
（`strata = [nonzero(tissue == t) for t in unique(tissue)]`，图像整簇 `rng.choice`，
NaN 剔除于分母，百分位 [2.5, 97.5]）。

- [ ] **Step 4: 跑测试确认通过**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_visibility_probe.py -q`
Expected: PASS（8 项全绿）

- [ ] **Step 5: 全量回归 + Commit**

```bash
git add src/nucseg/visibility_probe.py tests/test_visibility_probe.py
git commit -m "feat(e4): visibility-probe pure logic (windows, crop grouping, miss, DiD, stratified cluster bootstrap)"
```

---

### Task 2: GPU runner `scripts/probe_visibility.py`（端到端 + 绑定）

**Files:**
- Create: `scripts/probe_visibility.py`
- Test: `tests/test_probe_runner.py`

**Interfaces:**
- Consumes: Task 1 全部函数；`nucseg.cellvit.engine` 的 `TrainConfig, build_model, _forward_probs,
  _post`（预测先例）；`nucseg.cellvit.data` 的 UNI_MEAN/UNI_STD 归一化；`PanNukeFold(2)`；
  `nucseg.metrics.pannuke_eval.evaluate`；`nucseg.metrics.errors.classify_errors`（次端点
  split/merge 率）；`nucseg.hovernet.data.hv_targets`（次端点 ideal，min_size=0）。
- Produces:
  - `build_conditions(images, inst, typ, windows192, windows160)`：逐图产出五条件的输入帧
    与 GT 帧（C0 恒等；C1 shift_pad(12)；C-int 256→192→256 INTER_LINEAR；
    C2/C3 crop→resize 256 INTER_LINEAR + crop_group 身份表）。
  - `predict_condition(model, frames, device, batch=32, workers=16)`：(N,256,256) inst/type，
    复用 predict_fold 的 `_forward_probs`+`Pool(_post)`；C2/C3 输出 nearest resize 回
    裁窗原生 size。
  - `analyze_records(rec) -> dict`：rec = 逐核记录 DataFrame/dict 列表
    （image, tissue, gt_id, cls, area_native, original_full, cond, group, frag_area_bin,
    miss）；产出 H1/H2/H3 门数字 + 次端点汇总 —— 纯函数，测试直接喂合成记录。
  - `main(argv=None)`：guards（--fold 必 2、run config split 必 1、ckpt/pred allowlist、
    --out 拒覆写）、C0 绑定（evaluate(fold2 GT, pred_fold2) mPQ == eval_fold2 summary 逐位）、
    逐条件推理 + 逐核记录、`analyze_records`、写
    `runs/analysis/visibility_probe_20261013/{probe.json, tables.md, commands.md}`
    （provenance：final.pth / pred_fold2.npz / eval_fold2 summary sha256、代码指纹、
    窗口 seed 20261013、bootstrap seed 20261013）。
  - probe.json 门结构：`gates: {H1: {dead_gap, lo, hi, pass}, H2: {did_c2, lo, hi,
    did_c_int, pass}, H3: {gap, lo, hi, pass}}`，pass 布尔按 spec §2 冻结数字复算。

- [ ] **Step 1: 写失败测试**（合成逐核记录走 `analyze_records`；guards 单测）

```python
"""E4 runner tests: gate arithmetic on synthetic records, guards, binding shape."""
import json
import numpy as np
import pytest
from pathlib import Path

from scripts.probe_visibility import analyze_records, main


def _rec(cond, group, miss, cls=4, img=0, tissue="Breast", frag=">=100", full=True):
    return dict(image=img, tissue=tissue, gt_id=hash((cond, group, cls, frag, img, id(miss))),
                cls=cls, area_native=250, original_full=full, cond=cond, group=group,
                frag_area_bin=frag, miss=miss)


def test_h1_h2_h3_gate_arithmetic():
    rec = []
    # H1: Dead 贴边 miss .5 vs 内部 .2（差距 .30 ≥ .05 → pass 方向）
    rec += [_rec("C0", "border", True), _rec("C0", "border", False),
            _rec("C0", "interior", False), _rec("C0", "interior", False),
            _rec("C0", "interior", False), _rec("C0", "interior", True)]
    # H2: 新截断 C2-C0 +0.3 vs 保持完整 +0.05 → DiD 0.25（≥ .03 方向）；
    # C-int DiD 构造为 0
    for m0, m2 in [(0, 1), (1, 0), (0, 1), (0, 1)]:        # 截断 .25 -> .75
        rec += [_rec("C0", "truncated", bool(m0), img=1), _rec("C2", "truncated", bool(m2), img=1)]
    for m0, m2 in [(0, 0), (0, 1), (1, 1), (0, 0)]:        # 完整 .25 -> .5
        rec += [_rec("C0", "stays_full", bool(m0), img=1), _rec("C2", "stays_full", bool(m2), img=1)]
    rec += [_rec("C0", "truncated", False, img=2), _rec("C-int", "truncated", False, img=2),
            _rec("C0", "stays_full", False, img=2), _rec("C-int", "stays_full", False, img=2)]
    # H3: <30 档 miss 1.0 vs >=100 档 0.0（差距 1.0 ≥ .05 方向）
    rec += [_rec("C2", "truncated", True, cls=4, frag="<30", img=3),
            _rec("C2", "truncated", False, cls=4, frag=">=100", img=3)]
    out = analyze_records(rec, n_boot=200, seed=1)
    g = out["gates"]
    # H1: border 1/2 vs interior 1/4 → gap = 0.25
    assert g["H1"]["dead_gap"] == pytest.approx(0.25, abs=1e-12)
    # H2: 截断 C0 .25 -> C2 .75 (+.5)；完整 C0 .25 -> C2 .5 (+.25) ⇒ DiD = 0.25
    assert g["H2"]["did_c2"] == pytest.approx(0.25, abs=1e-12)
    assert g["H2"]["did_c_int"] == 0.0
    # H3: <30 档 1.0 vs >=100 档 0.0
    assert g["H3"]["gap"] == pytest.approx(1.0, abs=1e-12)


def test_gates_require_ci_lower_positive():
    # 全零差距 → CI 覆盖 0 → H1 pass=False（门要 lo>0）
    rec = ([_rec("C0", "border", False), _rec("C0", "interior", False, img=1)])
    out = analyze_records(rec, n_boot=100, seed=1)
    assert out["gates"]["H1"]["dead_gap"] == 0.0 and not out["gates"]["H1"]["pass"]


def test_fold_and_ckpt_guards(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"split": 3}))  # 非 split1
    with pytest.raises(SystemExit):
        main(["--run", str(tmp_path), "--fold", "2", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        main(["--run", "runs/cellvit_uni/split1", "--fold", "3",
              "--out", str(tmp_path / "o")])  # fold3 拒绝


def test_refuses_overwrite(tmp_path):
    out = tmp_path / "already"; out.mkdir(); (out / "probe.json").write_text("{}")
    with pytest.raises(SystemExit):
        main(["--run", "runs/cellvit_uni/split1", "--fold", "2", "--out", str(out)])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_probe_runner.py -q`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现 runner**

- `guards`：`--fold` 必 2；`--run`（默认 `runs/cellvit_uni/split1`）的 config.json split 必 1、
  ckpt 解析后必须 == allowlist 的 final.pth；pred_fold2.npz、eval_fold2/summary.json 同理；
  `--out/probe.json` 存在即 SystemExit。
- `C0 绑定`：`evaluate(f.gt_channels, f.inst, f.type, f.tissue, pred["inst"], pred["type"])
  ["summary"]["official"]["mPQ"] == json.load(eval_fold2)["official"]["mPQ"]`（逐位 `==`；
  同函数同输入，可达）。
- 模型加载（镜像 predict_cellvit.py）：`cfg = TrainConfig(split=1, out_dir=run,
  dead_expert=run_dead_expert(run), widen=run_widen(run))`；`build_model(cfg,
  pretrained=False).cuda()` + `load_state_dict(torch.load(final.pth, map_location="cpu",
  weights_only=False)["model"])`。
- 条件推理：C1/C-int 在 256 帧直接 evaluate miss（`miss_table(f_inst_shifted, pred)` /
  原帧）；C2/C3 用 `crop_group` 身份表 + 裁窗帧预测（输出 nearest 回 192/160）在裁窗帧
  `miss_table`；逐核记录写入 rec（H1 用 C0 的全部核：贴边 = original_full False；
  interior = True；area_native 用原图面积）。C0 miss 来自 pred_fold2。
- 次端点：per-condition `classify_errors(overlap(...))` 的 split/merge 率（crop 帧对可见
  GT）、逐核 visible-mask IoU（可见片段 vs 最佳预测，miss 定义的同源量）、matched 类型
  正确率、crop-frame mPQ/bPQ（C2/C3 对裁窗可见 GT、其余原帧）、逐组织 H1/H3 差距；
  HV 诊断（|pred_hv − hv_targets(visible_gt, min_size=0)| 片段内均值）从 `_forward_probs`
  的 hv 通道（256 帧）bilinear 回裁窗尺寸后计算。
- 输出 probe.json（records 压缩为逐条件数组 + gates + secondary + provenance）、
  tables.md（三条件主表 + 门判定行）、commands.md。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `~/.conda/envs/nuclei/bin/python -m pytest tests/test_probe_runner.py tests/test_visibility_probe.py -q && ~/.conda/envs/nuclei/bin/python -m pytest -q`
Expected: 新测试 PASS；全量无回归

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_visibility.py tests/test_probe_runner.py
git commit -m "feat(e4): visibility-probe runner (guards, C0 binding, per-nucleus records, gates from JSON)"
```

---

### Task 3: GPU 实跑 + 门判定 + 结果文档（执行任务）

- [ ] **Step 1: 运行**（LM1 空闲 A100；2523 图 × 4 条件 ≈ 每条件数分钟）

```bash
cd /data7/jinxinhao/bdd26 && ~/.conda/envs/nuclei/bin/python scripts/probe_visibility.py \
  --run runs/cellvit_uni/split1 --fold 2 --out runs/analysis/visibility_probe_20261013
```

- [ ] **Step 2: 核验**

- C0 绑定逐位通过（脚本内断言，失败即停）；窗口 offsets 从 JSON 重放 `sample_windows`
  逐位一致。
- 门判定三值从 probe.json 独立复算（临时只读脚本），与 tables.md 一致。
- 抽 2 图手工核对裁剪分组（新截断/保持完整/落窗外计数与裁窗几何一致）。

- [ ] **Step 3: 判定记录**

- 结果文档 `docs/E4_VISIBILITY_PROBE_2026-10-13.md`：门数字、次端点、限制（截断与位置
  不可分离、C2 zoom 与 FOV 共变由 DiD 分离、单检查点机制证据、fold2 同时是 T 轮验证折
  ——只读分析无选择，披露）。
- findings.md 新条目（2026-10-13 E4）：门判定二值文字按 spec §2 预注册键：
  三门全过 ⇒ "允许起草可见性监督训练轮 spec（另行立项）"；任一不过 ⇒ "路线 A 按止损关闭，
  记录失败位置"。次端点只作描述。
- RESEARCH_PLAN.md 头部 2026-10-13 进度注补一行 E4 判定。

- [ ] **Step 4: Commit**

```bash
git add docs/E4_VISIBILITY_PROBE_2026-10-13.md docs/findings.md docs/RESEARCH_PLAN.md
git commit -m "docs: E4 visibility-probe verdict (pre-registered H1/H2/H3 gates)"
```

## 2026-10-14 收尾核验（历史步骤不重演）

Task1/2代码已提交，最新修复988c0cd；Task3全量运行已落地新目录
runs/analysis/visibility_probe_20261014，旧目录保留为失效历史。独立输入/代码哈希、
窗口确定性和三门复算通过；结果文档/findings/RESEARCH_PLAN/README已同步新口径。
全量pytest收尾结果见ignored ops/progress_20261014_codex/test_status.json与pytest.log。
新训练规格仅草案，另行审阅；不重演历史RED步骤，也不重跑已有全量GPU工件。

收尾全量回归：367 passed，19074 warnings，434.61秒，退出码0；无新代码修改。
