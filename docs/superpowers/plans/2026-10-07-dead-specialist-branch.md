# Dead-Specialist Branch (DSB) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the DSB dead-detection expert branch, its gates A–D machinery, and the dev-round evaluation pipeline from the approved spec, CPU-first, with all GPU stages queued behind E2a completion.

**Architecture:** A fourth decoder branch (`dead_branch`) on `CellViTUNI` outputs Dead-only NP+HV channels; training uses positive-image loss masking; inference decodes expert candidates and merges them into the base instance set through an EF-style area-floor gate. Gates A–D are standalone scripts producing verdict JSONs; no gate passes, no training.

**Tech Stack:** Python 3.10, torch 2.5.1+cu124, numpy 1.23.5, existing pinned scipy/skimage/albumentations/pytest; `nucseg.metrics.pannuke_eval` for all evaluation.

**Spec:** `docs/superpowers/specs/2026-10-07-dead-specialist-branch-design.md` (§5 scarcity = loss masking; §6 gates; §7 rounds/controls; §8 sonnet blind review).

## Global Constraints

- **No GPU work of any kind until the E2a queues drain and the user confirms** (user, 2026-10-07: "等到hv分支在几小时内完成后再开始gpu训练"). Tasks 1–9 and 12 are CPU-only; Tasks 10–11, 13–14 have GPU stages that wait.
- Dev round sees **val fold2 only**; fold3 is read once, after the selection freeze (spec §7). Last checkpoint everywhere. Evaluate only via `nucseg.metrics.pannuke_eval`.
- Run `python -m pytest -q` after touching HoVer-Net plumbing or anything decode/metrics-adjacent (Tasks 4, 8, 10); every task runs its own new tests first (TDD: red → green).
- Pinned env `~/.conda/envs/nuclei`; no dependency changes. Portable code in `src/`/`scripts/`; machine launchers and queue records only in ignored `ops/`. Commit portable files only.
- Scientific config changes refuse resume (existing `prepare_train_config` mechanism covers new fields automatically — verify, don't reimplement).
- Do not modify, move or re-evaluate anything under `runs/hv_threshold_controls_20261006/` (read-only C0 reuse, HV-controls plan owns it).

## Cross-Plan Audit (user-requested, 2026-10-07)

1. **HV-controls plan owns E2a artifacts.** Its pending checklist (audited val evals → `hv_control_numbers.py` analysis → docs → master merge → push) runs on its own branch. DSB **reads** `runs/hv_threshold_controls_20261006/<arm>_seed<seed>/eval_val_fold2*` and the contrasts JSON; it never reruns or overwrites them.
2. **C0 reuse is arm-conditional** (spec gate A): base = `x1_hv30` by default; if gate A fires the 50% rule, base = `x1_hv8` (both arms exist in E2a, seeds 19/1 — no retraining either way).
3. **Merge order:** `research/hv-target-threshold-controls` merges to master first (authorized there), then this branch. This branch was cut from its HEAD; conflicts are limited to `docs/RESEARCH_PLAN.md` progress notes — keep DSB edits there additive.
4. **Gate A input = `hv_control_numbers.py` output JSON** (`--out` of that script; schema `{"arms": {...}, "contrasts": {...}}` printed to stdout). Task 12 parses it read-only; if the real schema differs from the documented keys, adapt the parser and its test fixture in the same commit.
5. **Test-fold discipline:** dev round selection happens on fold2 artifacts only; the single frozen test read is its own task with a user checkpoint before it.
6. **GPU exclusivity:** probe/dump/training launches all go through the LM2 queue runbook (sequential, no `pkill -f` + launch in one SSH command); storage: `df` before writes, bulky artifacts on LM2.

## Review Focus

1. **Historical checkpoints must still strict-load** when the model class gains new optional modules — Task 1 pins state-dict key equality with `dead_expert=False`.
2. **`dead_pos` computed after augmentation** — a crop that removes all Dead pixels must yield `dead_pos=False`, else the masked loss trains on stale positives — Task 2.
3. **All-negative batch must not detach the loss graph** (backward on a zero-dead batch crashes with "does not require grad") — Task 3.
4. **`hv_dead` under TTA dihedral** must get the same sign/swap remap as `hv` — Task 4.
5. **Dead prediction artifacts must never overwrite base artifacts** (separate `_dead` tag, Task 5) and the merge selection must be replayable bit-exactly from saved JSON (Task 6).

---

### Task 1: Model — dead expert branch and equal-param widening

**Files:**
- Modify: `src/nucseg/cellvit/model.py`
- Test: `tests/test_dead_branch.py`

**Interfaces:**
- Produces: `CellViTUNI(..., dead_expert: bool = False, widen: int = 0)`; forward adds keys `"np_dead"`/`"hv_dead"` iff `dead_expert`; `branch_param_count(model) -> dict[str, int]`; `solve_c1_widen() -> int`.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from nucseg.cellvit.model import CellViTUNI, branch_param_count, solve_c1_widen


def _tiny(**kw):
    m = CellViTUNI(uni_ckpt=None, dead_expert=kw.pop("dead_expert", False), **kw)
    # shrink encoder blocks so the test is fast; decoder untouched (what we test)
    m.encoder.blocks = torch.nn.ModuleList(m.encoder.blocks[:2])
    return m


def test_default_off_state_dict_identical():
    base, ext = _tiny(), _tiny()
    assert set(base.state_dict()) == set(ext.state_dict())
    assert not any("dead" in k for k in base.state_dict())


def test_dead_expert_forward_shapes():
    with torch.no_grad():
        out = _tiny(dead_expert=True).eval()(torch.randn(2, 3, 256, 256))
    assert out["np_dead"].shape == (2, 2, 256, 256)
    assert out["hv_dead"].shape == (2, 2, 256, 256)
    with torch.no_grad():
        plain = _tiny().eval()(torch.randn(1, 3, 256, 256))
    assert "np_dead" not in plain and "hv_dead" not in plain


def test_c1_widen_param_parity():
    dsb = CellViTUNI(uni_ckpt=None, dead_expert=True)
    c1 = CellViTUNI(uni_ckpt=None, widen=solve_c1_widen())
    p_dsb = sum(branch_param_count(dsb).values())
    p_c1 = sum(branch_param_count(c1).values())
    assert abs(p_dsb - p_c1) / p_dsb < 0.015
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_dead_branch.py -q`
Expected: FAIL (`dead_expert` unexpected kwarg).

- [ ] **Step 3: Implement**

In `CellViTUNI.__init__` signature add `dead_expert: bool = False, widen: int = 0`. After the existing branches:

```python
if dead_expert:
    self.dead_branch = self._branch(d)
    self.dead_np_head = nn.Conv2d(64, 2, 1)
    self.dead_hv_head = nn.Conv2d(64, 2, 1)
```

`_branch(self, d, b=None)` gets `b = 512 if b is None else b` (existing comment says b is the bottleneck dim); `self.np_branch, self.hv_branch = (self._branch(d, 512 + widen) for _ in range(2))` — widening only NP/HV per spec §7 (C1 folds expert params back there). In `forward`, after the existing `out` dict:

```python
if hasattr(self, "dead_branch"):
    f_dead = self._decode(skips, self.dead_branch)
    out["np_dead"] = self.dead_np_head(f_dead)
    out["hv_dead"] = self.dead_hv_head(f_dead)
```

Module-level helpers:

```python
def branch_param_count(model: CellViTUNI) -> dict[str, int]:
    """Trainable-capacity map per decoder-side group (encoder excluded)."""
    groups = {}
    for name in ("skip0", "skip1", "skip2", "skip3", "np_branch", "hv_branch",
                 "tp_branch", "dead_branch"):
        mod = getattr(model, name, None)
        if mod is not None:
            groups[name] = sum(p.numel() for p in mod.parameters())
    return groups


def solve_c1_widen() -> int:
    """Bottleneck widening whose NP+HV growth best matches the dead-branch parameter count."""
    dsb = sum(branch_param_count(CellViTUNI(uni_ckpt=None, dead_expert=True)).values())
    best, best_gap = 0, float("inf")
    for w in range(0, 257, 32):
        c1 = sum(branch_param_count(CellViTUNI(uni_ckpt=None, widen=w)).values())
        gap = abs(dsb - c1)
        if gap < best_gap:
            best, best_gap = w, gap
    return best
```

*(construction of full ViT-L twice in `solve_c1_widen` is ~2 s CPU; acceptable, and keeps the solver honest against the real `_branch`.)*

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_dead_branch.py -q` → PASS. Then full CPU suite `python -m pytest -q` (model touched; confirm no regression).
- [ ] **Step 5: Commit** — `git add src/nucseg/cellvit/model.py tests/test_dead_branch.py && git commit -m "feat: optional dead-expert decoder branch and C1 widening"`

### Task 2: Data — dead targets after augmentation

**Files:**
- Modify: `src/nucseg/cellvit/data.py`
- Test: `tests/test_dead_targets.py`

**Interfaces:**
- Produces: `PanNukeCellViT(..., dead_targets: bool = False)`; when True `__getitem__` adds `"dead_np_map"` (int64 (H,W) 0/1), `"dead_hv_map"` (float32 (H,W,2)), `"dead_pos"` (bool scalar). Class id lives in one place: add `DEAD_TYPE = 4` to `src/nucseg/constants.py`; `data.py`, `dead_merge.py` (Task 6) and the engine's existing `DEAD = 4` all import it (engine keeps its local alias for history).

- [ ] **Step 1: Write the failing tests** (fixture pattern from `tests/test_hv_training.py`: memory-mapped tiny fold; reuse its helper if importable, else replicate a 4-image stub with `inst/type` arrays)

```python
import numpy as np, torch
def test_dead_targets_present_and_correct(tiny_fold_ds):  # ds built with dead_targets=True, train=False
    s = tiny_fold_ds[0]
    dead = s["tp_map"] == 4
    assert set(np.unique(s["dead_np_map"].numpy())) <= {0, 1}
    assert torch.equal(s["dead_np_map"], dead.to(torch.int64))
    assert s["dead_hv_map"].shape == (*s["np_map"].shape, 2)
    assert bool(s["dead_pos"]) == bool(dead.any())
    # HV is exactly zero wherever the image carries no Dead pixels
    if not dead.any():
        assert not s["dead_hv_map"].any()


def test_dead_pos_reflects_augmented_crop(ds_train_dead):  # train=True, seed forcing a Dead-free crop
    seen = {bool(ds_train_dead[i]["dead_pos"]) for i in range(len(ds_train_dead))}
    assert seen <= {True, False}  # smoke; the crop case is exercised by the seeded fixture
```

Also assert a dataset built with `dead_targets=False` returns **exactly the historical key set** (backward-compat pin).

- [ ] **Step 2: RED** — `python -m pytest tests/test_dead_targets.py -q` → FAIL (unknown kwarg).
- [ ] **Step 3: Implement** — in `__init__` store `self.dead_targets = dead_targets`; in `__getitem__`, after the `upscale` block and before building the return dict:

```python
extra = {}
if self.dead_targets:
    dead_inst = np.where(typ == DEAD_TYPE, inst, 0).astype(np.int32)
    extra = {
        "dead_np_map": torch.from_numpy((dead_inst > 0).astype(np.int64)),
        "dead_hv_map": torch.from_numpy(hv_targets(dead_inst, min_size=self.hv_min_size)),
        "dead_pos": torch.tensor(bool((dead_inst > 0).any())),
    }
return {..., "index": i, **extra}
```

(computed **after** augmentation/upsample, so a crop that drops all Dead pixels correctly yields `dead_pos=False`; dead HV uses the same working-pixel cutoff as the main branch, spec §5).

- [ ] **Step 4: GREEN** + full suite.
- [ ] **Step 5: Commit** — `git add src/nucseg/cellvit/data.py tests/test_dead_targets.py && git commit -m "feat: dead-instance NP/HV targets with post-augmentation presence flag"`

### Task 3: Engine — positive-image masked dead loss

**Files:**
- Modify: `src/nucseg/cellvit/engine.py` (`cellvit_loss`), `TrainConfig`
- Test: extend `tests/test_cellvit_loss.py`

**Interfaces:**
- `TrainConfig` gains `dead_expert: bool = False`, `dead_neg_w: float = 0.0` (menu {0.0 = positive-only, 0.1}, spec §5), `widen: int = 0`.
- `cellvit_loss(pred, batch, cfg)` adds terms `dead_np_ft`, `dead_np_dice`, `dead_hv_mse`, `dead_hv_msge`, `dead_neg_bce` iff `cfg.dead_expert and "np_dead" in pred`.

- [ ] **Step 1: Failing tests**

```python
H = W = 8


def _pred():
    r = lambda *s: torch.randn(1, *s, requires_grad=True)
    return {"np": r(H, W, 2), "hv": r(H, W, 2), "tp": r(H, W, 6),
            "tissue": torch.randn(1, 19, requires_grad=True),
            "np_dead": r(H, W, 2), "hv_dead": r(H, W, 2)}


def _batch(positive=True):
    tp = torch.zeros(1, H, W, dtype=torch.long)
    if positive:
        tp[0, 2:5, 2:5] = 4  # Dead nucleus
    return {"np_map": (tp > 0).long(), "tp_map": tp, "hv_map": torch.zeros(1, H, W, 2),
            "dead_np_map": (tp == 4).long(), "dead_hv_map": torch.zeros(1, H, W, 2),
            "dead_pos": torch.tensor(positive), "tissue": torch.zeros(1, dtype=torch.long)}


def test_dead_terms_zero_but_attached_on_negative_batch():
    cfg = TrainConfig(split=1, out_dir="x", dead_expert=True)
    loss, terms = cellvit_loss(_pred(), _batch(positive=False), cfg)
    for k in ("dead_np_ft", "dead_np_dice", "dead_hv_mse", "dead_hv_msge"):
        assert terms[k] == 0.0
    loss.backward()  # must not raise "does not require grad"


def test_dead_terms_positive_and_neg_bce():
    cfg = TrainConfig(split=1, out_dir="x", dead_expert=True, dead_neg_w=0.1)
    _, terms = cellvit_loss(_pred(), _batch(positive=True), cfg)
    assert terms["dead_np_ft"] > 0 and terms["dead_hv_mse"] >= 0
    _, terms = cellvit_loss(_pred(), _batch(positive=False), cfg)
    assert terms["dead_neg_bce"] > 0  # negative images push dead-fg logits down


def test_default_cfg_no_dead_terms():
    loss, terms = cellvit_loss(_pred(), _batch(), TrainConfig(split=1, out_dir="x"))
    assert not any(k.startswith("dead_") for k in terms)
```

- [ ] **Step 2: RED** → **Step 3: Implement**, appended in `cellvit_loss` before the return:

```python
if cfg is not None and cfg.dead_expert and "np_dead" in pred:
    pos = batch["dead_pos"].bool()
    zero = 0.0 * pred["np_dead"].sum()  # keep the graph alive on all-negative batches
    if pos.any():
        sel = lambda t: t[pos]
        t_dnp = F.one_hot(sel(batch["dead_np_map"]), 2).float()
        p_dnp = F.softmax(sel(pred["np_dead"]), -1)
        terms["dead_np_ft"] = focal_tversky(p_dnp, t_dnp)
        terms["dead_np_dice"] = dice_loss(t_dnp, p_dnp)
        terms["dead_hv_mse"] = 2.5 * mse_loss(sel(batch["dead_hv_map"]), sel(pred["hv_dead"]))
        terms["dead_hv_msge"] = 8.0 * msge_loss(sel(batch["dead_hv_map"]), sel(pred["hv_dead"]), t_dnp[..., 1])
    else:
        for k in ("dead_np_ft", "dead_np_dice", "dead_hv_mse", "dead_hv_msge"):
            terms[k] = zero
    if cfg.dead_neg_w and (~pos).any():
        neg_logits = pred["np_dead"][~pos][..., 1]
        terms["dead_neg_bce"] = cfg.dead_neg_w * F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits))
```

Same coefficients as the main branches (spec §4 "同式损失族"). `train()`/`validate()` datasets: pass `dead_targets=cfg.dead_expert` to both `PanNukeCellViT(...)` constructors.

- [ ] **Step 4: GREEN** + full suite → **Step 5: Commit** `feat: positive-image masked dead-expert loss`

### Task 4: TTA — remap `hv_dead` under dihedral

**Files:**
- Modify: `src/nucseg/hovernet/engine.py` (`undo_dihedral`)
- Test: `tests/test_dead_tta.py`

- [ ] **Step 1: Failing test**

```python
def test_undo_dihedral_remaps_all_hv_keys():
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    hv = torch.randn(2, 4, 4, 2)
    for k in range(4):
        for flip in (False, True):
            out = undo_dihedral({"hv": hv.clone(), "hv_dead": hv.clone()}, k, flip)
            assert torch.allclose(out["hv_dead"], out["hv"])
```

- [ ] **Step 2: RED** → **Step 3:** replace the hardcoded `out["hv"]` block with a loop over `[n for n in out if n.startswith("hv")]` applying the identical flip/sign/swap remap; non-HV keys unchanged. → **Step 4:** `python -m pytest tests/test_dead_tta.py tests/test_hovernet.py -q` then **full suite** (HoVer-Net plumbing touched) → **Step 5: Commit** `fix: dihedral HV remap covers the dead-expert HV channel`

### Task 5: CLI wiring, resume safety, dead-aware prediction

**Files:**
- Modify: `scripts/train_cellvit.py`, `src/nucseg/cellvit/engine.py` (`build_model`, `predict_fold`), `scripts/predict_cellvit.py`
- Test: `tests/test_dead_cli.py` (follow `tests/test_hv_training.py`'s stubbed-trainer pattern)

**Interfaces:**
- CLI: `--dead-expert`, `--dead-neg-w` (float, default 0.0), `--widen` (int, default 0); `build_model(cfg)` passes `dead_expert=cfg.dead_expert and widen=cfg.widen`.
- `predict_fold(..., dead_expert: bool = False)` returns dead maps as a **final extra element** `(N, H, W) int32` when requested (same optional-tail pattern as `inst_probs`); per-patch decode builds `m_dead = concat([zeros(H,W,1), P(np_dead fg), hv_dead], -1)` → existing `_post` → `dead_inst`, with the fake all-zero tp channel giving type-0 instances (kept by `_post`; we force type later at merge). Prediction artifacts use a separate tag: `pred_fold{k}[_x2][_du..]_dead.npz` with keys `dead_inst`.
- `_forward_probs`/`_tta_forward` pass `"np_dead"` (softmaxed) and `"hv_dead"` (raw) through when present.

- [ ] **Step 1: Failing tests** (stub patterns copied from `tests/test_hv_training.py`)

```python
def test_cli_config_and_resume_freeze(tmp_path, stub_trainer):
    # (a) new fields reach TrainConfig
    cfg = main(["--split", "1", "--out", str(tmp_path / "r"), "--dead-expert",
                "--dead-neg-w", "0.1", "--widen", "64", "--train-only"])
    assert cfg.dead_expert and cfg.dead_neg_w == 0.1 and cfg.widen == 64
    # (b) resume with a changed scientific key is refused, original config bytes intact
    before = (tmp_path / "r" / "config.json").read_bytes()
    with pytest.raises(ValueError, match="dead_expert"):
        main(["--split", "1", "--out", str(tmp_path / "r"), "--train-only"])  # dead_expert now False
    assert (tmp_path / "r" / "config.json").read_bytes() == before


def test_predict_fold_dead_tail(stub_model_with_dead_channels, tiny_fold_ds, tmp_path):
    # dead maps come back as the optional final element, int32, ids disjoint from base
    *_, dead = predict_fold(stub_model_with_dead_channels, tiny_fold_ds, dead_expert=True,
                            upscale=1, workers=0)
    assert dead.dtype == np.int32


def test_dead_tag_never_overwrites_base(run_artifact_dir):
    # pred_fold2.npz exists; the dead pass must write pred_fold2_dead.npz and leave it untouched
    before = (run_artifact_dir / "pred_fold2.npz").read_bytes()
    write_dead_predictions(run_artifact_dir, fold=2)
    assert (run_artifact_dir / "pred_fold2.npz").read_bytes() == before
    assert (run_artifact_dir / "pred_fold2_dead.npz").exists()
```
- [ ] **Step 2: RED** → **Step 3: Implement** → **Step 4: GREEN + full suite** → **Step 5: Commit** `feat: dead-expert CLI, resume guard and fold-2 dead prediction artifacts`

### Task 6: Gate v0 — candidate merge + frozen dev evaluation

**Files:**
- Create: `src/nucseg/postproc/dead_merge.py`, `scripts/dsb_dev_eval.py`
- Test: `tests/test_dead_merge.py` (fixture style of `tests/test_existence_ef.py`)

**Interfaces:**
- `merge_dead(base, base_type, dead_inst, min_area: int) -> tuple[np.ndarray, np.ndarray, list[int]]` — adds whole, **fully disjoint** candidate objects (zero base overlap, same rule as `scale_fusion.fuse`) with `area >= min_area`, typed `DEAD_TYPE=4`, renumbered above `base.max()`; base pixels byte-identical.
- `scripts/dsb_dev_eval.py --run <dir> --fold 2`: reads `pred_fold2.npz` + `pred_fold2_dead.npz`, applies menu `[(identity), (a30), (a60)]` → `min_area ∈ {0, 30, 60}` (0 = identity), evaluates each via `pannuke_eval.evaluate`, writes `dsb_dev_eval/menu.json` with all rows + the frozen selection via `select_menu(rows) -> str` (importable from the script for tests): **argmax ΔDead_PQ subject to ΔbPQ ≥ −0.001, tie → smaller min_area; if none qualifies → identity**.

- [ ] **Step 1: Failing tests**

```python
def _maps():
    base = np.zeros((16, 16), np.int32); base[0:4, 0:4] = 1      # base instance
    btyp = np.zeros((16, 16), np.uint8); btyp[base == 1] = 2
    dead = np.zeros((16, 16), np.int32)
    dead[8:11, 8:11] = 1   # disjoint candidate, area 9
    dead[12:15, 2:4] = 2   # disjoint candidate, area 6
    dead[0:3, 0:3] = 3     # overlaps the base instance -> always rejected
    return base, btyp, dead


def test_merge_dead_rules():
    base, btyp, dead = _maps()
    out, typ, chosen = merge_dead(base, btyp, dead, min_area=0)
    assert (out[base > 0] == base[base > 0]).all() and typ[base > 0].tolist() == btyp[base > 0].tolist()
    assert 3 not in chosen and set(chosen) == {1, 2}      # id 3 overlaps base
    assert set(np.unique(out)) - set(np.unique(base)) == {2, 3}  # renumbered above base.max()
    _, _, chosen30 = merge_dead(base, btyp, dead, min_area=30)
    assert chosen30 == []                                  # both below floor


def test_selection_rule_frozen():
    rows = [{"name": "identity", "d_dead": 0.0, "d_bpq": 0.0},
            {"name": "a30", "d_dead": 0.006, "d_bpq": -0.0008},
            {"name": "a60", "d_dead": 0.008, "d_bpq": -0.002}]
    assert select_menu(rows) == "a30"   # a60's larger Dead gain fails the -0.001 bPQ guard
    rows[1]["d_dead"] = -0.001
    assert select_menu(rows) == "identity"  # nothing qualifies -> identity fallback
```
- [ ] **Step 2: RED** → **Step 3: Implement** (`merge_dead` reuses `scale_fusion.candidate_info` + `existence.candidate_geometry`; no GT in `dead_merge.py` itself) → **Step 4: GREEN + full suite (decode/metrics-adjacent)** → **Step 5: Commit** `feat: EF-style dead-candidate merge and frozen dev-fold selection`

### Task 7: Sanity montage builder (CPU, used by Task 13's sonnet review)

**Files:**
- Create: `scripts/dsb_montage.py`
- Test: `tests/test_dsb_montage.py`

- [ ] **Step 1: Failing test**: `build_grid(patches: list[np.ndarray], cols) -> np.ndarray` deterministic layout (shapes, no RNG); stratified sampler `stratify(records, tissue, border, area_bins, k)` returns exactly k indices, respecting strata proportions within ±1 (seeded).
- [ ] **Step 2: RED** → **Step 3: Implement** (follow `scripts/x2_montage.py` drawing conventions; rows = image crop with base instances outlined green, dead candidates outlined orange (TP/FP unknowable without GT — the reviewer sees overlays only)) → **Step 4: GREEN** → **Step 5: Commit** `feat: stratified DSB montage builder`

### Task 8: Gate D — Dead oracle ceiling (CPU, runs immediately)

**Files:**
- Create: `scripts/dsb_oracle_ceiling.py`
- Test: `tests/test_dsb_gates.py` (section `gate_d`)

**Interfaces:**
- CLI: `--run <dir>` (must contain `pred_fold2.npz`; prefer `runs/hv_threshold_controls_20261006/x1_hv30_seed19`, fall back to any historical split1 base run — **refuse** if neither exists), `--fold 2 --drop-frac 0.3 --seed 20261007 --out <json>`.
- Logic (helpers `unmatched_internal_dead(gt_inst, gt_typ, pred) -> list[int]` and `shrink_mask(mask, drop_frac, rng) -> np.ndarray`, both importable for tests): per image, GT Dead instances unmatched by any prediction (IoU > 0.5, same greedy pairing as the evaluator via `nucseg.metrics.instance.overlap`) **and internal** (GT mask border flag false, `existence.candidate_geometry`) get inserted as new predictions: exact mask (IoU = 1.0 ceiling) and shrunk variant (seeded per-instance dropout of 30% of mask pixels → IoU exactly 0.7). Evaluate before/after with `pannuke_eval.evaluate` → ΔDead PQ, ΔmPQ, n_added. Verdict: `pass` iff ΔDead(internal, IoU 0.7) ≥ +0.010 (spec §6 D).

- [ ] **Step 1: Failing tests**

```python
def test_unmatched_internal_dead_finder():
    gt_inst = np.zeros((16, 16), np.int32); gt_typ = np.zeros((16, 16), np.uint8)
    gt_inst[0:3, 0:3] = 1; gt_typ[gt_inst == 1] = 4   # internal, missed -> ADD
    gt_inst[0:3, 14:16] = 2; gt_typ[gt_inst == 2] = 4 # border, missed    -> skip
    gt_inst[8:11, 8:11] = 3; gt_typ[gt_inst == 3] = 4 # internal, matched -> skip
    pred = np.zeros((16, 16), np.int32); pred[8:11, 8:11] = 1   # IoU 1.0 with GT 3
    to_add = unmatched_internal_dead(gt_inst, gt_typ, pred)
    assert to_add == [1]


def test_dropout_shrink_exact_iou_and_determinism():
    mask = np.zeros((16, 16), bool); mask[4:8, 4:9] = True    # area 20
    s1 = shrink_mask(mask, drop_frac=0.3, rng=np.random.default_rng(7))
    s2 = shrink_mask(mask, drop_frac=0.3, rng=np.random.default_rng(7))
    assert (s1 == s2).all() and s1.sum() == 14                # 20 * (1 - 0.3)
    inter, union = (s1 & mask).sum(), (s1 | mask).sum()
    assert abs(inter / union - 0.7) < 1e-9                    # dropout of f gives IoU exactly 1-f
```
- [ ] **Step 2: RED** → **Step 3: Implement** → **Step 4: GREEN + full suite** → **Step 5: Commit** `feat: gate D dead-oracle ceiling probe`, then **run it for real (CPU)**: `python scripts/dsb_oracle_ceiling.py --run <chosen> --fold 2 --out runs/analysis/dsb_gates_20261007/gate_d.json`; paste the verdict line into Task 12's ledger (the analysis output itself lives outside Git under `runs/`).

### Task 9: Gate C — per-class decision decoupling (code now; GPU dump later)

**Files:**
- Modify: `src/nucseg/postproc/recovery.py` (`postprocess`/`proc_np_hv` accept an (H,W) threshold map)
- Create: `scripts/dsb_gate_c.py`
- Test: extend `tests/test_recovery.py`, add gate_c section to `tests/test_dsb_gates.py`

**Interfaces:**
- `Recovery` unchanged; new `thr_map: np.ndarray | None` argument threading: `proc_np_hv(pred, thr=thr_map)` already broadcasts an (H,W) array (`blb_raw >= thr`) — thread it through `postprocess` the same way `cfg.thr` flows, with the parity test **uniform 0.5 map == scalar 0.5 == official**.
- `dsb_gate_c.py --maps <dump dir> --fold 2`: frozen menu `τ_dead ∈ {0.5(identity), 0.4, 0.35, 0.3}`, all other classes 0.5; threshold map = `τ[predicted tp argmax]`; decode fold2 from dumped maps (`dump_cellvit_maps.py` layout: `maps_fold2.npy` (N,256,256,9) = [P_NP(fg), HVx, HVy, P_TP(0..5)]); evaluate each config; verdict `close_line` iff any τ_dead gives ΔDead ≥ +0.005 with ΔbPQ tax ≤ 0.001 (spec §6 C), else `proceed`.

- [ ] **Step 1: Failing tests**

```python
def test_thr_map_parity_with_official(rng_maps):  # existing fixtures in test_recovery.py
    thr = np.full((256, 256), 0.5)
    a = decode_pred_map(rng_maps, nr_types=6)
    b = postprocess_thr_map(rng_maps, thr_map=thr)   # new entry: same decode, per-pixel threshold
    assert (a[0] == b[0]).all()


def test_thr_map_varies_only_where_class_matches():
    # tau_dead 0.3 on pixels the TP argmax calls Dead, 0.5 elsewhere: the decoded blob set can
    # only grow at Dead-typed pixels; every non-Dead instance must be bit-identical
    base = run_decode(thr_dead=0.5)
    low = run_decode(thr_dead=0.3)
    assert base_dead_count(low) >= base_dead_count(base)


def test_gate_c_verdict_rule():
    rows = [{"tau_dead": 0.5, "d_dead": 0.0, "d_bpq": 0.0},
            {"tau_dead": 0.4, "d_dead": 0.006, "d_bpq": -0.0005}]
    assert gate_c_verdict(rows) == "close_line"   # decision decoupling alone already wins
    rows[1] = {"tau_dead": 0.4, "d_dead": 0.003, "d_bpq": -0.004}
    assert gate_c_verdict(rows) == "proceed"
```
- [ ] **Step 2: RED** → **Step 3: Implement** → **Step 4: GREEN + full suite** → **Step 5: Commit** `feat: per-pixel decode threshold + gate C decision-decoupling probe (CPU stage)`

### Task 10: Gate B — gradient-conflict probe (code now; GPU run after E2a)

**Files:**
- Create: `scripts/dsb_gradient_probe.py`
- Test: `tests/test_dsb_gradient_probe.py`

**Interfaces:**
- CLI: `--run <base ckpt dir> --batches 64 --batch 8 --device cuda --out <json>`.
- Per batch (fold1 train split, `PanNukeCellViT([1], train=False)`, fixed order, fp32): `L_dead` = per-pixel mean of NP-CE + TP-CE + HV-MSE on pixels with `tp_map == 4`; `L_common` = same on foreground pixels with `tp_map != 4` (background excluded from both, recorded in output metadata). `torch.autograd.grad(..., retain_graph=True)`; cosine of flattened gradients per group {`skips` (skip0–3), `np_branch`, `hv_branch`, `tp_branch`, `encoder`}; conflict fraction = share of the 64 batches with cosine < 0.
- Verdict (spec §6 B): `proceed` iff conflict fraction ≥ 0.30 on the decoder composite (skips+np+hv+tp).

- [ ] **Step 1: Failing tests**

```python
def test_group_cosine_math():
    g = torch.tensor([1.0, 2.0])
    assert abs(grad_cosine(g, g) - 1.0) < 1e-9
    assert abs(grad_cosine(g, -g) + 1.0) < 1e-9
    assert abs(grad_cosine(g, torch.tensor([0.0, 5.0]))) < 1e-9


def test_conflict_fraction():
    assert conflict_fraction([0.2, -0.1, -0.3, 0.0]) == 0.5   # 2 of 4 batches negative


def test_loss_group_masking():
    logits = torch.zeros(1, 4, 4, 6, requires_grad=True)
    tp = torch.zeros(1, 4, 4, dtype=torch.long); tp[0, :2] = 4
    l_dead = group_pixel_loss(logits, tp, group="dead")
    l_common = group_pixel_loss(logits, tp, group="common")
    assert l_dead.grad_fn is not None
    # swapping the dead pixels' class moves only the dead-group loss
    tp2 = tp.clone(); tp2[tp2 == 4] = 2
    assert group_pixel_loss(logits, tp2, "dead") == 0.0       # no Dead pixels -> zero (attached)
```
- [ ] **Step 2: RED** → **Step 3: Implement** → **Step 4: GREEN** → **Step 5: Commit** `feat: gate B class-group gradient-conflict probe`

### Task 11: Gate A reader (CPU; runs once HV analysis exists)

**Files:**
- Create: `scripts/dsb_gate_a.py`
- Test: gate_a section in `tests/test_dsb_gates.py`

**Interfaces:**
- CLI: `--hv-numbers <hv_control_numbers output.json> --out <json>`; extracts val-fold2 Dead PQ (and bPQ) for arms `x1_hv30`, `x1_hv8`, `x2_hv30` from `arms`/`contrasts`; rules (spec §6 A): with `r = Dead(x1_hv8) − Dead(x1_hv30)`, `R = Dead(x2_hv30) − Dead(x1_hv30)`: `R ≤ 0` → base `hv30`, proceed; `r ≥ 0.9·R` **and** bPQ(x1_hv8) − bPQ(x1_hv30) ≥ −0.001 → `downgrade` (architecture premise lost; return to user); `r ≥ 0.5·R` → base `hv8`, proceed; else base `hv30`, proceed.
- Schema note: adapt parser + fixture in the same commit if the real JSON nests differently (audit item 4).

- [ ] **Step 1: Failing tests** over a synthetic numbers JSON (all four rule branches)

```python
def _nums(dead30, dead8, deadx2, bpq30=0.6, bpq8=0.6):
    return {"arms": {a: {"Dead PQ": d} for a, d in
                     [("x1_hv30", dead30), ("x1_hv8", dead8), ("x2_hv30", deadx2)]},
            "contrasts": {}}


def test_gate_a_rules():
    assert gate_a(_nums(.170, .176, .180))["base"] == "hv8"          # r=.006 >= .5*R=.005
    assert gate_a(_nums(.170, .1795, .180))["verdict"] == "downgrade" # r=.0095 >= .9*R, no tax
    assert gate_a(_nums(.170, .172, .180))["base"] == "hv30"         # r=.002 < .5*R
    assert gate_a(_nums(.170, .175, .175))["verdict"] == "proceed"   # R<=0: HV support not the driver
    assert gate_a(_nums(.170, .1795, .180, bpq8=.598))["verdict"] == "proceed"  # tax blocks downgrade
```
- [ ] **Step 2: RED** → **Step 3: Implement** → **Step 4: GREEN** → **Step 5: Commit** `feat: gate A E2a-contrast reader`

### Task 12: Gates verdict doc + experiment log entry (CPU)

**Files:**
- Modify: `docs/RESEARCH_PLAN.md` (additive progress note), this file (checkboxes), `docs/findings.md` (when results exist)

- [ ] **Step 1:** Create `runs/analysis/dsb_gates_20261007/README.md` listing each gate's command, output JSON path and verdict line (gates B/C GPU stages marked *pending-E2a* until run).
- [ ] **Step 2:** Add a dated progress paragraph to `docs/RESEARCH_PLAN.md` pointing at the spec, this plan, and the gate table; no results claimed that don't exist yet.
- [ ] **Step 3:** `python -m pytest -q` → green; `git diff --check` clean → **Commit** `docs: DSB gates ledger and research-plan progress note`

### Task 13: GPU stages, after E2a drains and the user confirms (checkpoint!)

**Preconditions (all four, verified in order):** E2a queues empty (runbook check); `hv_control_numbers.py` analysis exists (HV branch's task); gates B, C executed; gates A–D verdicts recorded in Task 12's ledger. **Any gate failed → stop, report, do not train.**

- [ ] **Step 1: Gate B run** (~30–60 min A100): `python scripts/dsb_gradient_probe.py --run <C0 base> --out runs/analysis/dsb_gates_20261007/gate_b.json`; record verdict.
- [ ] **Step 2: Gate C maps + scan**: `dump_cellvit_maps.py --run <C0 base> --fold 2 --out data/cache/maps/<base>` (GPU ~minutes; keep on LM2, `df` first), then `dsb_gate_c.py` (CPU); record verdict.
- [ ] **Step 3: Gate A read**: `dsb_gate_a.py` on the HV numbers JSON; record base arm (`hv30`/`hv8`) or the downgrade verdict.
- [ ] **Step 4: Smoke run** (2 epochs, `--train-only`, split1, seed 19, DSB arm) → `predict_cellvit.py --fold 2 --no-eval` + dead preds → `dsb_montage.py` → **sonnet subagent blind review** (spec §8: stratified montage, reviewer told layouts but not which arm; look for BN-drift artifacts, global over-firing, Uterus-collapse). Findings → fix → re-smoke; only a clean review proceeds.
- [ ] **Step 5: Dev round launch** (4 runs, sequential queue per runbook, batch 16 — never reduced to squeeze a busy device): `dsb_seed19`, `dsb_seed1`, `c1_seed<solve_c1_widen()>_seed19`, `c1_..._seed1`, out dirs `runs/dsb_dev_20261007/<arm>_seed<seed>`; each: `train_cellvit.py --split 1 --train-only --dead-expert [--dead-neg-w 0.1 if the smoke review flagged over-firing, else omitted]` / `--widen <solved>` for C1; then fold-2 predict (base + dead tags) + `dsb_dev_eval.py`. The `dead_neg_w` menu item is frozen **before** the first full run: default 0.0; switch to 0.1 only on the smoke-review over-firing finding (global background firing), never re-chosen after seeing dev endpoints.
- [ ] **Step 6: Record** all four arms' menu tables + selections in the gates ledger; **stop before any fold3 read** and report to the user (single test read is its own user checkpoint).

### Task 14: Dev-round analysis and freeze (CPU; after Task 13)

- [ ] **Step 1:** `scripts/dsb_dev_eval.py` outputs consolidated; check both seeds same sign on ΔDead (spec §3 dev entry rule); assemble montage of dev final predictions → second sonnet blind review.
- [ ] **Step 2:** Write `docs/findings.md` dated entry (numbers only from artifacts; negative results included), including the C3 cost reference: DSB single-model inference p50/p95 vs base x1 and vs the x1+x2+EF two-model system, measured with the E0-style timing protocol on the same device/batch (spec §3 secondary endpoints); update plan checkboxes; `pytest -q`; commit `analysis: DSB dev round results`.
- [ ] **Step 3:** Report to user: gates + dev verdict → decide confirmatory round (3 folds × 3 seeds, spec §7) as a **new plan**; the single frozen fold3 read for the dev arms happens only on explicit user approval.
