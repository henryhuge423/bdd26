#!/usr/bin/env python
"""Gate B (DSB spec §6): class-group gradient-conflict probe on a trained base checkpoint.

    python scripts/dsb_gradient_probe.py --run runs/hv_threshold_controls_20261006/x1_hv30_seed19 \
        --fold 1 --batches 64 --batch 8 --device cuda \
        --out runs/analysis/dsb_gates_20261007/gate_b.json

H1 says Dead and the common classes contend for SHARED decoder capacity. Per training batch we
form two losses over disjoint pixel groups — L_dead on GT Dead pixels, L_common on non-Dead
foreground pixels (background belongs to neither; recorded in the output) — backprop each, and
take the cosine of the flattened gradients per module group. Conflict fraction = share of
batches whose composite decoder cosine is negative. Continue condition (frozen, arbitrary but
pre-registered): conflict fraction >= 0.30 on the decoder composite (skips + np + hv + tp), OR
per-layer significantly negative mean cosines (spec §6 B).

Amendment (2026-10-08, pre-registered in runs/analysis/dsb_gates_20261007/README.md before the
rerun): the first run was VOID — sequential first-256 sampling hit only Breast images with one
Dead-positive batch, capping the fraction at 1/64 before any gradient geometry was measured.
Batches are now Dead-stratified (seeded shuffled passes over the fold's Dead-positive images
only), cosines/fractions use L_dead-active batches as the denominator (n_active recorded; the
run is VOID if n_active < 16, or if activity was never recorded — v1-schema inputs), and the
per-layer clause is implemented as one-sided one-sample t-tests (mean cosine < 0) per module
group with Holm correction across the six groups.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sps

from nucseg.cellvit.data import PanNukeCellViT
from nucseg.cellvit.engine import TrainConfig, build_model, run_hv_min_size, run_upscale, to_nhwc
from nucseg.constants import DEAD_TYPE

THRESHOLD = 0.30
MIN_ACTIVE = 16
ALPHA = 0.05
SAMPLE_SEED = 20261008
GROUPS = ("encoder", "skips", "np_branch", "hv_branch", "tp_branch", "decoder")


def grad_cosine(a, b) -> float:
    a, b = torch.as_tensor(a).flatten().double(), torch.as_tensor(b).flatten().double()
    denom = a.norm() * b.norm()
    return 0.0 if denom == 0 else float((a @ b) / denom)


def list_cosine(ga, gb) -> float:
    """Cosine over a list of parameter gradients WITHOUT concatenating (a full-model fp64 concat
    allocates GiB-scale spikes; per-parameter fp64 temporaries stay tens of MB)."""
    if not ga or not gb:
        return 0.0
    dot = sum(float(ta.flatten().double() @ tb.flatten().double()) for ta, tb in zip(ga, gb))
    na = sum(float(ta.flatten().double().pow(2).sum()) for ta in ga) ** 0.5
    nb = sum(float(tb.flatten().double().pow(2).sum()) for tb in gb) ** 0.5
    return 0.0 if na * nb == 0 else dot / (na * nb)


def conflict_fraction(cosines) -> float:
    vals = [float(c) for c in cosines]
    return sum(c < 0 for c in vals) / len(vals) if vals else 0.0


def active_conflict_fraction(cosines, active) -> tuple[float, int]:
    """Conflict fraction over L_dead-active batches only (amendment (b)); inactive batches
    leave the denominator entirely instead of counting as structural non-conflicts."""
    vals = [float(c) for c, a in zip(cosines, active) if a]
    return (sum(c < 0 for c in vals) / len(vals) if vals else 0.0), len(vals)


def dead_stratified_batches(dead_idx, n_batches: int, batch: int, seed: int) -> list[list[int]]:
    """Seeded shuffled passes over the Dead-positive indices, cut into fixed-size batches via a
    circular stream (every batch has exactly `batch` images; each pass covers every Dead-positive
    image — amendment (a): the VOID run's sequential sampling measured 63/64 zero-Dead batches)."""
    idx = [int(i) for i in dead_idx]
    if not idx or n_batches <= 0 or batch <= 0:
        return []
    rng = np.random.default_rng(seed)
    stream: list[int] = []
    while len(stream) < n_batches * batch:
        stream.extend(int(j) for j in rng.permutation(idx))
    return [stream[s:s + batch] for s in range(0, n_batches * batch, batch)]


def per_layer_significance(cos_by_group: dict) -> dict:
    """Spec §6 B alternative clause: 'per-layer significantly negative'. One-sided one-sample
    t-tests (H1: mean cosine < 0) over per-batch cosines, Holm-Bonferroni across the module
    groups; significant for any group => clause passes. Caveat (ledger): batches repeat images
    across passes, so independence is approximate — the fraction clause stays primary."""
    modules = {}
    for g, vals in cos_by_group.items():
        v = np.asarray([float(x) for x in vals], dtype=float)
        if len(v) < 2 or not np.isfinite(v).all() or np.std(v) == 0.0:
            modules[g] = {"t": None, "p": 1.0, "p_holm": 1.0, "significant": False, "n": len(v)}
        else:
            t, p = sps.ttest_1samp(v, 0.0, alternative="less")
            modules[g] = {"t": float(t), "p": float(p), "p_holm": None, "significant": None,
                          "n": len(v)}
    running = 0.0
    m = len(modules)
    for rank, g in enumerate(sorted(modules, key=lambda k: modules[k]["p"])):
        running = max(running, (m - rank) * modules[g]["p"])
        modules[g]["p_holm"] = min(1.0, running)
        modules[g]["significant"] = bool(modules[g]["p_holm"] < ALPHA)
    return {"alpha": ALPHA, "modules": modules,
            "any_significant": any(x["significant"] for x in modules.values())}


def group_pixel_loss(np_logits, tp_logits, hv, np_map, tp_map, hv_map, group: str):
    """Per-pixel mean of NP-CE + TP-CE + HV-MSE over the group's pixels; 0.0*sum keeps the graph
    attached when the group is empty (never backwarded on NaN)."""
    sel = (tp_map == DEAD_TYPE) if group == "dead" else (np_map > 0) & (tp_map != DEAD_TYPE)
    if not sel.any():
        return 0.0 * np_logits.sum()
    np_ce = F.cross_entropy(np_logits.permute(0, 3, 1, 2), np_map, reduction="none")[sel].mean()
    tp_ce = F.cross_entropy(tp_logits.permute(0, 3, 1, 2), tp_map, reduction="none")[sel].mean()
    hv_mse = ((hv - hv_map) ** 2).mean(-1)[sel].mean()
    return np_ce + tp_ce + hv_mse


def gate_b_verdict(result: dict) -> str:
    n_active = result.get("n_active")
    if n_active is None:
        return "void"  # v1 schema recorded no activity — validity cannot be certified
    if n_active < MIN_ACTIVE:
        return "void"  # too few informative batches to measure the pre-registered statistic
    if result["decoder"]["conflict_fraction"] >= THRESHOLD:
        return "proceed"
    if (result.get("per_layer") or {}).get("any_significant"):
        return "proceed"
    return "fail"


def dead_positive_indices(ds):
    """Indices of images containing at least one Dead GT nucleus (class_presence column
    DEAD_TYPE-1; unit-pinned so a wrong-column regression fails loudly instead of silently
    sampling another class's positives)."""
    return np.where(ds.class_presence()[:, DEAD_TYPE - 1])[0]


def batch_group_activity(tp_map, np_map) -> tuple[bool, bool]:
    """(L_dead active, L_common active) for one collated batch."""
    return (bool((tp_map == DEAD_TYPE).any()),
            bool(((np_map > 0) & (tp_map != DEAD_TYPE)).any()))


def _params(model, group: str):
    if group == "encoder":
        return [p for p in model.encoder.parameters() if p.requires_grad]
    names = {"skips": ("skip0", "skip1", "skip2", "skip3"), "np_branch": ("np_branch",),
             "hv_branch": ("hv_branch",), "tp_branch": ("tp_branch",),
             "decoder": ("skip0", "skip1", "skip2", "skip3", "np_branch", "hv_branch", "tp_branch")}
    mods = [getattr(model, n) for n in names[group]]
    return [p for m in mods for p in m.parameters()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fold", type=int, default=1, help="training fold to draw batches from")
    p.add_argument("--batches", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default="final.pth")
    p.add_argument("--sample-seed", type=int, default=SAMPLE_SEED,
                   help="seed for the Dead-stratified batch sampling (amendment (a))")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    from fold_guard import ensure_dev_fold, run_split
    split = run_split(a.run)
    ensure_dev_fold(a.fold, split)
    cfg = TrainConfig(split=split, out_dir=str(a.run), upscale=run_upscale(a.run),
                      hv_min_size=run_hv_min_size(a.run))
    model = build_model(cfg, pretrained=False).to(a.device)
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location=a.device,
                                     weights_only=False)["model"])
    model.eval()  # deterministic forward (no dropout/stochastic depth); grads still flow
    ds = PanNukeCellViT([a.fold], train=False, small_area=cfg.small_area,
                        upscale=cfg.upscale, hv_min_size=cfg.hv_min_size)
    dead_idx = dead_positive_indices(ds)
    sampled = dead_stratified_batches(dead_idx, a.batches, a.batch, a.sample_seed)
    cos: dict[str, list[float]] = {g: [] for g in GROUPS}
    active_flags: list[bool] = []
    n_common_active = 0
    torch.manual_seed(0)
    for ids in sampled:
        batch = torch.utils.data.default_collate([ds[i] for i in ids])
        batch = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        dead_active, common_active = batch_group_activity(batch["tp_map"], batch["np_map"])
        active_flags.append(dead_active)
        if not dead_active:
            continue  # L_dead structurally zero: excluded from every denominator (amendment (b))
        if common_active:
            n_common_active += 1  # counted only for batches whose gradients were computed
        pred = to_nhwc(model(batch["img"]))
        losses = {g: group_pixel_loss(pred["np"], pred["tp"], pred["hv"], batch["np_map"],
                                      batch["tp_map"], batch["hv_map"], g)
                  for g in ("dead", "common")}
        grads: dict[str, dict[str, list[torch.Tensor]]] = {}
        for which, loss in losses.items():
            grads[which] = {g: torch.autograd.grad(loss, _params(model, g), retain_graph=True,
                                                   allow_unused=True)
                            for g in GROUPS}
        for g in GROUPS:
            ga = [t for t in grads["dead"][g] if t is not None]
            gb = [t for t in grads["common"][g] if t is not None]
            cos[g].append(list_cosine(ga, gb))
        del pred, losses, grads
    n_active = int(sum(active_flags))
    # cos lists are active-only by construction (inactive batches are skipped before any
    # gradient work), so every recorded cosine belongs to an L_dead-active batch
    assert len(cos[GROUPS[0]]) == n_active, "cosine/active bookkeeping mismatch"
    active = [True] * n_active
    result = {"run": str(a.run), "fold": a.fold, "batches": a.batches, "batch": a.batch,
              "groups_excluded_background": "L_dead on GT Dead pixels, L_common on non-Dead "
                                            "foreground pixels; background belongs to neither",
              "threshold_conflict_fraction": THRESHOLD, "min_active": MIN_ACTIVE,
              "sampling": {"scheme": "dead-stratified seeded shuffled passes (amendment 2026-10-08)",
                           "seed": a.sample_seed,
                           "n_dead_positive_images": int(len(dead_idx)),
                           "batches_sampled": sampled},
              "n_active": n_active, "n_common_active": n_common_active,
              "per_batch_cosines": cos, "active_flags": active_flags,
              "modules": {}, "decoder": None, "per_layer": None, "verdict": None}
    for g, v in cos.items():
        frac, _ = active_conflict_fraction(v, active)
        result["modules"][g] = {"mean_cosine": sum(v) / len(v) if v else 0.0,
                                "conflict_fraction": frac}
    result["decoder"] = result["modules"]["decoder"]
    result["per_layer"] = per_layer_significance(cos)
    result["verdict"] = gate_b_verdict(result)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
