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
pre-registered): conflict fraction >= 0.30 on the decoder composite (skips + np + hv + tp).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nucseg.cellvit.data import PanNukeCellViT
from nucseg.cellvit.engine import TrainConfig, build_model, run_hv_min_size, run_upscale, to_nhwc
from nucseg.constants import DEAD_TYPE

THRESHOLD = 0.30
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
    return "proceed" if result["decoder"]["conflict_fraction"] >= THRESHOLD else "fail"


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
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cfg = TrainConfig(split=1, out_dir=str(a.run), upscale=run_upscale(a.run),
                      hv_min_size=run_hv_min_size(a.run))
    model = build_model(cfg, pretrained=False).to(a.device)
    model.load_state_dict(torch.load(a.run / a.ckpt, map_location=a.device,
                                     weights_only=False)["model"])
    model.eval()  # deterministic forward (no dropout/stochastic depth); grads still flow
    ds = PanNukeCellViT([a.fold], train=False, small_area=cfg.small_area,
                        upscale=cfg.upscale, hv_min_size=cfg.hv_min_size)
    cos: dict[str, list[float]] = {g: [] for g in GROUPS}
    torch.manual_seed(0)
    for s in range(0, min(a.batches * a.batch, len(ds)), a.batch):
        batch = torch.utils.data.default_collate([ds[i] for i in range(s, s + a.batch)])
        batch = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
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
    result = {"run": str(a.run), "fold": a.fold, "batches": a.batches, "batch": a.batch,
              "groups_excluded_background": "L_dead on GT Dead pixels, L_common on non-Dead "
                                            "foreground pixels; background belongs to neither",
              "threshold_conflict_fraction": THRESHOLD,
              "modules": {g: {"mean_cosine": sum(v) / len(v) if v else 0.0,
                              "conflict_fraction": conflict_fraction(v)} for g, v in cos.items()},
              "decoder": None}
    result["decoder"] = result["modules"]["decoder"]
    result["verdict"] = gate_b_verdict(result)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
