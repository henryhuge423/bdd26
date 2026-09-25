"""CellViT-UNI training / inference on PanNuke (CellViT PanNuke recipe; HoVer-Net post-processing).

Losses and weights follow the CellViT paper config: NP = focal-Tversky + dice, HV = 2.5 MSE + 8 MSGE,
TP = 0.5 CE + 0.2 dice + 0.5 multi-class focal-Tversky, tissue = 0.1 CE. Optimiser AdamW (3e-4,
betas 0.85/0.95, wd 1e-4), exponential LR decay 0.85/epoch, encoder frozen for the first 25 epochs,
130 epochs, cell+tissue weighted sampling. Post-processing and TTA are shared with our HoVer-Net
pipeline so differences between models are attributable to the networks, not to post-processing.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from ..augment.copy_paste import CopyPasteConfig
from ..constants import NUM_CLASSES
from ..hovernet.engine import _post, dihedral, undo_dihedral
from ..text.conch_prior import instance_mean_probs
from ..hovernet.official import dice_loss, mse_loss, msge_loss, xentropy_loss
from .data import PanNukeCellViT, cell_tissue_weights
from .model import UNI_MEAN, UNI_STD, CellViTUNI

NR_TYPES = NUM_CLASSES + 1
DEAD = 4


@dataclass
class TrainConfig:
    split: int
    out_dir: str
    uni_ckpt: str = "weights/MahmoodLab/UNI/pytorch_model.bin"
    epochs: int = 130
    unfreeze_epoch: int = 25
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.85
    sampling_gamma: float = 0.85
    workers: int = 8
    seed: int = 19
    val_every: int = 5
    # targeted foreground supervision (off by default = CellViT recipe): extra pixel-weighted CE on the
    # NP branch, weight 1 + dead_w * [Dead nucleus pixel] + small_w * [pixel of a nucleus < small_area px]
    np_wce: float = 0.0
    dead_w: float = 0.0
    small_w: float = 0.0
    small_area: int = 100
    # failure-driven copy-paste (pillar B control arm CP1); prob 0 disables it
    cp_prob: float = 0.0
    cp_lam: float = 3.0
    cp_dead_w: float = 0.4
    cp_area: tuple = (50, 400)
    cp_clearance: int = 8
    # synthetic data mixing (pillar B): dir with images/inst/type/base .npy, appended at frac of |real|
    synth: str | None = None
    synth_frac: float = 0.0


def focal_tversky(p: torch.Tensor, t: torch.Tensor, alpha=0.7, beta=0.3, gamma=4 / 3, smooth=1e-6,
                  per_class: bool = False) -> torch.Tensor:
    """p, t: (B, H, W, C) probabilities / one-hot. CellViT FocalTverskyLoss (per_class=False, all channels
    pooled) and MCFocalTverskyLoss (per_class=True, summed over classes)."""
    dims = (0, 1, 2) if per_class else (0, 1, 2, 3)
    tp = (p * t).sum(dims)
    fp = ((1 - t) * p).sum(dims)
    fn = (t * (1 - p)).sum(dims)
    tv = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return ((1 - tv) ** gamma).sum()


def to_nhwc(out: dict) -> dict:
    return {k: (v.permute(0, 2, 3, 1).float() if v.ndim == 4 else v.float()) for k, v in out.items()}


def np_pixel_weights(batch: dict, cfg: TrainConfig) -> torch.Tensor:
    w = torch.ones_like(batch["np_map"], dtype=torch.float32)
    if cfg.dead_w:
        w = w + cfg.dead_w * (batch["tp_map"] == DEAD).float()
    if cfg.small_w:
        w = w + cfg.small_w * batch["small_map"].float()
    return w


def cellvit_loss(pred: dict, batch: dict, cfg: TrainConfig | None = None) -> tuple[torch.Tensor, dict]:
    """pred: NHWC logits (np, hv, tp) + tissue logits."""
    t_np = F.one_hot(batch["np_map"], 2).float()
    t_tp = F.one_hot(batch["tp_map"], NR_TYPES).float()
    p_np = F.softmax(pred["np"], -1)
    p_tp = F.softmax(pred["tp"], -1)
    terms = {
        "np_ft": focal_tversky(p_np, t_np),
        "np_dice": dice_loss(t_np, p_np),
        "hv_mse": 2.5 * mse_loss(batch["hv_map"], pred["hv"]),
        "hv_msge": 8.0 * msge_loss(batch["hv_map"], pred["hv"], t_np[..., 1]),
        "tp_ce": 0.5 * xentropy_loss(t_tp, p_tp),
        "tp_dice": 0.2 * dice_loss(t_tp, p_tp),
        "tp_ft": 0.5 * focal_tversky(p_tp, t_tp, per_class=True),
        "tissue_ce": 0.1 * F.cross_entropy(pred["tissue"], batch["tissue"]),
    }
    if cfg is not None and cfg.np_wce:
        ce = F.cross_entropy(pred["np"].permute(0, 3, 1, 2), batch["np_map"], reduction="none")
        terms["np_wce"] = cfg.np_wce * (ce * np_pixel_weights(batch, cfg)).mean()
    return sum(terms.values()), {k: float(v) for k, v in terms.items()}


def _to_device(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _save(path: Path, **state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def build_model(cfg: TrainConfig | None = None, pretrained: bool = True) -> CellViTUNI:
    return CellViTUNI(uni_ckpt=cfg.uni_ckpt if (cfg and pretrained) else None)


def train(cfg: TrainConfig, train_folds: list[int], val_folds: list[int], device="cuda", model=None):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    writer = SummaryWriter(out / "tb")
    log = open(out / "log.jsonl", "a")

    ckpt_path = out / "last.pth"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else None
    model = (model or build_model(cfg, pretrained=state is None)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.85, 0.95), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, cfg.gamma)
    ep0, global_step = 0, 0
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        ep0, global_step = state["epoch"] + 1, state["global_step"]
        print(f"[resume] epoch {ep0}")

    cp = CopyPasteConfig(prob=cfg.cp_prob, lam=cfg.cp_lam, dead_w=cfg.cp_dead_w,
                         area=tuple(cfg.cp_area), clearance=cfg.cp_clearance) if cfg.cp_prob else None
    train_ds = PanNukeCellViT(train_folds, train=True, small_area=cfg.small_area, copy_paste=cp,
                              synth=Path(cfg.synth) if cfg.synth else None,
                              synth_frac=cfg.synth_frac)
    sampler = WeightedRandomSampler(cell_tissue_weights(train_ds, cfg.sampling_gamma), len(train_ds),
                                    replacement=True, generator=torch.Generator().manual_seed(cfg.seed))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True,
                          num_workers=cfg.workers, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(PanNukeCellViT(val_folds, train=False, small_area=cfg.small_area), batch_size=32, num_workers=cfg.workers)

    for ep in range(ep0, cfg.epochs):
        model.freeze_encoder(ep < cfg.unfreeze_epoch)
        model.train()
        t0, agg, n = time.time(), defaultdict(float), 0
        for batch in train_dl:
            batch = _to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = to_nhwc(model(batch["img"]))
            loss, terms = cellvit_loss(pred, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, v in terms.items():
                agg[k] += v
            agg["loss"] += float(loss)
            n += 1
            global_step += 1
        sched.step()
        rec = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "time": time.time() - t0,
               **{f"train/{k}": v / n for k, v in agg.items()}}
        if (ep + 1) % cfg.val_every == 0 or ep + 1 == cfg.epochs:
            rec.update(validate(model, val_dl, device, cfg))
        for k, v in rec.items():
            if isinstance(v, float):
                writer.add_scalar(k, v, global_step)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        _save(ckpt_path, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
              epoch=ep, global_step=global_step)
    _save(out / "final.pth", model=model.state_dict())
    writer.close()
    return model


@torch.no_grad()
def validate(model, dl, device, cfg: TrainConfig | None = None) -> dict:
    model.eval()
    agg, n = defaultdict(float), 0
    inter = union = correct = total = 0.0
    for batch in dl:
        batch = _to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = to_nhwc(model(batch["img"]))
        loss, terms = cellvit_loss(pred, batch, cfg)
        for k, v in terms.items():
            agg[k] += v
        agg["loss"] += float(loss)
        fg = pred["np"].argmax(-1) == 1
        gt = batch["np_map"] == 1
        inter += float((fg & gt).sum())
        union += float(fg.sum() + gt.sum())
        correct += float((pred["tissue"].argmax(-1) == batch["tissue"]).sum())
        total += len(batch["tissue"])
        n += 1
    out = {f"val/{k}": v / n for k, v in agg.items()}
    out["val/fg_dice"] = 2 * inter / max(union, 1)
    out["val/tissue_acc"] = correct / max(total, 1)
    return out


# ---------------------------------------------------------------- inference

def _forward_probs(model, imgs_u8: torch.Tensor) -> dict:
    """imgs_u8 (B, H, W, 3) uint8 on device -> NHWC softmaxed np/tp, raw hv, tissue probs."""
    mean = torch.tensor(UNI_MEAN, device=imgs_u8.device)
    std = torch.tensor(UNI_STD, device=imgs_u8.device)
    x = ((imgs_u8.float() / 255.0 - mean) / std).permute(0, 3, 1, 2).contiguous()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = to_nhwc(model(x))
    return {"np": F.softmax(p["np"], -1), "tp": F.softmax(p["tp"], -1), "hv": p["hv"],
            "tissue": F.softmax(p["tissue"], -1)}


def _tta_forward(model, x: torch.Tensor) -> dict:
    """8-fold dihedral TTA (same transforms and HV remapping as the HoVer-Net pipeline)."""
    acc = None
    for k in range(4):
        for flip in (False, True):
            p = _forward_probs(model, dihedral(x, k, flip))
            tissue = p.pop("tissue")
            p = undo_dihedral(p, k, flip)
            p["tissue"] = tissue
            acc = p if acc is None else {n: acc[n] + p[n] for n in acc}
    return {n: v / 8 for n, v in acc.items()}


@torch.no_grad()
def predict_fold(model, fold_ds, device="cuda", batch_size=32, workers=16, tta: bool = False,
                 inst_probs: bool = False):
    """Returns (inst, type, tissue_probs) for every patch of a fold, in fold order; with inst_probs=True
    also a per-instance table (img index, inst id, mean type probabilities (M, 6))."""
    model.eval()
    insts, types, tissues, tab = [], [], [], ([], [], [])
    with Pool(workers) as pool:
        for s in range(0, len(fold_ds), batch_size):
            x = torch.from_numpy(np.asarray(fold_ds.images[s:s + batch_size])).to(device)
            p = _tta_forward(model, x) if tta else _forward_probs(model, x)
            tp = p["tp"].argmax(-1, keepdim=True).float()
            maps = torch.cat([tp, p["np"][..., 1:], p["hv"]], -1).cpu().numpy()
            res = pool.map(_post, ((m,) for m in maps), chunksize=2)
            insts += [r[0] for r in res]
            types += [r[1] for r in res]
            tissues.append(p["tissue"].cpu().numpy())
            if inst_probs:
                prob = p["tp"].cpu().numpy()
                for b, (inst, _) in enumerate(res):
                    ids = np.unique(inst)[1:]
                    tab[0].append(np.full(len(ids), s + b))
                    tab[1].append(ids)
                    tab[2].append(instance_mean_probs(inst, prob[b], ids))
    out = (np.stack(insts), np.stack(types), np.concatenate(tissues))
    if inst_probs:
        out += (tuple(np.concatenate(t) for t in tab),)
    return out
