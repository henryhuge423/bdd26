"""HoVer-Net training / inference on PanNuke (official model, losses, schedule, post-processing)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..constants import NUM_CLASSES
from .data import PAD, PanNukeHoVer, pad_image
from .official import (
    HoVerNet, convert_pytorch_checkpoint, dice_loss, msge_loss, mse_loss, post_process, xentropy_loss,
)

NR_TYPES = NUM_CLASSES + 1


@dataclass
class Phase:
    epochs: int
    batch_size: int
    freeze: bool
    lr: float = 1.0e-4
    step: int = 25  # StepLR step (epochs), gamma 0.1


@dataclass
class TrainConfig:
    split: int
    out_dir: str
    pretrained: str = "weights/hovernet/ImageNet-ResNet50-Preact_pytorch.tar"
    phases: list = field(default_factory=lambda: [Phase(50, 16, True), Phase(50, 4, False)])
    workers: int = 8
    amp: bool = False
    seed: int = 10
    val_every: int = 5


def build_model(freeze: bool) -> HoVerNet:
    return HoVerNet(input_ch=3, nr_types=NR_TYPES, freeze=freeze, mode="fast")


def forward_crop(model, imgs: torch.Tensor) -> dict:
    """imgs (B, 352, 352, 3) uint8 -> dict of (B, 256, 256, C) raw outputs."""
    x = imgs.permute(0, 3, 1, 2).float()
    out = model(x)
    m = (next(iter(out.values())).shape[-1] - 256) // 2
    return {k: v[..., m:m + 256, m:m + 256].permute(0, 2, 3, 1) for k, v in out.items()}


def hovernet_loss(pred: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    true_np = F.one_hot(batch["np_map"], 2).float()
    true_tp = F.one_hot(batch["tp_map"], NR_TYPES).float()
    p_np = F.softmax(pred["np"], -1)
    p_tp = F.softmax(pred["tp"], -1)
    terms = {
        "np_bce": xentropy_loss(true_np, p_np), "np_dice": dice_loss(true_np, p_np),
        "hv_mse": mse_loss(batch["hv_map"], pred["hv"]),
        "hv_msge": msge_loss(batch["hv_map"], pred["hv"], true_np[..., 1]),
        "tp_bce": xentropy_loss(true_tp, p_tp), "tp_dice": dice_loss(true_tp, p_tp),
    }
    return sum(terms.values()), {k: float(v) for k, v in terms.items()}


def _to_device(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _save(path: Path, **state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(cfg: TrainConfig, train_folds: list[int], val_folds: list[int], device="cuda"):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    writer = SummaryWriter(out / "tb")
    log = open(out / "log.jsonl", "a")

    ckpt_path = out / "last.pth"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else None
    start_phase = state["phase"] if state else 0
    start_epoch = state["epoch"] + 1 if state else 0

    train_ds = PanNukeHoVer(train_folds, train=True, seed=cfg.seed)
    val_ds = PanNukeHoVer(val_folds, train=False)
    val_dl = DataLoader(val_ds, batch_size=16, num_workers=cfg.workers, shuffle=False)

    global_step = state["global_step"] if state else 0
    for pi, ph in enumerate(cfg.phases):
        ph = Phase(**ph) if isinstance(ph, dict) else ph
        if pi < start_phase:
            continue
        model = build_model(ph.freeze).to(device)
        if state is not None and pi == start_phase:
            model.load_state_dict(state["model"])
        elif pi == 0:
            sd = convert_pytorch_checkpoint(torch.load(cfg.pretrained, map_location="cpu", weights_only=False)["desc"])
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[pretrained] missing {len(missing)} unexpected {len(unexpected)}")
        else:
            prev = torch.load(out / f"phase{pi - 1}_final.pth", map_location="cpu", weights_only=False)
            model.load_state_dict(prev["model"])
        opt = torch.optim.Adam(model.parameters(), lr=ph.lr, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.StepLR(opt, ph.step)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        ep0 = 0
        if state is not None and pi == start_phase:
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            scaler.load_state_dict(state["scaler"])
            ep0 = start_epoch
        train_dl = DataLoader(train_ds, batch_size=ph.batch_size, shuffle=True, drop_last=True,
                              num_workers=cfg.workers, pin_memory=True, persistent_workers=True)
        for ep in range(ep0, ph.epochs):
            model.train()
            t0, agg, n = time.time(), defaultdict(float), 0
            for batch in train_dl:
                batch = _to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                    pred = forward_crop(model, batch["img"])
                pred = {k: v.float() for k, v in pred.items()}
                loss, terms = hovernet_loss(pred, batch)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                for k, v in terms.items():
                    agg[k] += v
                agg["loss"] += float(loss)
                n += 1
                global_step += 1
            sched.step()
            rec = {"phase": pi, "epoch": ep, "lr": opt.param_groups[0]["lr"], "time": time.time() - t0,
                   **{f"train/{k}": v / n for k, v in agg.items()}}
            if (ep + 1) % cfg.val_every == 0 or ep + 1 == ph.epochs:
                rec.update(validate(model, val_dl, device, cfg.amp))
            for k, v in rec.items():
                if isinstance(v, float):
                    writer.add_scalar(k, v, global_step)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            _save(ckpt_path, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  scaler=scaler.state_dict(), phase=pi, epoch=ep, global_step=global_step)
        _save(out / f"phase{pi}_final.pth", model=model.state_dict())
        state = None
    _save(out / "final.pth", model=model.state_dict())
    writer.close()


@torch.no_grad()
def validate(model, dl, device, amp=False) -> dict:
    model.eval()
    agg, n = defaultdict(float), 0
    inter = union = 0.0
    for batch in dl:
        batch = _to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = forward_crop(model, batch["img"])
        pred = {k: v.float() for k, v in pred.items()}
        loss, terms = hovernet_loss(pred, batch)
        for k, v in terms.items():
            agg[k] += v
        agg["loss"] += float(loss)
        fg = pred["np"].argmax(-1) == 1
        gt = batch["np_map"] == 1
        inter += float((fg & gt).sum())
        union += float(fg.sum() + gt.sum())
        n += 1
    out = {f"val/{k}": v / n for k, v in agg.items()}
    out["val/fg_dice"] = 2 * inter / max(union, 1)
    return out


# ---------------------------------------------------------------- inference

def _post(args):
    pred_map, = args
    inst, info = post_process(pred_map, nr_types=NR_TYPES)
    typ = np.zeros(inst.shape, np.uint8)
    keep = np.zeros(inst.shape, bool)
    for iid, d in info.items():
        m = inst == iid
        keep |= m
        typ[m] = d["type"] if d["type"] is not None else 0
    inst = np.where(keep, inst, 0).astype(np.int32)
    return inst, typ


@torch.no_grad()
def predict_fold(model, fold_ds, device="cuda", batch_size=32, workers=16, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Run HoVer-Net on every patch of a fold; returns (inst, type) arrays in fold order."""
    model.eval()
    maps = []
    for s in range(0, len(fold_ds), batch_size):
        imgs = np.stack([pad_image(np.asarray(fold_ds.images[i])) for i in range(s, min(s + batch_size, len(fold_ds)))])
        x = torch.from_numpy(imgs).to(device)
        p = _tta_forward(model, x) if tta else _softmaxed(forward_crop(model, x))
        tp = p["tp"].argmax(-1, keepdim=True).float()
        np_ = p["np"][..., 1:]
        maps.append(torch.cat([tp, np_, p["hv"]], -1).cpu().numpy())
    maps = np.concatenate(maps)
    with Pool(workers) as pool:
        res = pool.map(_post, ((m,) for m in maps), chunksize=8)
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def _softmaxed(p: dict) -> dict:
    return {"np": F.softmax(p["np"], -1), "tp": F.softmax(p["tp"], -1), "hv": p["hv"]}


def dihedral(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Spatial transform on (B, H, W, ...) tensors: rotate k*90 degrees, then flip width."""
    x = torch.rot90(x, k, (1, 2))
    return x.flip(2) if flip else x


def undo_dihedral(p: dict, k: int, flip: bool) -> dict:
    """Invert `dihedral` on a prediction dict. The HV vector field (x = horizontal, y = vertical)
    needs its components remapped as well as its pixels moved."""
    out = {}
    for name, v in p.items():
        if flip:
            v = v.flip(2)
        out[name] = torch.rot90(v, -k, (1, 2))
    h, v = out["hv"][..., 0], out["hv"][..., 1]
    if flip:
        h = -h
    for _ in range(k % 4):
        h, v = -v, h
    out["hv"] = torch.stack([h, v], -1)
    return out


def _tta_forward(model, x: torch.Tensor) -> dict:
    """8-fold dihedral TTA averaged on softmax probabilities and corrected HV maps."""
    acc = None
    for k in range(4):
        for flip in (False, True):
            p = undo_dihedral(_softmaxed(forward_crop(model, dihedral(x, k, flip))), k, flip)
            acc = p if acc is None else {n: acc[n] + p[n] for n in acc}
    return {n: v / 8 for n, v in acc.items()}
