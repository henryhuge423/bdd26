#!/usr/bin/env python
"""LoRA-adapt PixCell-256 Cell-ControlNet to 40x PanNuke (pillar B, TRAIN folds only).

ROUTE ABANDONED (findings 2026-09-26): r16/5000 over-adapted (od_mean_l1 .098 -> .266); the r8 /
lr 5e-5 snapshot sweep was WORSE than the base model at every checkpoint (od_mean_l1 .60-1.10,
uni_cos .13-.25 vs base .116/.496) — adaptation conflicts with the 20x prior. Kept for reference;
synthesis uses the base model + Reinhard LAB colour matching.

The released generator is 20x/GigaPath-domain; the smoke test (2026-09-26) showed flat chromatin,
pink->violet shift and size-biased under-rendering of large masks on PanNuke 40x. This script
freezes everything and trains LoRA adapters on the DiT attention projections (optionally also the
ControlNet's) with the standard epsilon objective, using exactly the inference conditioning:
controlnet input = VAE latent of the binary mask, context = UNI2-h CLS embedding of the paired
REAL image, CFG dropout 0.1 (dropped rows get zero controlnet contribution — the block injection
is additive, so zeros == "no ControlNet", exactly matching CFG inference).

    python scripts/train_pixcell_lora.py --fold 1 --out runs/pixcell/lora_fold1 --steps 5000
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from nucseg.data.pannuke import PanNukeFold
from nucseg.pixcell import (encode_condition, encode_image, load_pipeline, load_uni2h, save_lora,
                            uni_embed, wrap_lora)

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, required=True, help="TRAIN fold to adapt on (e.g. 1 for split 1)")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=16)
p.add_argument("--dropout-cond", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--val", type=int, default=100, help="held-out patches at the end of the fold")
p.add_argument("--val-every", type=int, default=250)
p.add_argument("--snapshot-every", type=int, default=250,
               help="keep step-stamped adapter snapshots (epsilon-val is too noisy for selection; "
                    "pick checkpoints with the image-space pixcell_eval)")
p.add_argument("--lora-controlnet", action="store_true", help="also LoRA the ControlNet attention")
p.add_argument("--seed", type=int, default=19)
a = p.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
out = a.out
out.mkdir(parents=True, exist_ok=True)
(out / "args.json").write_text(json.dumps(vars(a), indent=1, default=str))

# ------------------------------------------------------------------ cached conditions
cache = out / f"cache_fold{a.fold}.npz"
pipe = load_pipeline()
if cache.exists():
    z = np.load(cache)
    img_lat, msk_lat, emb = z["img_lat"], z["msk_lat"], z["emb"]
    print(f"[cache] {cache}")
else:
    f = PanNukeFold(a.fold)
    uni, tf = load_uni2h()
    img_lat = np.zeros((len(f), 16, 32, 32), np.float16)
    msk_lat = np.zeros((len(f), 16, 32, 32), np.float16)
    emb = np.zeros((len(f), 1536), np.float16)
    for j in range(len(f)):
        img = np.asarray(f.images[j])
        inst = np.asarray(f.inst[j])
        img_lat[j] = encode_image(pipe, img).half().cpu().numpy()[0]
        msk_lat[j] = encode_condition(pipe, np.where(inst > 0, 255, 0).astype(np.uint8)[..., None]
                                      .repeat(3, -1)).half().cpu().numpy()[0]
        emb[j] = uni_embed(uni, tf, img).half().cpu().numpy()[0, 0]
        if (j + 1) % 250 == 0:
            print(f"[encode] {j + 1}/{len(f)}", flush=True)
    np.savez(cache, img_lat=img_lat, msk_lat=msk_lat, emb=emb)
    del uni
    torch.cuda.empty_cache()

n = len(img_lat)
val_idx = np.arange(n - a.val, n)
tr_idx = np.arange(n - a.val)

# ------------------------------------------------------------------ model
from diffusers import DDPMScheduler  # noqa: E402

transformer = wrap_lora(pipe.transformer, a.rank, a.alpha)
controlnet = wrap_lora(pipe.controlnet, a.rank, a.alpha) if a.lora_controlnet else pipe.controlnet
if not a.lora_controlnet:
    for prm in controlnet.parameters():
        prm.requires_grad_(False)
sched_train = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                            beta_schedule="linear", prediction_type="epsilon")
uncond = pipe.transformer.caption_projection.uncond_embedding.detach().clone()[None]  # (1, 1, 1536)
cn_ctx = contextlib.nullcontext if a.lora_controlnet else torch.no_grad

params = [prm for prm in list(transformer.parameters()) + list(controlnet.parameters()) if prm.requires_grad]
print(f"[trainable] {sum(prm.numel() for prm in params) / 1e6:.1f}M params")
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
log = open(out / "log.jsonl", "a")


def batch(idx):
    img = torch.from_numpy(img_lat[idx].astype(np.float32))
    msk = torch.from_numpy(msk_lat[idx].astype(np.float32))
    e = torch.from_numpy(emb[idx].astype(np.float32))
    return img.cuda(), msk.cuda(), e.cuda()


def forward_loss(noisy, t, msk, e, noise):
    """Epsilon MSE; condition-dropped rows get zero controlnet contribution (== CFG's uncond path)."""
    keep = torch.rand(len(e), device=e.device) >= a.dropout_cond
    enc = torch.where(keep[:, None, None], e, uncond.expand(len(e), -1, -1).to(e.dtype))
    with cn_ctx():
        cn_out = controlnet(hidden_states=noisy, conditioning=msk, encoder_hidden_states=enc,
                            timestep=t, return_dict=False)[0]
        # the block injection is additive: zero contribution == the CFG uncond path (no ControlNet)
        full = [torch.where(keep[:, None, None], blk, torch.zeros_like(blk)) for blk in cn_out]
    pred = transformer(noisy, encoder_hidden_states=enc, controlnet_outputs=full,
                       timestep=t, return_dict=False)[0]
    if pred.shape[1] == 2 * noisy.shape[1]:  # learned sigma: predict epsilon only
        pred = pred.chunk(2, dim=1)[0]
    return torch.nn.functional.mse_loss(pred, noise)


def run_val():
    transformer.eval()
    vl, vn = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(val_idx), a.batch):
            img, msk, e = batch(val_idx[s:s + a.batch])
            noise = torch.randn_like(img)
            t = torch.randint(0, 1000, (len(img),), device=img.device)
            vl += float(forward_loss(sched_train.add_noise(img, noise, t), t, msk, e, noise))
            vn += 1
    transformer.train()
    return vl / vn


step, t_last = 0, time.time()
loss_acc, loss_n = 0.0, 0
while step < a.steps:
    sel = np.random.choice(tr_idx, a.batch, replace=False)
    img, msk, e = batch(sel)
    noise = torch.randn_like(img)
    t = torch.randint(0, 1000, (len(img),), device=img.device)
    noisy = sched_train.add_noise(img, noise, t)
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = forward_loss(noisy, t, msk, e, noise)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    loss_acc += float(loss)
    loss_n += 1
    step += 1
    if step % 50 == 0:
        rec = {"step": step, "loss": round(loss_acc / loss_n, 4), "lr": opt.param_groups[0]["lr"],
               "steps_per_s": round(50 / (time.time() - t_last), 3)}
        t_last = time.time()
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        loss_acc, loss_n = 0.0, 0
    if step % a.val_every == 0 or step == a.steps:
        rec = {"step": step, "val_loss": round(run_val(), 4)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        save_lora(transformer, out / "transformer_lora.pth")
        if a.snapshot_every and step % a.snapshot_every == 0 and step != a.steps:
            save_lora(transformer, out / f"transformer_lora_step{step}.pth")
        if a.lora_controlnet:
            save_lora(controlnet, out / "controlnet_lora.pth")
print("done", out)
