#!/usr/bin/env python
"""Failure-driven layout sampling + batched PixCell synthesis + teacher labelling (pillar B, FTCS).

Round-1 layout arm (from failure mining): insert ISOLATED small nucleus MASKS into the empty stroma
of real TRAIN-fold layouts (Dead-weighted donors, 8 px clearance, mask domain) and generate images
with the (optionally LoRA-adapted) PixCell ControlNet — batched for throughput. Context embeddings
come from random real train-fold patches (stain/tissue diversity; cached by train_pixcell_lora).

Labels: synthetic images are labelled by an OUT-OF-FOLD teacher (split-2 CellViT-UNI; fold 1 = its
val fold, never trained). A layout object keeps a label iff the teacher detects an instance covering
>= .4 of it AND assigns a type with probability >= --type-thresh; unconfirmed objects are removed
from the label map (they still contribute image realism). Samples whose confirmed-object pixel
fraction < --keep-frac are dropped entirely. CAVEAT (documented in findings): the teacher's own
blind spots bias labels towards teacher-consistency; detection (NP/HV) supervision on isolated
small nuclei is the primary signal, type labels secondary.

    python scripts/synth_pannuke.py --fold 1 --n 3000 --out runs/pixcell/synth_fold1 \
        --lora runs/pixcell/lora_fold1/transformer_lora.pth --teacher runs/cellvit_uni/split2/final.pth
    python scripts/synth_pannuke.py --verify 4 ...   # batched sampler vs the reference pipeline
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from nucseg.augment.copy_paste import DEAD, NucleusBank
from nucseg.cellvit.engine import _forward_probs, _post, build_model
from nucseg.data.pannuke import PanNukeFold
from nucseg.pixcell import encode_condition, load_lora, load_pipeline, uni_embed, wrap_lora
from nucseg.text.conch_prior import instance_mean_probs

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, required=True, help="train fold whose layouts/embeds are used")
p.add_argument("--cache", type=Path, default=None, help="cache_fold{k}.npz from train_pixcell_lora")
p.add_argument("--n", type=int, default=3000)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--lora", type=Path, default=None)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--teacher", type=Path, default=None)
p.add_argument("--ins", type=float, default=3.0, help="Poisson mean inserted objects per layout")
p.add_argument("--dead-w", type=float, default=0.4)
p.add_argument("--area", type=int, nargs=2, default=[50, 400])
p.add_argument("--clearance", type=int, default=8)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--guidance", type=float, default=2.5)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--amp", action="store_true", help="bf16 sampling (faster; default fp32)")
p.add_argument("--type-thresh", type=float, default=0.6)
p.add_argument("--keep-frac", type=float, default=0.7)
p.add_argument("--random-ctx", action="store_true",
               help="context embedding from a random patch instead of the layout's own base patch "
                    "(the 2026-09-26 review showed random contexts push generation off-manifold: "
                    "black-crushed nuclei, bimodal per-patch colour)")
p.add_argument("--verify", type=int, default=0, help="compare batched vs pipeline on N patches")
p.add_argument("--seed", type=int, default=19)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
torch.manual_seed(a.seed)


# ------------------------------------------------------------------ layout sampler (mask domain)
def perturb_layout(inst, bank, base_img):
    """Insert isolated donor NUCLEUS MASKS into empty stroma of a real layout (same placement rules
    as copy_paste: clearance to all nuclei, patch border, Otsu-dark unannotated material)."""
    H, W = inst.shape
    inst = inst.copy()
    occupied = inst > 0
    kern = np.ones((2 * a.clearance + 1, 2 * a.clearance + 1), np.uint8)
    blocked = cv2.dilate(occupied.astype(np.uint8), kern)
    gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
    otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gray < otsu_thr) & ~blocked.astype(bool)
    free = np.argwhere(~blocked.astype(bool))
    if not len(free):
        return inst, []
    rng.shuffle(free)
    n_ins = max(1, int(rng.poisson(a.ins)))
    next_id, inserted, pi = int(inst.max()) + 1, [], 0
    weights = np.full(8, (1 - a.dead_w) / 4.0)
    weights[DEAD] = a.dead_w
    weights[0] = 0.0
    for _ in range(n_ins):
        cls = int(rng.choice(8, p=weights / weights.sum()))
        cand = bank.by_class[cls][(bank.rows.area[bank.by_class[cls]] >= a.area[0])
                                  & (bank.rows.area[bank.by_class[cls]] <= a.area[1])]
        if not len(cand):
            continue
        r = bank.rows[int(rng.choice(cand))]
        src = np.asarray(bank.folds[r.fi].inst[r.img])
        cmask = (src == r.id)[r.y0:r.y1, r.x0:r.x1]
        h, w = cmask.shape
        for _try in range(40):
            if pi >= len(free):
                pi = 0
            cy, cx = free[pi]
            pi += 1
            ty0, tx0 = int(cy) - h // 2, int(cx) - w // 2
            if ty0 < 4 or tx0 < 4 or ty0 + h > H - 4 or tx0 + w > W - 4:
                continue
            if (cmask & blocked[ty0:ty0 + h, tx0:tx0 + w].astype(bool)).any():
                continue
            if (cmask & dark[ty0:ty0 + h, tx0:tx0 + w]).sum() > 0.3 * cmask.sum():
                continue
            region = np.zeros((H, W), bool)
            region[ty0:ty0 + h, tx0:tx0 + w] = cmask
            inst[region] = next_id
            inserted.append(next_id)
            next_id += 1
            occupied |= region
            blocked = cv2.dilate(occupied.astype(np.uint8), kern)
            break
    return inst, inserted


# ------------------------------------------------------------------ batched sampling core
@torch.no_grad()
def batched_generate(pipe, embs, mask_rgbs, steps, guidance, seeds, amp=False):
    """PixCellControlNetPipeline semantics, batched. CFG uncond rows get ZERO controlnet
    contribution (identical to the pipeline's controlnet_outputs=None uncond pass, because the
    block injection is additive). embs (B,1,1536) fp32; mask_rgbs list of HWC uint8."""
    dev = embs.device
    B = len(embs)
    sch = pipe.scheduler
    sch.set_timesteps(steps, device=dev)
    raw = pipe.transformer.caption_projection.uncond_embedding.detach().to(dev)
    uncond = raw[None].expand(B, -1, -1) if raw.ndim == 2 else raw.expand(B, -1, -1)

    mlat = torch.cat([encode_condition(pipe, m) for m in mask_rgbs])  # (B,16,32,32)
    g = torch.Generator(dev)
    noise = torch.stack([torch.randn(pipe.transformer.config.in_channels, 32, 32, device=dev,
                                     generator=g.manual_seed(int(s))) for s in seeds])
    latents = noise * sch.init_noise_sigma
    for t in sch.timesteps:
        lat_in = torch.cat([latents, latents], 0)
        tt = t.expand(2 * B)
        emb_all = torch.cat([uncond, embs], 0)
        cn_cond = pipe.controlnet(hidden_states=latents, conditioning=mlat,
                                  encoder_hidden_states=embs, timestep=t.expand(B),
                                  return_dict=False)[0]
        cn_all = [torch.cat([torch.zeros_like(b), b], 0) for b in cn_cond]
        pred = pipe.transformer(sch.scale_model_input(lat_in, t), encoder_hidden_states=emb_all,
                                controlnet_outputs=cn_all, timestep=tt, return_dict=False)[0]
        if pred.shape[1] == 2 * latents.shape[1]:
            pred = pred.chunk(2, dim=1)[0]
        u, c = pred.chunk(2, dim=0)
        latents = sch.step(u + guidance * (c - u), t, latents, return_dict=False)[0]
    img = pipe.vae.decode((latents / pipe.vae.config.scaling_factor)
                          + getattr(pipe.vae.config, "shift_factor", 0), return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
    return (img * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def verify(pipe, f, n):
    """Batched sampler must reproduce the reference pipeline (fp32, same seeds)."""
    from PIL import Image

    idx = list(rng.choice(len(f), n, replace=False))
    from nucseg.pixcell import load_uni2h
    uni, tf = load_uni2h()
    embs = torch.cat([uni_embed(uni, tf, Image.fromarray(np.asarray(f.images[j]))) for j in idx])
    masks = [np.where(np.asarray(f.inst[j]) > 0, 255, 0).astype(np.uint8)[..., None].repeat(3, -1)
             for j in idx]
    seeds = [10 ** 6 + i for i in range(n)]
    mine = batched_generate(pipe, embs, masks, a.steps, a.guidance, seeds)
    for b, j in enumerate(idx):
        g = torch.Generator("cuda").manual_seed(seeds[b])
        ref = pipe(uni_embeds=embs[b:b + 1], controlnet_input=masks[b],
                   negative_uni_embeds=pipe.get_unconditional_embedding(1),
                   guidance_scale=a.guidance, num_inference_steps=a.steps, generator=g).images[0]
        ref = np.asarray(ref)
        diff = float(np.abs(ref.astype(int) - mine[b].astype(int)).mean())
        print(f"[verify] img {j}: mean|ref-batched| = {diff:.3f}/255")
        assert diff < 2.0, "batched sampler diverges from the pipeline"
    print("[verify] batched sampler matches the reference pipeline")


def main():
    a.out.mkdir(parents=True, exist_ok=True)
    f = PanNukeFold(a.fold)
    bank = NucleusBank([f])
    pipe = load_pipeline()
    if a.lora is not None:
        pipe.transformer = wrap_lora(pipe.transformer, a.rank)
        load_lora(pipe.transformer, a.lora)
        print(f"[lora] {a.lora}")

    if a.verify:
        verify(pipe, f, a.verify)
        return

    cache = a.cache or (Path("runs/pixcell") / f"lora_fold{a.fold}" / f"cache_fold{a.fold}.npz")
    emb_all = torch.from_numpy(np.load(cache)["emb"].astype(np.float32)).cuda()

    layouts, inserted_ids, ctx_idx, seeds, base_idx = [], [], [], [], []
    for _ in range(a.n):
        j = int(rng.integers(len(f)))
        inst, ins_ids = perturb_layout(np.asarray(f.inst[j]).astype(np.int32), bank,
                                       np.asarray(f.images[j]))
        layouts.append(inst)
        inserted_ids.append(ins_ids)
        # PAIRED context (the base patch's own embedding) matches training conditioning; random
        # contexts produce off-manifold appearance (see --random-ctx help)
        ctx_idx.append(int(rng.integers(len(f))) if a.random_ctx else j)
        seeds.append(int(rng.integers(2 ** 31)))
        base_idx.append(j)
    print(f"[layout] {a.n} layouts, {np.mean([len(x) for x in inserted_ids]):.2f} inserted each")

    gens = []
    for s in range(0, a.n, a.batch):
        embs = emb_all[ctx_idx[s:s + a.batch]].unsqueeze(1)
        masks = [np.where(m > 0, 255, 0).astype(np.uint8)[..., None].repeat(3, -1)
                 for m in layouts[s:s + a.batch]]
        if a.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gens.append(batched_generate(pipe, embs, masks, a.steps, a.guidance,
                                             seeds[s:s + a.batch]))
        else:
            gens.append(batched_generate(pipe, embs, masks, a.steps, a.guidance,
                                         seeds[s:s + a.batch]))
        if (s // a.batch) % 10 == 0:
            print(f"[gen] {s}/{a.n}", flush=True)
    gens = np.concatenate(gens)
    np.save(a.out / "images.npy", gens)
    print("[gen] saved", gens.shape)

    # ------------------------------------------------------------------ teacher labelling
    model = build_model(pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(a.teacher, map_location="cpu", weights_only=False)["model"])
    kept_inst = np.zeros(gens.shape[:3], np.int32)
    kept_type = np.zeros_like(kept_inst)
    n_obj_all = n_obj_kept = n_drop_img = 0
    with Pool(8) as pool:
        for s in range(0, a.n, a.batch):
            x = torch.from_numpy(gens[s:s + a.batch]).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pr = _forward_probs(model, x)
            prob = pr["tp"].cpu().numpy()
            tp = pr["tp"].argmax(-1, keepdim=True).float()
            maps = torch.cat([tp, pr["np"][..., 1:], pr["hv"]], -1).cpu().numpy()
            for b, m in enumerate(maps):
                pred, _ = _post((m,))
                pids = np.unique(pred)[1:]
                probs = instance_mean_probs(pred, prob[b], pids) if len(pids) else np.zeros((0, 6))
                img_i = s + b
                inst = layouts[img_i]
                out = np.zeros_like(inst)
                tout = np.zeros_like(inst)
                ids = np.unique(inst)[1:]
                n_obj_all += len(ids)
                for k in ids:
                    obj = inst == k
                    best, best_cov = 0, 0.0
                    for pi, pid in enumerate(pids):
                        inter = (pred == pid) & obj
                        if inter.sum():
                            cov = inter.sum() / obj.sum()
                            if cov > best_cov:
                                best, best_cov = pi, cov
                    if best_cov < 0.4 or probs[best].max() < a.type_thresh:
                        continue  # unconfirmed -> object leaves the label (image keeps it)
                    out[obj] = k
                    tout[obj] = int(probs[best].argmax())
                    n_obj_kept += 1
                if len(ids) and (out > 0).sum() / max((inst > 0).sum(), 1) < a.keep_frac:
                    n_drop_img += 1
                    continue
                kept_inst[img_i] = out
                kept_type[img_i] = tout
            if (s // a.batch) % 20 == 0:
                print(f"[teacher] {s}/{a.n}", flush=True)
    np.save(a.out / "inst.npy", kept_inst.astype(np.uint16))
    np.save(a.out / "type.npy", kept_type.astype(np.uint8))
    np.save(a.out / "base.npy", np.array(base_idx, np.int32))  # tissue label source
    np.save(a.out / "ctx.npy", np.array(ctx_idx, np.int32))
    stats = {"n": a.n, "objects": int(n_obj_all), "objects_kept": int(n_obj_kept),
             "keep_rate": round(n_obj_kept / max(n_obj_all, 1), 4),
             "images_dropped": int(n_drop_img)}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
