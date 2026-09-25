#!/usr/bin/env python
"""Quantitative generator evaluation on REAL layouts (pillar B; before/after LoRA adaptation).

Stratified fold patches (Dead-rich / large-nucleus-rich / random) are re-generated from their GT
binary masks (context = UNI2-h embedding of the real patch, as in LoRA training). Each generated
image is scored with

  - per-object rendering: an out-of-fold CellViT-UNI detector (split 2 for fold 1: fold 1 was its
    validation fold, never trained on) predicts instances on the generated image; coverage =
    max over predictions of |pred cap obj| / |obj|. rendered >= .5, partial .2-.5, dropped < .2;
    rates stratified by object size and class;
  - phantom nuclei: predicted instances covered < .2 by the union of mask objects (px fraction);
  - texture: Laplacian variance ratio gen/real (global and inside nuclei);
  - colour: OD-space channel-mean L1 distance gen vs real;
  - UNI2-h cosine(gen, real).

    python scripts/pixcell_eval.py --fold 1 --detector runs/cellvit_uni/split2/final.pth \
        [--lora runs/pixcell/lora_fold1/transformer_lora.pth] --out runs/pixcell/eval_base
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from nucseg.cellvit.engine import _forward_probs, build_model
from nucseg.data.pannuke import PanNukeFold
from nucseg.hovernet.engine import _post
from nucseg.pixcell import load_lora, load_pipeline, load_uni2h, uni_embed, wrap_lora

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=1)
p.add_argument("--n-dead", type=int, default=100)
p.add_argument("--n-large", type=int, default=50)
p.add_argument("--n-random", type=int, default=50)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--guidance", type=float, default=2.5)
p.add_argument("--detector", type=Path, required=True, help="CellViT final.pth, out-of-fold for --fold")
p.add_argument("--lora", type=Path, default=None)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

DEAD = 4
STRATA = [(0, 100), (100, 200), (200, 10 ** 9)]


def od_mean(img_u8):
    od = -np.log((img_u8.astype(np.float32) + 1.0) / 256.0)
    return od.reshape(-1, 3).mean(0)


def lap_var(gray):
    return cv2.Laplacian(np.ascontiguousarray(gray, np.float32), cv2.CV_32F).var()


def rate(rows):
    if not rows:
        return {"n": 0}
    cov = np.array([r[1] for r in rows])
    return {"n": len(rows), "rendered": round(float((cov >= 0.5).mean()), 4),
            "partial": round(float(((cov >= 0.2) & (cov < 0.5)).mean()), 4),
            "dropped": round(float((cov < 0.2).mean()), 4),
            "median_area": int(np.median([r[0] for r in rows]))}


def main():
    a.out.mkdir(parents=True, exist_ok=True)
    f = PanNukeFold(a.fold)
    n = len(f)

    # stratified patch pick (Dead-rich, large-nucleus-rich, random; disjoint)
    info = []
    for j in range(n):
        inst, typ = np.asarray(f.inst[j]), np.asarray(f.type[j])
        ids = np.unique(inst)[1:]
        areas = np.bincount(inst.ravel(), minlength=int(inst.max()) + 1)[ids]
        dead = int(sum(np.bincount(typ[inst == k], minlength=8).argmax() == DEAD for k in ids))
        info.append((j, dead, int((areas > 200).sum())))
    dead_rich = [j for j, *_ in sorted(info, key=lambda r: -r[1])[:a.n_dead]]
    large_rich = [j for j, _, nl in sorted(info, key=lambda r: -r[2]) if j not in dead_rich][:a.n_large]
    pool_r = [j for j, *_ in info if j not in dead_rich and j not in large_rich]
    idx = dead_rich + large_rich + list(np.random.default_rng(0).choice(
        pool_r, min(a.n_random, len(pool_r)), replace=False))

    pipe = load_pipeline()
    if a.lora is not None:
        pipe.transformer = wrap_lora(pipe.transformer, a.rank)
        load_lora(pipe.transformer, a.lora)
    uni, tf = load_uni2h()
    uncond = pipe.get_unconditional_embedding(1)

    gens, reals, masks = [], [], []
    for k, j in enumerate(idx):
        img = Image.fromarray(np.asarray(f.images[j]))
        inst = np.asarray(f.inst[j])
        mask_rgb = np.where(inst > 0, 255, 0).astype(np.uint8)[..., None].repeat(3, -1)
        emb = uni_embed(uni, tf, img)
        g = torch.Generator("cuda").manual_seed(0)
        out = pipe(uni_embeds=emb, controlnet_input=mask_rgb, negative_uni_embeds=uncond,
                   guidance_scale=a.guidance, num_inference_steps=a.steps, generator=g).images[0]
        gens.append(np.asarray(out))
        reals.append(np.asarray(f.images[j]))
        masks.append(inst)
        if (k + 1) % 25 == 0:
            print(f"[gen] {k + 1}/{len(idx)}", flush=True)

    # ------------------------------------------------------------------ OOF detector
    model = build_model(pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(a.detector, map_location="cpu", weights_only=False)["model"])
    preds = []
    with torch.no_grad(), Pool(8) as pool:
        for s in range(0, len(gens), a.batch):
            x = torch.from_numpy(np.stack(gens[s:s + a.batch])).cuda()
            pr = _forward_probs(model, x)
            tp = pr["tp"].argmax(-1, keepdim=True).float()
            maps = torch.cat([tp, pr["np"][..., 1:], pr["hv"]], -1).cpu().numpy()
            preds += [r[0] for r in pool.map(_post, ((m,) for m in maps), chunksize=2)]

    # ------------------------------------------------------------------ per-object + phantom scoring
    rows, phantom_px, pred_px = [], 0, 0
    for gen, real, inst, pred, j in zip(gens, reals, masks, preds, idx):
        typ = np.asarray(f.type[j])
        union = inst > 0
        pred_ids = np.unique(pred)[1:]
        for pid in pred_ids:
            pm = pred == pid
            pred_px += int(pm.sum())
            if (pm & union).sum() / max(pm.sum(), 1) < 0.2:
                phantom_px += int(pm.sum())
        cov_by_obj = {}
        for k in np.unique(inst)[1:]:
            obj = inst == k
            cov = 0.0
            for pid in pred_ids:
                inter = (pred == pid) & obj
                if inter.sum():
                    cov = max(cov, inter.sum() / obj.sum())
            rows.append((int(obj.sum()), round(float(cov), 4),
                         int(np.bincount(typ[obj], minlength=8).argmax()), int(j)))
    with open(a.out / "objects.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["area", "coverage", "class", "img"])
        w.writerows(rows)

    summary = {"lora": str(a.lora), "n_images": len(idx), "n_objects": len(rows),
               "phantom_px_frac": round(phantom_px / max(pred_px, 1), 4)}
    for lo, hi in STRATA:
        summary[f"area_{lo}_" + (str(hi) if hi < 10 ** 9 else "up")] = rate(
            [r for r in rows if lo <= r[0] < hi])
    summary["Dead_all"] = rate([r for r in rows if r[2] == DEAD])
    summary["other_all"] = rate([r for r in rows if r[2] != DEAD])
    for lo, hi in STRATA:
        summary[f"Dead_{lo}_" + (str(hi) if hi < 10 ** 9 else "up")] = rate(
            [r for r in rows if r[2] == DEAD and lo <= r[0] < hi])

    # ------------------------------------------------------------------ texture / colour / embedding
    laps, laps_in, od_d, sims = [], [], [], []
    for gen, real, inst in zip(gens, reals, masks):
        m = inst > 0
        laps.append(lap_var(gen.mean(-1)) / max(lap_var(real.mean(-1)), 1e-6))
        laps_in.append(lap_var((gen * m[..., None]).mean(-1)) / max(lap_var((real * m[..., None]).mean(-1)), 1e-6))
        od_d.append(float(np.abs(od_mean(gen) - od_mean(real)).mean()))
        sims.append(float(torch.nn.functional.cosine_similarity(
            uni_embed(uni, tf, Image.fromarray(gen)).flatten().float(),
            uni_embed(uni, tf, Image.fromarray(real)).flatten().float(), dim=0)))
    summary["laplacian_ratio_global"] = round(float(np.median(laps)), 4)
    summary["laplacian_ratio_in_nuclei"] = round(float(np.median(laps_in)), 4)
    summary["od_mean_l1"] = round(float(np.mean(od_d)), 4)
    summary["uni_cos"] = round(float(np.mean(sims)), 4)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))

    montage_rows = [np.concatenate([reals[j], gens[j],
                                    np.where(masks[j] > 0, 255, 0).astype(np.uint8)[..., None].repeat(3, -1)],
                                   axis=1) for j in range(min(12, len(idx)))]
    Image.fromarray(np.concatenate(montage_rows, axis=0)).save(a.out / "montage.png")
    print(json.dumps(summary, indent=1))
    print("wrote", a.out / "summary.json")


if __name__ == "__main__":
    main()
