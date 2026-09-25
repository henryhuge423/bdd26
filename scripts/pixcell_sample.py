#!/usr/bin/env python
"""PixCell-256 Cell-ControlNet sampling smoke test / montage (pillar B).

Condition the generator on REAL PanNuke instance masks (optionally through a LoRA adapter trained
by scripts/train_pixcell_lora.py) and render

    [real 40x | generated (seed A) | generated (seed B) | mask]

plus UNI2-h cosine(gen, real) as a crude appearance-similarity sanity number.

    python scripts/pixcell_sample.py --fold 1 --n 6 --out runs/pixcell/smoke [--lora runs/pixcell/lora_fold1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold
from nucseg.pixcell import load_pipeline, load_uni2h, load_lora, uni_embed, wrap_lora

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=1)
p.add_argument("--n", type=int, default=6)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--guidance", type=float, default=2.5)
p.add_argument("--lora", type=Path, default=None, help="transformer_lora.pth from train_pixcell_lora")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--out", type=Path, default=Path("runs/pixcell/smoke"))
a = p.parse_args()


def main():
    a.out.mkdir(parents=True, exist_ok=True)
    f = PanNukeFold(a.fold)

    # pick patches: half with the most Dead GT nuclei, half random (varied tissue)
    inst = f.inst
    dead_cnt = np.array([(np.asarray(f.type[i]) == CLASS_NAMES.index("Dead") + 1).sum() for i in range(len(f))])
    by_dead = np.argsort(-dead_cnt)
    rng = np.random.default_rng(0)
    rand = rng.choice(np.setdiff1d(np.arange(len(f)), by_dead[:a.n]), a.n - a.n // 2, replace=False)
    idx = list(by_dead[:a.n // 2]) + list(rand)

    print("loading UNI2-h + PixCell pipeline ...")
    uni, tf = load_uni2h()
    pipe = load_pipeline()
    if a.lora is not None:
        pipe.transformer = wrap_lora(pipe.transformer, a.rank)
        load_lora(pipe.transformer, a.lora)
        print(f"[lora] loaded {a.lora}")

    rows, stats = [], []
    for k, i in enumerate(idx):
        img = Image.fromarray(np.asarray(f.images[i]))  # uint8 RGB 40x
        inst_i = np.asarray(inst[i])
        mask = np.where(inst_i > 0, 255, 0).astype(np.uint8)
        mask_rgb = np.stack([mask] * 3, -1)
        emb = uni_embed(uni, tf, img)
        uncond = pipe.get_unconditional_embedding(1)
        gens = []
        for seed in (0, 1):
            g = torch.Generator("cuda").manual_seed(seed)
            out = pipe(uni_embeds=emb, controlnet_input=mask_rgb,
                       negative_uni_embeds=uncond, guidance_scale=a.guidance,
                       num_inference_steps=a.steps, generator=g).images[0]
            gens.append(np.asarray(out))
        sim = [float(torch.nn.functional.cosine_similarity(
            uni_embed(uni, tf, Image.fromarray(g)).flatten().float(),
            emb.flatten().float(), dim=0)) for g in gens]
        stats.append({"img": int(i), "tissue": f.tissue[i], "dead_px": int(dead_cnt[i]),
                      "uni_cos": sim})
        rows.append([np.asarray(img), gens[0], gens[1], np.broadcast_to(mask_rgb, gens[0].shape)])
        print(f"[{k+1}/{len(idx)}] img {i} ({f.tissue[i]}) dead_px {dead_cnt[i]} uni_cos {sim}")

    H = 256
    W = sum(r.shape[1] for r in rows[0])
    canvas = np.zeros((H * len(rows) + 20 * (len(rows) - 1), W, 3), np.uint8)
    for r, row in enumerate(rows):
        canvas[r * (H + 20):r * (H + 20) + H] = np.concatenate(row, 1)
    tag = f"smoke_fold{a.fold}" + (f"_lora{a.rank}" if a.lora else "")
    Image.fromarray(canvas).save(a.out / f"{tag}.png")
    (a.out / f"{tag}.json").write_text(json.dumps(stats, indent=1))
    print("wrote", a.out / f"{tag}.png")


if __name__ == "__main__":
    main()
