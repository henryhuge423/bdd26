#!/usr/bin/env python
"""PixCell-256 Cell-ControlNet sampling smoke test (pillar B).

Condition the released ControlNet on REAL PanNuke instance masks and check whether the 20x-trained
generator produces usable 40x H&E appearance from 40x PanNuke context images (decision input:
LoRA-adapt to 40x vs 2x-downsampled generation). For each test patch we render

    [real 40x | generated (seed A) | generated (seed B) | mask]

plus UNI2-h cosine(gen, real) as a crude appearance-similarity sanity number.

    python scripts/pixcell_sample.py --fold 1 --n 6 --out runs/pixcell/smoke
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=1)
p.add_argument("--n", type=int, default=6)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--guidance", type=float, default=2.5)
p.add_argument("--out", type=Path, default=Path("runs/pixcell/smoke"))
a = p.parse_args()

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
DEV = "cuda"


def load_uni2h():
    import timm
    import torchvision.transforms as T

    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    m = timm.create_model("vit_giant_patch14_224", pretrained=False, **kw)
    sd = torch.load(WEIGHTS / "MahmoodLab/UNI2-h/pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith("head")], f"missing {missing[:5]}"
    m.eval().to(DEV)
    # UNI2-h cfg (its HF config.json): resize 224 (crop_pct 1), ImageNet mean/std
    tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return m, tf


@torch.inference_mode()
def uni_embed(model, tf, img: Image.Image) -> torch.Tensor:
    return model(tf(img).unsqueeze(0).to(DEV)).unsqueeze(1)  # (1, 1, D)


def main():
    import sys

    from diffusers import DiffusionPipeline

    # custom_pipeline loads only pipeline.py; its sibling modules must be importable
    sys.path.insert(0, str(ROOT / "third_party/pixcell_pipeline"))
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
    pipe = DiffusionPipeline.from_pretrained(
        WEIGHTS / "StonyBrook-CVLab/PixCell-256-Cell-ControlNet",
        custom_pipeline=str(ROOT / "third_party/pixcell_pipeline"), trust_remote_code=True,
        torch_dtype=torch.float32).to(DEV)  # pipeline hardcodes fp32 controlnet input
    pipe.set_progress_bar_config(disable=True)

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
            g = torch.Generator(DEV).manual_seed(seed)
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
    Image.fromarray(canvas).save(a.out / f"smoke_fold{a.fold}.png")
    (a.out / f"smoke_fold{a.fold}.json").write_text(json.dumps(stats, indent=1))
    print("wrote", a.out / f"smoke_fold{a.fold}.png")


if __name__ == "__main__":
    main()
