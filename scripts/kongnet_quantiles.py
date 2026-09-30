#!/usr/bin/env python
"""One-off: per-channel prob quantiles of KongNet on a few train-fold patches.
Resolves whether heads 2-5 channels 0/1 are dead or just lower-scale than 0.5."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/tmp/cgf2604/repos/KongNet_Inference_Main")
from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from scripts.kongnet_eval import build_model, fold_probs  # noqa: E402

repo = Path("/tmp/cgf2604/repos/KongNet_Inference_Main")
model = build_model(repo, "cuda")
ckpt = torch.load("/tmp/cgf2604/weights_a1/kongnet/KongNet_PanNuke_1.pth",
                  map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"], strict=True)
del ckpt

fold = PanNukeFold(1)
idx = np.linspace(0, len(fold) - 1, 4).astype(int)
probs = fold_probs(model, np.asarray(fold.images[idx]), "cuda", 4).astype(np.float32)
names = ["h0c0", "h0c1", "h0c2/hm", "neo_seg", "neo_ct", "neo_hm", "inf_seg", "inf_ct",
         "inf_hm", "con_seg", "con_ct", "con_hm", "dead_seg", "dead_ct", "dead_hm",
         "epi_seg", "epi_ct", "epi_hm"]
print(f"{'ch':>8} {'p50':>7} {'p90':>7} {'p99':>7} {'p999':>7} {'max':>7}")
for c in range(18):
    v = probs[..., c].ravel()
    print(f"{names[c]:>8} {np.percentile(v, 50):7.4f} {np.percentile(v, 90):7.4f} "
          f"{np.percentile(v, 99):7.4f} {np.percentile(v, 99.9):7.4f} {v.max():7.4f}")
