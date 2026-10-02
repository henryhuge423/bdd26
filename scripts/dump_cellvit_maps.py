#!/usr/bin/env python
"""Dump CellViT-UNI network outputs of a fold for post-processing sweeps (GPU stage).

    python scripts/dump_cellvit_maps.py --run runs/cellvit_uni/split1 --fold 2 --out data/cache/maps/split1

Writes <out>/maps_fold{k}[_tta].npy, float16 (N, 256, 256, 9): [P_NP(fg), HV_h, HV_v, P_TP(0..5)]
(~1.2 MB per patch, ~3.2 GB per fold) + <out>/tissue_fold{k}[_tta].npy. Written to a .tmp file and
renamed when complete. Keep these large intermediate maps separate from final run artifacts.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from nucseg.cellvit.engine import _forward_probs, _tta_forward, build_model, run_upscale
from nucseg.data.pannuke import PanNukeFold

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--ckpt", default="final.pth")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--tta", action="store_true")
a = p.parse_args()
if run_upscale(a.run) != 1:
    raise SystemExit("dump_cellvit_maps supports native-scale checkpoints only; "
                     "use probe_cellvit_scales.py for explicit cross-scale experiments")

tag = f"fold{a.fold}" + ("_tta" if a.tta else "")
a.out.mkdir(parents=True, exist_ok=True)
final = a.out / f"maps_{tag}.npy"
if final.exists():
    raise SystemExit(f"{final} exists")
model = build_model(pretrained=False).cuda()
model.load_state_dict(torch.load(a.run / a.ckpt, map_location="cpu", weights_only=False)["model"])
model.eval()
f = PanNukeFold(a.fold)
tmp = a.out / f"maps_{tag}.tmp.npy"
shape = (len(f), 256, 256, 9)
tissue = np.zeros((len(f), 19), np.float32)
# stream to disk (no memmap: mapping the whole file consumes the process's address-space budget)
with open(tmp, "wb") as fh, torch.no_grad():
    np.lib.format.write_array_header_1_0(fh, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                                              "fortran_order": False, "shape": shape})
    for s in range(0, len(f), a.batch):
        x = torch.from_numpy(np.asarray(f.images[s:s + a.batch])).cuda()
        pr = _tta_forward(model, x) if a.tta else _forward_probs(model, x)
        fh.write(np.ascontiguousarray(torch.cat([pr["np"][..., 1:], pr["hv"], pr["tp"]], -1).half().cpu().numpy()).tobytes())
        tissue[s:s + len(x)] = pr["tissue"].cpu().numpy()
with open(tmp, "rb") as fh:  # validate header + size without mapping the file
    np.lib.format.read_magic(fh)
    hdr_shape, _, hdr_dtype = np.lib.format.read_array_header_1_0(fh)
    data_start = fh.tell()
assert hdr_shape == shape and hdr_dtype == np.float16
assert tmp.stat().st_size == data_start + int(np.prod(shape)) * 2, "incomplete map file"
np.save(a.out / f"tissue_{tag}.npy", tissue)
os.replace(tmp, final)
print("wrote", final, f"{final.stat().st_size / 1e9:.2f} GB")
