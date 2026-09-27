#!/usr/bin/env python
"""Post-processing ablation for the HoVer-NeXt port (sanity gap: our mPQ .358 vs paper .477).

Runs the model once on a fold subset, then scores three decodings of the SAME raw maps to
isolate whether the gap is in the maps or in the decoding:

A) port      per-class fg/seed thresholds + one-hot majority typing (hover_next_inference)
B) valid     flat fg=0.7/seed=0.3 + softmax-sum typing (hover_next_train validation — the path
             that produced the paper number)
C) valid+pp  B + hole fill + PANNUKE size filters

    python scripts/hn_postproc_ablation.py --weights runs/hovernext_t/split1 --fold 3 --n 400
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import find_objects

from nucseg.data.external import LazyGTChannels
from nucseg.data.pannuke import PanNukeFold
from nucseg.hovernext.engine import (MAX_HOLE_SIZE, MAX_THRESHS_PANNUKE, MIN_THRESHS_PANNUKE,
                                     _faster_instance_seg, _post_per_class, _post_proc_inst,
                                     _remove_obj_cls, build_model)
from nucseg.metrics.pannuke_eval import evaluate, format_summary

p = argparse.ArgumentParser()
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--n", type=int, default=400)
p.add_argument("--bs", type=int, default=16)
a = p.parse_args()

hn = build_model(a.weights)
m = hn.model.cuda().eval()
f = PanNukeFold(a.fold)
idx = np.linspace(0, len(f.images) - 1, a.n).astype(int)
gt_inst, gt_type = np.asarray(f.inst)[idx], np.asarray(f.type)[idx]
gt_ch = LazyGTChannels(gt_inst, gt_type)
gt_tis = np.asarray(f.tissue)[idx]

maps = []  # (inst3, cls6) per selected image, in idx order
with torch.inference_mode():
    for s in range(0, len(idx), a.bs):
        x = torch.from_numpy(np.asarray(f.images[idx[s:s + a.bs]], np.float32) / 255.0)
        x = x.permute(0, 3, 1, 2).cuda()
        with torch.autocast("cuda", torch.float16):
            out = m(x).float()
        for b in range(len(x)):
            maps.append((out[b, 2:5].softmax(0).cpu().numpy(), out[b, 5:].softmax(0).cpu().numpy()))


def onehot_sem(cls6):
    am = cls6[1:].argmax(0)
    return np.stack([am == c for c in range(5)])


def decode_valid(inst3, cls6, postproc):
    lab, _ = _faster_instance_seg(inst3[:2], np.ones((1, *inst3.shape[1:]), bool),
                                  np.array([0.7]), np.array([0.3]))
    ct = {}
    probs = cls6[1:]
    for j, sl in enumerate(find_objects(lab)):
        if not sl:
            continue
        sel = lab[sl] == (j + 1)
        ct[j + 1] = int(np.sum(probs[(slice(None), *sl)][:, sel], axis=1).argmax()) + 1
    if postproc:
        lab = _post_proc_inst(lab, MAX_HOLE_SIZE)
        lab, ct = _remove_obj_cls(lab, ct, MIN_THRESHS_PANNUKE, MAX_THRESHS_PANNUKE)
    typ = np.zeros(lab.shape, np.uint8)
    for k, v in ct.items():
        typ[lab == k] = v
    return lab, typ


def run(name, decode):
    insts, typs = [], []
    for inst3, cls6 in maps:
        inst, typ = decode(inst3, cls6)
        insts.append(inst)
        typs.append(typ)
    r = evaluate(gt_ch, gt_inst, gt_type, gt_tis, np.stack(insts), np.stack(typs), workers=8)
    print(f"\n===== {name} =====")
    print(format_summary(r["summary"]))


run("A_port", lambda i3, c6: _post_per_class((i3, c6, hn.fg, hn.seed)))
run("B_valid", lambda i3, c6: decode_valid(i3, c6, postproc=False))
run("C_valid_pp", lambda i3, c6: decode_valid(i3, c6, postproc=True))
