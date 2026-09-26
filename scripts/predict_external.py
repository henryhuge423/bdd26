#!/usr/bin/env python
"""Predict (+ evaluate) a PanNuke-trained checkpoint on a converted external dataset (pillar C).

    python scripts/predict_external.py --model cellvit --run runs/cellvit_uni/split1 --data conic [--tta]
    python scripts/predict_external.py --model hovernet --ckpt runs/hovernet/split1/final.pth --data monusac

Zero-shot cross-domain: no external image is ever seen in PanNuke training (CoNIC's PanNuke-derived
patches are excluded at conversion). Report lands in <run>/eval_ext_{data}[_tta].
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from nucseg.data.external import ExternalSet
from nucseg.metrics.pannuke_eval import evaluate, format_summary, save_report

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["cellvit", "hovernet"], default="cellvit")
p.add_argument("--run", type=Path, help="cellvit run dir (final.pth)")
p.add_argument("--ckpt", type=Path, help="hovernet checkpoint (defaults to <run>/final.pth)")
p.add_argument("--data", required=True, help="external dataset name, e.g. conic / monusac")
p.add_argument("--tta", action="store_true")
p.add_argument("--bs", type=int, default=32)
a = p.parse_args()

data_path = Path(a.data)
if data_path.parent != Path("."):
    ds = ExternalSet(data_path.name, root=data_path.parent)
    a.data = data_path.name  # report dir name: keep it flat even when --data is a path
else:
    ds = ExternalSet(a.data)
if a.model == "cellvit":
    from nucseg.cellvit.engine import build_model, predict_fold
    model = build_model(pretrained=False).cuda()
    model.load_state_dict(torch.load(a.run / "final.pth", map_location="cpu", weights_only=False)["model"])
    inst, typ, _ = predict_fold(model, ds, tta=a.tta, batch_size=a.bs)
else:
    from nucseg.hovernet.engine import build_model, predict_fold
    ckpt = a.ckpt or a.run / "final.pth"
    model = build_model(freeze=False).cuda()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    inst, typ = predict_fold(model, ds, tta=a.tta, batch_size=a.bs)

out = (a.run or a.ckpt.parent) / f"eval_ext_{a.data}{'_tta' if a.tta else ''}"
np.savez_compressed(out / "pred.npz", inst=inst.astype(np.int32), type=typ)
res = evaluate(ds.gt_channels, ds.inst, ds.type, ds.tissue, inst, typ,
               tissue_names=ds.meta["tissue_names"])
save_report(res, out)
print(a.data, "\n" + format_summary(res["summary"]))
