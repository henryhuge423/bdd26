#!/usr/bin/env python
"""Pillar A: re-type a segmenter's predicted instances with region-pooled CONCH embeddings.

    python scripts/retype_conch.py --run runs/cellvit_uni/split1 --split 1 [--tta]

Needs <run>/pred_fold{val,test}[_tta].npz from scripts/predict_cellvit.py (per-instance type probs).
Every knob (fusion weight alpha, zero-shot temperature, class biases) is tuned on the VALIDATION fold
with the exact fast mPQ (nucseg.metrics.fast_retype); the test fold is scored once per method.

Methods (type of each predicted instance):
  seg_vote       pipeline output (pixel majority vote of the TP map)             -- baseline
  seg            argmax of the instance-mean TP probabilities
  seg+bias       seg with per-class log-biases tuned on val                      -- recalibration control
  seg*zs         log-linear fusion with zero-shot CONCH (text prototypes)       -- no training
  seg*lin        fusion with a linear head on CONCH embeddings (train-fold GT nuclei, balanced CE)
  lin            the CONCH linear head alone
  seg*lin+bias   fusion + tuned class biases
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold, split_folds
from nucseg.metrics.fast_retype import RetypeEvaluator
from nucseg.metrics.pannuke_eval import evaluate, format_summary, instance_classes, save_report
from nucseg.text.conch_prior import ConchNucleusPrior, instance_table
from nucseg.text.retype import paint_types, softmax, train_anchored_linear

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--split", type=int, required=True)
p.add_argument("--tta", action="store_true")
p.add_argument("--radius", type=float, default=32)
p.add_argument("--protos", default="weights/text_protos/conch_v1.pt")
p.add_argument("--full-eval", nargs="*", default=["seg+bias", "seg*zs", "seg*lin", "seg*lin+bias"],
               help="methods to also score with the full evaluator on test (reports saved)")
a = p.parse_args()

tr, va, te = split_folds(a.split)
tag = "_tta" if a.tta else ""
R = int(a.radius)
prior = None


def get_prior():
    global prior
    if prior is None:
        prior = ConchNucleusPrior(a.protos, radius=a.radius)
    return prior


def gt_embeddings(k):
    path = Path(f"runs/conch_prior/gt_fold{k}_r{R}.npz")
    if not path.exists():
        f = PanNukeFold(k)
        img, ids, emb = get_prior().embed(f.images, f.inst)
        typ = np.array([np.bincount(np.asarray(f.type[j])[np.asarray(f.inst[j]) == i]).argmax() for j, i in zip(img, ids)])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, img=img, ids=ids, emb=emb.astype(np.float16), type=typ)
    d = np.load(path)
    return d["emb"].astype(np.float32), d["type"]


def pred_embeddings(k, pred):
    path = a.run / f"conch_fold{k}{tag}_r{R}.npz"
    if not path.exists():
        img, ids, emb = get_prior().embed(PanNukeFold(k).images, pred["inst"])
        np.savez_compressed(path, img=img, ids=ids, emb=emb.astype(np.float16))
    d = np.load(path)
    assert np.array_equal(d["img"], pred["inst_img"]) and np.array_equal(d["ids"], pred["inst_id"])
    return d["emb"].astype(np.float32)


def load_split(k):
    pred = dict(np.load(a.run / f"pred_fold{k}{tag}.npz"))
    f = PanNukeFold(k)
    ev = RetypeEvaluator(f.gt_channels, pred["inst"], pred["inst_img"], pred["inst_id"], f.tissue)
    vote = np.concatenate([instance_classes(pred["inst"][j], pred["type"][j])[1] for j in range(len(f))])
    return dict(pred=pred, f=f, ev=ev, vote=vote, emb=pred_embeddings(k, pred))


def fused_logits(p_seg, p_txt, alpha, eps=1e-6):
    ps = p_seg[:, 1:] / np.maximum(p_seg[:, 1:].sum(1, keepdims=True), eps)
    z = (1 - alpha) * np.log(ps + eps)
    return z if p_txt is None else z + alpha * np.log(p_txt + eps)


def tune_bias(ev, z, rounds=3, grid=np.linspace(-2, 2, 21)):
    b = np.zeros(z.shape[1])
    best = ev.mpq((z + b).argmax(1) + 1)[0]
    for _ in range(rounds):
        for c in range(z.shape[1]):
            for v in grid:
                bb = b.copy()
                bb[c] = v
                m = ev.mpq((z + bb).argmax(1) + 1)[0]
                if m > best + 1e-9:
                    best, b = m, bb
    return b, best


V, T = load_split(va), load_split(te)
P = torch.load(a.protos)
text = P["desc"][1:].numpy()

# experts on CONCH embeddings
Xtr, ytr = gt_embeddings(tr)
W, bl = train_anchored_linear(Xtr, ytr, None, balanced=True)
for S in (V, T):
    S["lin"] = softmax(S["emb"] @ W.T + bl)
    S["cos"] = S["emb"] @ text.T
mu = V["cos"].mean(0, keepdims=True)  # zero-shot calibration from the val fold only

alphas = np.round(np.linspace(0, 1, 21), 2)
results, chosen = {}, {}


def record(name, cls_v, cls_t, **params):
    mv, pcv = V["ev"].mpq(cls_v)
    mt, pct = T["ev"].mpq(cls_t)
    results[name] = {"val_mPQ": mv, "test_mPQ": mt, "test_per_class": dict(zip(CLASS_NAMES, map(float, pct))),
                     "val_per_class": dict(zip(CLASS_NAMES, map(float, pcv))), **params}
    chosen[name] = cls_t
    print(f"{name:14s} val {mv:.4f}  test {mt:.4f}  Dead {pct[3]:.4f}  "
          + "  ".join(f"{k}={v}" for k, v in params.items()))


record("seg_vote", V["vote"], T["vote"])
zv, zt = fused_logits(V["pred"]["inst_prob"], None, 0), fused_logits(T["pred"]["inst_prob"], None, 0)
record("seg", zv.argmax(1) + 1, zt.argmax(1) + 1)
b, _ = tune_bias(V["ev"], zv)
record("seg+bias", (zv + b).argmax(1) + 1, (zt + b).argmax(1) + 1, bias=np.round(b, 2).tolist())

best = (-1, None, None)
for tau in (10, 25, 50, 100):
    pv = softmax(tau * (V["cos"] - mu))
    for al in alphas:
        m = V["ev"].mpq(fused_logits(V["pred"]["inst_prob"], pv, al).argmax(1) + 1)[0]
        if m > best[0]:
            best = (m, tau, al)
_, tau, al = best
record("seg*zs", fused_logits(V["pred"]["inst_prob"], softmax(tau * (V["cos"] - mu)), al).argmax(1) + 1,
       fused_logits(T["pred"]["inst_prob"], softmax(tau * (T["cos"] - mu)), al).argmax(1) + 1,
       alpha=float(al), tau=tau)

al = max(alphas, key=lambda x: V["ev"].mpq(fused_logits(V["pred"]["inst_prob"], V["lin"], x).argmax(1) + 1)[0])
zv, zt = fused_logits(V["pred"]["inst_prob"], V["lin"], al), fused_logits(T["pred"]["inst_prob"], T["lin"], al)
record("seg*lin", zv.argmax(1) + 1, zt.argmax(1) + 1, alpha=float(al))
record("lin", V["lin"].argmax(1) + 1, T["lin"].argmax(1) + 1)
b, _ = tune_bias(V["ev"], zv)
record("seg*lin+bias", (zv + b).argmax(1) + 1, (zt + b).argmax(1) + 1, alpha=float(al), bias=np.round(b, 2).tolist())

out = a.run / f"retype{tag}_r{R}"
out.mkdir(exist_ok=True)
(out / "results.json").write_text(json.dumps({"split": a.split, "val_fold": va, "test_fold": te, "tta": a.tta,
                                              "radius": a.radius, "methods": results}, indent=2))
f, pred = T["f"], T["pred"]
for name in a.full_eval:
    typ = paint_types(pred["inst"], pred["inst_img"], pred["inst_id"], chosen[name])
    res = evaluate(f.gt_channels, f.inst, f.type, f.tissue, pred["inst"], typ)
    save_report(res, out / f"eval_test_{name.replace('*', 'x').replace('+', '_')}")
    assert abs(res["summary"]["official"]["mPQ"] - results[name]["test_mPQ"]) < 1e-9
    print(f"--- {name} (full eval, test fold {te})\n" + format_summary(res["summary"]))
print("->", out)
