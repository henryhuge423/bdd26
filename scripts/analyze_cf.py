#!/usr/bin/env python
"""PanNuke-CF analysis (pillar C): arm tables, generator noise floor, paired deltas, rank agreement.

    python scripts/analyze_cf.py --cf-root runs/pixcell/cf_fold3 \
        --evals runs/cellvit_uni:split1,split2 runs/hovernet:split1,split2 \
        --ext conic monusac

Per evaluator (model run + split) it reads <run>/eval_ext_{arm}/ (written by predict_external
--data <cf-root>/<arm>) for the CF arms, plus eval_ext_{ext} for the real cross-domain scores and
test_fold*/ for the in-domain reference. Reports:

1. arm table      official mPQ / bPQ / strict / F_d / Dead PQ / Epith PQ per arm
2. noise floor    spread of the control replicates (generator seed variance) per evaluator
3. paired deltas  per-image mPQ delta of each arm vs the control mean (arms share patch order,
                  base.npy) with image-bootstrap 95% CI; per-class Dead/Epith deltas
4. rank agreement Spearman between CF drop (real arm - stress arm) and real domain drop
                  (PanNuke test - mean external mPQ) across evaluators
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--cf-root", type=Path, default=Path("runs/pixcell/cf_fold3"))
p.add_argument("--evals", nargs="+", required=True, help="<run_dir>:<split>[,<split>...] entries")
p.add_argument("--ext", nargs="+", default=["conic", "monusac"], help="real external datasets")
p.add_argument("--arms", nargs="+", default=None, help="default: auto-detect from --cf-root")
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()

rng = np.random.default_rng(0)


def load(run: Path, name: str):
    d = run / f"eval_ext_{name}"
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    per = np.load(d / "per_image.npz")
    return {"s": s, "mPQ": per["mPQ"], "cls": per["class_PQ"]}


arms = a.arms or sorted(d.name for d in a.cf_root.iterdir() if (d / "images.npy").exists())
controls = [x for x in arms if x.startswith("control")]
stress = [x for x in arms if not x.startswith("control") and x != "real"]
print(f"[cf] arms: {arms}\n")

DEAD, EPITH = 3, 4  # class ids in class_PQ (Neopl Infla Conn Dead Epi)
evaluators = []
for spec in a.evals:
    run_s, splits = spec.split(":")
    for sp in splits.split(","):
        evaluators.append((f"{Path(run_s).name}/{sp}", Path(run_s), sp))

tab_rows, cf_drops, real_scores = [], {}, {}
for label, run, sp in evaluators:
    data = {arm: load(run / sp, arm) for arm in arms}
    have = [x for x in arms if data[x]]
    if "control_rep0" not in have:
        print(f"[skip] {label}: no CF evals found")
        continue
    print(f"\n===== {label} =====")
    print(f"  {'arm':14s} {'mPQ':>7s} {'bPQ':>7s} {'strict':>7s} {'F_d':>6s} {'Dead':>6s} {'Epith':>6s}")
    for arm in have:
        s = data[arm]["s"]
        print(f"  {arm:14s} {s['official']['mPQ']:7.4f} {s['official']['bPQ']:7.4f} "
              f"{s['strict']['mPQ']:7.4f} {s['detection']['F_d']:6.3f} "
              f"{s['per_class_PQ']['Dead']:6.3f} {s['per_class_PQ']['Epithelial']:6.3f}")

    # ---- noise floor: official mPQ spread across control replicates
    cvs = [data[c]["s"]["official"]["mPQ"] for c in controls if data[c]]
    if len(cvs) > 1:
        print(f"  [noise] control mPQ spread: std {np.std(cvs):.4f} (range {max(cvs) - min(cvs):.4f})")

    # ---- paired per-image deltas vs control mean
    ctrl_img = np.nanmean(np.stack([data[c]["mPQ"] for c in controls if data[c]]), 0)
    n = len(ctrl_img)
    print(f"  [paired deltas vs control mean, {n} images]  (mPQ  95%CI  Dead  Epith)")
    for arm in have:
        if arm.startswith("control"):
            continue
        d_img = data[arm]["mPQ"] - ctrl_img
        bs = np.array([np.nanmean(d_img[rng.integers(0, n, n)]) for _ in range(a.boot)])
        dd = data[arm]["cls"] - np.nanmean(np.stack([data[c]["cls"] for c in controls if data[c]]), 0)
        print(f"    {arm:12s} {np.nanmean(d_img):+.4f}  [{np.percentile(bs, 2.5):+.4f},"
              f"{np.percentile(bs, 97.5):+.4f}]  Dead {np.nanmean(dd[:, DEAD]):+.4f}"
              f"  Epith {np.nanmean(dd[:, EPITH]):+.4f}")
        if arm != "real":
            cf_drops[(label, arm)] = ctrl_img - data[arm]["mPQ"]  # >0 = arm hurts

    # ---- real-domain reference: PanNuke test fold + external zero-shot
    te = {1: 3, 2: 3, 3: 1}[int(sp.removeprefix("split"))]
    in_dom = None
    for d in sorted((run / sp).glob(f"*test_fold{te}")):
        if (d / "summary.json").exists():
            in_dom = json.loads((d / "summary.json").read_text())["official"]["mPQ"]
    ext = [load(run / sp, e) for e in a.ext]
    ext = [x for x in ext if x]
    if in_dom is not None and ext:
        real_scores[label] = (in_dom, np.mean([x["s"]["official"]["mPQ"] for x in ext]))
        print(f"  [ref] PanNuke test mPQ {in_dom:.4f} | ext mean mPQ "
              f"{real_scores[label][1]:.4f} ({', '.join(a.ext)})")

# ---- rank agreement: CF drop vs real domain drop across evaluators
print("\n===== rank agreement (CF drop vs real domain drop across evaluators) =====")
if len(real_scores) >= 3:
    rd = {k: v[0] - v[1] for k, v in real_scores.items()}
    print("  real domain drop:", {k: round(v, 4) for k, v in rd.items()})
    for arm in stress:
        keys = [k for k in real_scores if (k, arm) in cf_drops]
        if len(keys) < 3:
            continue
        x = [np.nanmean(cf_drops[(k, arm)]) for k in keys]
        y = [rd[k] for k in keys]
        rho_v, rho_p = stats.spearmanr(x, y)
        tau_v, _ = stats.kendalltau(x, y)
        print(f"  {arm:10s} CF drops {['%.4f' % v for v in x]}  spearman {rho_v:+.2f}"
              f" (p {rho_p:.2f})  kendall {tau_v:+.2f}")
else:
    print("  [skip] need >=3 evaluators with external + CF evals")
