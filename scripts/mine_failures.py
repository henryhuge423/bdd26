#!/usr/bin/env python
"""Mine local-layout statistics of detection failures (pillar B, failure-driven synthesis input).

For every GT nucleus of a fold, given out-of-fold predictions (a model that never trained on this
fold; e.g. split-2 model -> fold 1, which is split 2's val fold), record:

    image, class, area, status (matched/merged/split/missed_bg/missed_shape), n_touch,
    per-class touching-neighbour flags, # touching neighbours that are themselves missed

and summarise: miss rate vs (class, area, n_touch), and the touching-class-pair distribution of
missed vs matched nuclei (which layouts to synthesise: e.g. Dead adjacent to Neoplastic).

    python scripts/mine_failures.py --pred runs/cellvit_uni/split2/pred_fold1.npz --fold 1 \
        --out runs/analysis
"""
import argparse
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from nucseg.constants import CLASS_NAMES
from nucseg.data.pannuke import PanNukeFold
from nucseg.metrics.errors import GT_STATUS, classify_errors
from nucseg.metrics.instance import overlap
from nucseg.metrics.pannuke_eval import instance_classes

p = argparse.ArgumentParser()
p.add_argument("--pred", type=Path, required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--out", type=Path, default=Path("runs/analysis"))
p.add_argument("--workers", type=int, default=16)
a = p.parse_args()

f = PanNukeFold(a.fold)
z = np.load(a.pred)
assert z["inst"].shape[0] == len(f), "pred/fold image count mismatch"

AREA_BINS = [60, 100, 200]
AREA_LABELS = ["<60", "60-100", "100-200", ">200"]


def _one(args):
    """Per-image GT-nucleus layout records; see header for columns."""
    gi, ti, pi, pty = args
    g_lab, g_cls = instance_classes(gi, ti)
    p_lab, _ = instance_classes(pi, pty)
    ov = overlap(g_lab, p_lab)
    n = ov.nt
    if n == 0:
        return np.zeros((0, 11), np.int32)
    gs, _, _ = classify_errors(ov)
    touch = np.zeros((n, 5), bool)  # does instance touch another of class c (4-adjacency)
    ps = []
    for aa, bb in ((g_lab[:, :-1], g_lab[:, 1:]), (g_lab[:-1, :], g_lab[1:, :])):
        m = (aa != bb) & (aa > 0) & (bb > 0)
        ps += [aa[m] * (n + 1) + bb[m], bb[m] * (n + 1) + aa[m]]  # both directions, 1D keys
    keys = np.unique(np.concatenate(ps))
    touch[keys // (n + 1) - 1, g_cls[keys % (n + 1) - 1] - 1] = True
    missed_any = gs >= GT_STATUS.index("merged")  # merged/split/missed_* = not matched
    n_miss_nb = (touch & missed_any[:, None]).sum(1)
    return np.column_stack([g_cls, ov.area_t, gs, touch.sum(1), touch, n_miss_nb]).astype(np.int32)


if __name__ == "__main__":
    zinst, ztype = z["inst"], z["type"]  # decompress once; NpzFile re-decompresses per access
    jobs = [(np.asarray(f.inst[i]), np.asarray(f.type[i]), zinst[i], ztype[i])
            for i in range(len(f))]
    with Pool(a.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=32)
    rec = np.concatenate([np.column_stack([np.full(len(r), i), r]) for i, r in enumerate(rows) if len(r)])
    cols = ["img", "cls", "area", "status", "n_touch"] + [f"t_{c}" for c in CLASS_NAMES] + ["n_miss_nb"]
    out_csv = a.out / f"failures_fold{a.fold}.csv.gz"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_csv, "wt") as fh:
        fh.write(",".join(cols) + "\n")
        np.savetxt(fh, rec, fmt="%d", delimiter=",")

    # ---- summaries
    cls_, area_, status_, ntouch = rec[:, 1] - 1, rec[:, 2], rec[:, 3], rec[:, 4]
    missed = status_ >= 1
    missed_bg = status_ == 3
    summ = {"n_nuclei": int(len(rec)), "n_images": int(len(f)), "pred": str(a.pred),
            "missed_any": float(missed.mean()), "missed_bg": float(missed_bg.mean())}
    print(f"fold {a.fold}: {len(rec)} GT nuclei; missed(any) {missed.mean():.3f}, "
          f"missed_bg {missed_bg.mean():.3f}")

    print("\n== miss rate (missed_bg) by class x area bin ==")
    ab = np.digitize(area_, AREA_BINS)
    for c in range(5):
        line = f"{CLASS_NAMES[c]:13s}"
        for b in range(4):
            m = (cls_ == c) & (ab == b)
            line += f"  {AREA_LABELS[b]}:{missed_bg[m].mean():.2f}(n={m.sum()})" if m.any() else ""
        print(line)

    print("\n== miss rate (missed_bg) by class x n_touch (0 / 1 / 2+ touching) ==")
    nt = np.minimum(ntouch, 2)
    for c in range(5):
        line = f"{CLASS_NAMES[c]:13s}"
        for t in range(3):
            m = (cls_ == c) & (nt == t)
            line += f"  t{t}:{missed_bg[m].mean():.2f}(n={m.sum()})" if m.any() else f"  t{t}:-"
        print(line)

    print("\n== fraction of nuclei touching class c (missed_bg vs matched) ==")
    pair = {}
    for grp, m in (("missed", missed_bg), ("matched", status_ == 0)):
        tab = {CLASS_NAMES[c]: {CLASS_NAMES[d]: float(rec[m & (cls_ == c), 5 + d].mean())
                                for d in range(5)} for c in range(5)}
        pair[grp] = tab
        print(f"-- {grp}:")
        print("             " + "  ".join(f"{d[:6]:>6s}" for d in CLASS_NAMES))
        for c in range(5):
            print(f"{CLASS_NAMES[c]:13s}" + "  ".join(f"{tab[CLASS_NAMES[c]][d]:6.2f}" for d in CLASS_NAMES))
    summ["touch_pairs"] = pair

    print("\n== Dead nuclei: status x touching class (n) ==")
    dead = cls_ == CLASS_NAMES.index("Dead")
    for s_i, sname in enumerate(GT_STATUS):
        sel = dead & (status_ == s_i)
        if sel.sum() == 0:
            continue
        counts = {CLASS_NAMES[d]: int(rec[sel, 5 + d].sum()) for d in range(5)}
        print(f"  {sname:12s} n={int(sel.sum()):4d}  " +
              "  ".join(f"{k[:6]}:{v}" for k, v in counts.items()))

    (a.out / f"failures_fold{a.fold}.json").write_text(json.dumps(summ, indent=1))
    print(f"\nwrote {out_csv}")
