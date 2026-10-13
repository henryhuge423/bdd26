#!/usr/bin/env python
"""M diagnostic: official/strict mPQ under all 10 class-pair merges (GT+pred remapped
together) plus the typing-oracle interventions recomputed in the {Inf,Conn} merged space.

Read-only diagnostic over the frozen 3-split test predictions (spec 2026-10-13
`docs/superpowers/specs/2026-10-13-merged-class-m-design.md`): no training, no tuning,
no selection; results are metric-side descriptions of label-boundary ambiguity weight,
never a method result and never additive "ambiguity share" claims.

    python scripts/merged_class_diag.py [--out runs/analysis/merged_class_20261013]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lever_postproc import file_provenance  # noqa: E402
from typing_oracle import instance_table  # noqa: E402

from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.metrics.fast_retype import RetypeEvaluator  # noqa: E402
from nucseg.metrics.merged_class import bootstrap_delta, class_map, mpq, mpq_strict  # noqa: E402
from nucseg.metrics.instance import overlap, pq_from_overlap, relabel  # noqa: E402
from nucseg.metrics.pannuke_eval import instance_classes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEST_FOLD = {1: 3, 2: 3, 3: 1}
ALLOWED_PREDS = {s: (ROOT / f"runs/cellvit_uni/split{s}/pred_test_fold{f}.npz").resolve()
                 for s, f in TEST_FOLD.items()}
FROZEN_EVAL = {s: ROOT / f"runs/cellvit_uni/split{s}/eval_test_fold{TEST_FOLD[s]}/summary.json"
               for s in TEST_FOLD}
PAIRS = [(a, b) for a in range(1, 6) for b in range(a + 1, 6)]
BOOT_SEED = 20261013
N_BOOT = 1000
TOL = 1e-12  # frozen values come from pannuke_eval's matrix-sum order; the fast path
# accumulates bincounts row-wise, so per-class/bPQ can differ at the last 1-2 ulp
# (measured 2026-10-13 on typing_oracle vs eval summary: mPQ bit-exact, per-class ulp).


def _bpq_img(args):
    """Official binary PQ of one image, computed on the INST maps exactly like
    pannuke_eval._eval_image (NaN when the image has no GT instances).

    The match tables are channel-based, and PanNuke channels disagree with the inst
    maps on a small fraction of pixels (measured 2026-10-13: 173/2722 fold3 images,
    e.g. img 77's 3 foreign-type pixels) — so bPQ must NOT be derived from them."""
    gt_inst, pred_inst = args
    g_lab, _ = relabel(np.asarray(gt_inst))
    p_lab, _ = relabel(np.asarray(pred_inst))
    ov = overlap(g_lab, p_lab)
    if ov.nt == 0:
        return np.nan
    s = pq_from_overlap(ov)
    d = s.tp + 0.5 * s.fp + 0.5 * s.fn
    return (s.tp / d) * (s.sum_iou / (s.tp + 1.0e-6))


def bpq_inst_maps(fold, pred_inst, workers: int = 16) -> tuple[float, np.ndarray]:
    """(tissue-averaged bPQ, per-image bPQ) from the inst maps — invariant 1 anchor."""
    from multiprocessing import Pool

    from nucseg.constants import TISSUES
    n = len(fold.tissue)
    jobs = ((fold.inst[j], pred_inst[j]) for j in range(n))
    with Pool(min(workers, n)) as pool:
        per = np.array(pool.map(_bpq_img, jobs, chunksize=16))
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        per_t = [np.nanmean(per[fold.tissue == t]) for t in TISSUES
                 if (fold.tissue == t).any()]
        return float(np.nanmean(per_t)), per


def load_split(split: int, pred_override: dict[int, str] | None = None) -> tuple[dict, object, dict]:
    path = Path(pred_override.get(split, ALLOWED_PREDS[split])).resolve()
    if path != ALLOWED_PREDS[split]:
        raise SystemExit(f"{path} is not the allowlisted prediction for split {split} "
                         f"({ALLOWED_PREDS[split]}); spec 2026-10-13 §5 refuses other inputs")
    pred = dict(np.load(path))
    fold = PanNukeFold(TEST_FOLD[split])
    frozen = json.loads(FROZEN_EVAL[split].read_text())
    return pred, fold, frozen


def analyze_split(pred: dict, fold, split: int, frozen: dict | None) -> dict:
    inst_img, inst_id = instance_table(pred)
    ev = RetypeEvaluator(fold.gt_channels, pred["inst"], inst_img, inst_id, fold.tissue)
    cls = np.concatenate([instance_classes(pred["inst"][j], pred["type"][j])[1]
                          for j in range(len(fold.tissue))])

    ident = class_map(None)
    m0_m, m0_per = mpq(ev, cls, ident)
    m0_s, m0_sper = mpq_strict(ev, cls, ident)
    b_m, _ = bpq_inst_maps(fold, pred["inst"])
    m0 = {"mpq": float(m0_m), "strict": float(m0_s),
          "per_class": [float(v) for v in m0_per],
          "per_class_strict": [float(v) for v in m0_sper],
          "bpq_recomputed": float(b_m)}
    invariants: dict[str, bool] = {}
    if frozen is not None:
        names = list(frozen["per_class_PQ"].keys())
        if m0["mpq"] != frozen["official"]["mPQ"]:
            raise SystemExit(f"split{split}: M0 mPQ {m0['mpq']!r} != frozen "
                             f"{frozen['official']['mPQ']!r} (binding, bit-exact)")
        if abs(m0["bpq_recomputed"] - frozen["official"]["bPQ"]) > TOL:
            raise SystemExit(f"split{split}: recomputed bPQ {m0['bpq_recomputed']!r} != frozen "
                             f"{frozen['official']['bPQ']!r} (tol {TOL})")
        for i, nm in enumerate(names):
            if abs(m0["per_class"][i] - frozen["per_class_PQ"][nm]) > TOL:
                raise SystemExit(f"split{split}: M0 per-class {nm} {m0['per_class'][i]!r} != "
                                 f"frozen {frozen['per_class_PQ'][nm]!r} (tol {TOL})")
        invariants["m0_bitexact"] = True
        invariants["bpq_matches_frozen"] = True

    merges: dict[str, dict] = {}
    for pair in PAIRS:
        cmap = class_map(pair)
        m, per = mpq(ev, cls, cmap)
        s, sper = mpq_strict(ev, cls, cmap)
        merges[f"({pair[0]},{pair[1]})"] = {
            "mpq": float(m), "strict": float(s),
            "per_class": [float(v) for v in per],
            "per_class_strict": [float(v) for v in sper],
            "delta_official": bootstrap_delta(ev, cls, ident, cmap, N_BOOT, BOOT_SEED + split),
            "delta_strict": bootstrap_delta(ev, cls, ident, cmap, N_BOOT, BOOT_SEED + split,
                                            strict=True)}

    # 不变量 2：merge(2,3) 不触碰 Dead —— 合并空间 Dead 列 == 恒等空间 Dead 列（逐位）
    dead_ok = merges["(2,3)"]["per_class"][2] == m0["per_class"][3]
    if not dead_ok:
        raise SystemExit(f"split{split}: Dead PQ changed under merge(2,3) "
                         f"({merges['(2,3)']['per_class'][2]!r} vs {m0['per_class'][3]!r})")
    invariants["dead_pq_unchanged"] = dead_ok

    cmap23 = class_map((2, 3))
    oracle = cls.copy()
    oracle[ev.m_row] = ev.m_cls
    inf_conn = cls.copy()
    sel = np.isin(ev.m_cls, [2, 3])
    inf_conn[ev.m_row[sel]] = ev.m_cls[sel]
    oum: dict[str, dict] = {}
    for name, assignment in [("current", cls), ("oracle", oracle), ("inf_conn", inf_conn)]:
        m, _ = mpq(ev, assignment, cmap23)
        oum[name] = {"mpq": float(m)}
        if name != "current":  # current vs itself would be a vacuous all-zero bootstrap
            oum[name]["delta_vs_current"] = bootstrap_delta(
                ev, cls, cmap23, cmap23, N_BOOT, BOOT_SEED + split, cls_b=assignment)
    return {"split": split, "fold": TEST_FOLD[split], "m0": m0, "merges": merges,
            "oracle_under_merge": oum, "invariants": invariants}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=ROOT / "runs/analysis/merged_class_20261013")
    p.add_argument("--pred", nargs="*", default=[], metavar="SPLIT=PATH",
                   help="optional; every path must resolve to the allowlisted prediction")
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists — refusing to overwrite a round directory")
    overrides = {}
    for item in a.pred:
        s, _, path = item.partition("=")
        if s.lstrip("-").isdigit() and int(s) in TEST_FOLD:
            overrides[int(s)] = path
        else:
            raise SystemExit(f"--pred entry '{item}' is not SPLIT=PATH with SPLIT in {sorted(TEST_FOLD)}")

    t0 = time.time()
    rows = []
    provenance = {}
    hashed_gt = set()
    for split in (1, 2, 3):
        ts = time.time()
        pred, fold, frozen = load_split(split, overrides)
        provenance[f"pred_split{split}"] = file_provenance(ALLOWED_PREDS[split])
        provenance[f"frozen_eval_split{split}"] = file_provenance(FROZEN_EVAL[split])
        if TEST_FOLD[split] not in hashed_gt:  # splits 1/2 share fold3's GT file
            provenance[f"gt_channels_fold{TEST_FOLD[split]}"] = file_provenance(
                fold.dir / "gt_channels.npz")
            hashed_gt.add(TEST_FOLD[split])
        rows.append(analyze_split(pred, fold, split, frozen))
        print(f"split{split} done in {time.time() - ts:.1f}s", flush=True)
    provenance["script"] = file_provenance(Path(__file__))["sha256"]
    provenance["merged_class_py"] = file_provenance(
        Path(__file__).resolve().parents[1] / "src/nucseg/metrics/merged_class.py")["sha256"]
    provenance["fast_retype_py"] = file_provenance(
        Path(__file__).resolve().parents[1] / "src/nucseg/metrics/fast_retype.py")["sha256"]

    mean = {"m0_mpq": float(np.mean([r["m0"]["mpq"] for r in rows])),
            "m0_strict": float(np.mean([r["m0"]["strict"] for r in rows]))}
    for pair in PAIRS:
        key = f"({pair[0]},{pair[1]})"
        mean[key] = {"mpq": float(np.mean([r["merges"][key]["mpq"] for r in rows])),
                     "strict": float(np.mean([r["merges"][key]["strict"] for r in rows])),
                     "delta": float(np.mean([r["merges"][key]["mpq"] for r in rows])) - mean["m0_mpq"]}
    for name in ("current", "oracle", "inf_conn"):
        mean[f"oum_{name}"] = float(np.mean([r["oracle_under_merge"][name]["mpq"] for r in rows]))

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(
        {"rows": rows, "mean": mean, "pairs": [list(p) for p in PAIRS],
         "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "tol": TOL,
         "provenance": provenance}, indent=2))

    lines = ["# M merged-class diagnostic (spec 2026-10-13, descriptive — no gates)", "",
             "3-split mean Δ vs M0; brackets = [min lo, max hi] envelope of the three",
             "per-split image-cluster bootstrap CIs (per-split CIs live in results.json).", "",
             "| pair | Δofficial | envelope | Δstrict | envelope | merged mPQ | merged strict |",
             "|---|---|---|---|---|---|---|"]
    for pair in PAIRS:
        key = f"({pair[0]},{pair[1]})"
        d = mean[key]
        lo = min(r["merges"][key]["delta_official"]["lo"] for r in rows)
        hi = max(r["merges"][key]["delta_official"]["hi"] for r in rows)
        dsm = float(np.mean([r["merges"][key]["strict"] for r in rows])) - mean["m0_strict"]
        los = min(r["merges"][key]["delta_strict"]["lo"] for r in rows)
        his = max(r["merges"][key]["delta_strict"]["hi"] for r in rows)
        lines.append(f"| {key} | {d['delta']:+.4f} | [{lo:+.4f},{hi:+.4f}] "
                     f"| {dsm:+.4f} | [{los:+.4f},{his:+.4f}] "
                     f"| {d['mpq']:.4f} | {d['strict']:.4f} |")
    lines += ["", f"M0 (identity): mPQ {mean['m0_mpq']:.4f} strict {mean['m0_strict']:.4f}",
              "", "bPQ is class-agnostic: identical under every merge by construction (merges",
              "never touch binary matching); per split it is recomputed from the inst maps and",
              "bound to the frozen eval (see invariants in results.json).", "",
              "oracle-under-merge(2,3): " +
              " ".join(f"{k} {mean[f'oum_{k}']:.4f}" for k in ("current", "oracle", "inf_conn")),
              "", "Per-class PQ (3-split mean, official): M0 vs merged classes of (2,3)",
              "| class | M0 | merge(2,3) space |", "|---|---|---|"]
    pc0 = np.mean([r["m0"]["per_class"] for r in rows], 0)
    pc23 = np.mean([r["merges"]["(2,3)"]["per_class"] for r in rows], 0)
    # merge(2,3) 合并类序 = [Neo, Inf+Conn, Dead, Epi]；原类 0/3/4 -> 合并列 0/2/3
    for i, nm in enumerate(["Neo", "Inf", "Conn", "Dead", "Epi"]):
        col = 1 if i in (1, 2) else {0: 0, 3: 2, 4: 3}[i]
        tag = " (merged Inf+Conn)" if i in (1, 2) else ""
        lines.append(f"| {nm} | {pc0[i]:.4f} | {pc23[col]:.4f}{tag} |")
    (a.out / "tables.md").write_text("\n".join(lines) + "\n")
    (a.out / "commands.md").write_text(
        f"# commands\n\n- host: {__import__('socket').gethostname()}\n"
        f"- cmd: `python scripts/merged_class_diag.py --out {a.out}`\n"
        f"- env: ~/.conda/envs/nuclei (numpy {np.__version__})\n"
        f"- wall: {time.time() - t0:.1f}s\n"
        f"- binding tolerances: mPQ bit-exact; per-class/bPQ <= {TOL} "
        "(frozen values use pannuke_eval matrix-sum order; fast path bincount row order —\n"
        " measured ulp-level on typing_oracle 2026-10-13)\n")
    print(f"\nwrote {a.out}/results.json")
    print(f"M0 mPQ {mean['m0_mpq']:.4f} | merge(2,3) {mean['(2,3)']['mpq']:.4f} "
          f"(Δ {mean['(2,3)']['delta']:+.4f}) | oracle-under-merge "
          f"{mean['oum_oracle']:.4f} vs current {mean['oum_current']:.4f}")


if __name__ == "__main__":
    main()
