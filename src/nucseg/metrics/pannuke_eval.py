"""PanNuke evaluation: official mPQ/bPQ (bit-exact semantics) plus the extensions in the plan.

Prediction format (one file per test fold), `.npz` with
    inst : (N, 256, 256) int   instance ids, 0 = background
    type : (N, 256, 256) uint8 per-pixel class 1..5 (instance class = majority vote inside it)

Reported:
  official   tissue-averaged mPQ / bPQ exactly as PanNuke-metrics run.py (per-image PQ, class
             NaN when absent from GT, image mean -> tissue nanmean -> mean over 19 tissues)
  strict     same, but a class absent from GT and present in the prediction scores 0, not NaN
  plus       CoNIC-style mPQ+ / bPQ+ (TP/FP/FN/IoU pooled over the whole fold, per class)
  per class  PQ (image nanmean, as the official script prints), plus DQ/SQ
  AJI, AJI+  image means
  detection  HoVer-Net F_d, type accuracy and F_c per class (centroid pairing within 12 px)
  errors     merge / split / missed / FP taxonomy, per tissue and per class, and per-GT records
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import CLASS_NAMES, NUM_CLASSES, TISSUES
from .errors import GT_STATUS, PRED_STATUS, classify_errors, touching_counts
from .instance import aji_from_overlap, centroids, overlap, pair_coordinates, pq_from_overlap, relabel


def instance_classes(inst: np.ndarray, typ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabelled instance map and the majority class (0 if none) of each instance 1..n."""
    lab, n = relabel(inst)
    if n == 0:
        return lab, np.zeros(0, np.int64)
    votes = np.zeros((n + 1, NUM_CLASSES + 1), np.int64)
    np.add.at(votes, (lab.ravel(), typ.ravel().astype(np.int64)), 1)
    votes[:, 0] = 0
    cls = votes[1:].argmax(1)
    cls[votes[1:].sum(1) == 0] = 0
    return lab, cls


def _eval_image(args):
    gt_ch, gt_inst, gt_type, pred_inst, pred_type = args
    out = {}
    g_lab, g_cls = instance_classes(gt_inst, gt_type)
    p_lab, p_cls = instance_classes(pred_inst, pred_type)

    # binary
    ov = overlap(g_lab, p_lab)
    s = pq_from_overlap(ov)
    out["bin"] = (s.tp, s.fp, s.fn, s.sum_iou, ov.nt)
    out["aji"] = aji_from_overlap(ov) if ov.nt else np.nan
    out["aji_plus"] = aji_from_overlap(ov, plus=True) if ov.nt else np.nan

    # per class (GT from the original channels, as the official script)
    cls_stats = []
    for c in range(NUM_CLASSES):
        t = gt_ch[..., c]
        p = np.where(np.isin(p_lab, np.nonzero(p_cls == c + 1)[0] + 1), p_lab, 0)
        ovc = overlap(t, p)
        sc = pq_from_overlap(ovc)
        cls_stats.append((sc.tp, sc.fp, sc.fn, sc.sum_iou, ovc.nt, ovc.np_))
    out["cls"] = cls_stats

    # detection / classification (HoVer-Net compute_stats, 12 px at 40x)
    _, gc = centroids(g_lab)
    _, pc = centroids(p_lab)
    pairing, un_t, un_p = pair_coordinates(gc, pc, 12)
    out["det"] = (g_cls[pairing[:, 0]], p_cls[pairing[:, 1]], g_cls[un_t], p_cls[un_p])

    # error taxonomy
    gs, ps, max_iou = classify_errors(ov)
    touch = touching_counts(g_lab, ov.nt)
    matched_pred_cls = np.zeros(ov.nt, np.int64)
    if len(s.paired):
        matched_pred_cls[s.paired[:, 0]] = p_cls[s.paired[:, 1]]
    out["gt_records"] = np.stack([g_cls, ov.area_t.astype(np.int64), touch, gs.astype(np.int64),
                                  matched_pred_cls, np.round(max_iou * 1000).astype(np.int64)], 1) \
        if ov.nt else np.zeros((0, 6), np.int64)
    out["pred_records"] = np.stack([p_cls, ov.area_p.astype(np.int64), ps.astype(np.int64)], 1) \
        if ov.np_ else np.zeros((0, 3), np.int64)
    return out


def _pq(tp, fp, fn, siou):
    d = tp + 0.5 * fp + 0.5 * fn
    if d == 0:
        return np.nan
    return (tp / d) * (siou / (tp + 1.0e-6))


def _f1_type(pt, pp, ut, up, c, w=(2, 2, 1, 1)):
    sel = (pt == c) | (pp == c)
    pt, pp = pt[sel], pp[sel]
    tp_tn = ((pt == c) & (pp == c)).sum() + ((pt != c) & (pp != c)).sum()
    fp_dt = ((pt != c) & (pp == c)).sum()
    fn_dt = ((pt == c) & (pp != c)).sum()
    fp_d, fn_d = (up == c).sum(), (ut == c).sum()
    den = 2 * tp_tn + w[0] * fp_dt + w[1] * fn_dt + w[2] * fp_d + w[3] * fn_d
    return 2 * tp_tn / den if den else np.nan


def evaluate(gt_channels, gt_inst, gt_type, tissue, pred_inst, pred_type, workers: int = 16,
             tissue_names: list[str] | None = None) -> dict:
    """tissue_names: labels for the ids in `tissue` (defaults to the 19 PanNuke tissues); external
    datasets pass their own pseudo-tissue names so per_tissue reports are readable."""
    tnames = tissue_names or TISSUES
    n = len(tissue)
    assert pred_inst.shape == gt_inst.shape and pred_type.shape == gt_type.shape
    jobs = ((gt_channels[i], np.asarray(gt_inst[i]), np.asarray(gt_type[i]),
             np.asarray(pred_inst[i]), np.asarray(pred_type[i])) for i in range(n))
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(_eval_image, jobs, chunksize=16)
    else:
        res = [_eval_image(j) for j in jobs]

    # ---- per-image PQ tables
    bpq = np.array([_pq(*r["bin"][:4]) if r["bin"][4] > 0 else np.nan for r in res])
    cls = np.array([r["cls"] for r in res], dtype=np.float64)  # (n, 5, 6)
    gt_present = cls[..., 4] > 0
    pred_present = cls[..., 5] > 0
    cls_pq = np.full((n, NUM_CLASSES), np.nan)
    for i in range(n):
        for c in range(NUM_CLASSES):
            if gt_present[i, c]:
                cls_pq[i, c] = _pq(*cls[i, c, :4])
    strict_pq = np.where(~gt_present & pred_present, 0.0, cls_pq)

    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        img_mpq = np.nanmean(cls_pq, 1)
        img_mpq_strict = np.nanmean(strict_pq, 1)

        def tissue_avg(v):
            per = {t: float(np.nanmean(v[tissue == t])) for t in tnames if (tissue == t).any()}
            return float(np.nanmean(list(per.values()))), per

        mpq, mpq_t = tissue_avg(img_mpq)
        bpq_m, bpq_t = tissue_avg(bpq)
        mpq_s, mpq_s_t = tissue_avg(img_mpq_strict)
        per_class_pq = {CLASS_NAMES[c]: float(np.nanmean(cls_pq[:, c])) for c in range(NUM_CLASSES)}
        per_class_pq_strict = {CLASS_NAMES[c]: float(np.nanmean(strict_pq[:, c])) for c in range(NUM_CLASSES)}
        # tissue x class PQ (image nanmean within tissue)
        tissue_class = {t: {CLASS_NAMES[c]: float(np.nanmean(cls_pq[tissue == t, c])) for c in range(NUM_CLASSES)}
                        for t in tnames if (tissue == t).any()}

    # ---- pooled (CoNIC-style) PQ+
    pooled = cls[..., :4].sum(0)
    pq_plus = {CLASS_NAMES[c]: float(_pq(*pooled[c])) for c in range(NUM_CLASSES)}
    dq_plus = {CLASS_NAMES[c]: float(pooled[c, 0] / max(pooled[c, 0] + 0.5 * pooled[c, 1] + 0.5 * pooled[c, 2], 1e-9))
               for c in range(NUM_CLASSES)}
    sq_plus = {CLASS_NAMES[c]: float(pooled[c, 3] / (pooled[c, 0] + 1e-6)) for c in range(NUM_CLASSES)}
    bin_pooled = np.array([r["bin"][:4] for r in res], dtype=np.float64).sum(0)

    # ---- detection / classification
    pt = np.concatenate([r["det"][0] for r in res])
    pp = np.concatenate([r["det"][1] for r in res])
    ut = np.concatenate([r["det"][2] for r in res])
    up = np.concatenate([r["det"][3] for r in res])
    tp_d, fp_d, fn_d = len(pt), len(up), len(ut)
    f_d = 2 * tp_d / (2 * tp_d + fp_d + fn_d)
    prec_d, rec_d = tp_d / max(tp_d + fp_d, 1), tp_d / max(tp_d + fn_d, 1)
    f_c = {CLASS_NAMES[c - 1]: float(_f1_type(pt, pp, ut, up, c)) for c in range(1, NUM_CLASSES + 1)}

    # ---- error taxonomy
    gt_rec = pd.DataFrame(np.concatenate([np.c_[np.full(len(r["gt_records"]), i), r["gt_records"]]
                                          for i, r in enumerate(res)]),
                          columns=["image", "cls", "area", "n_touch", "status", "pred_cls", "max_iou_x1000"])
    gt_rec["tissue"] = tissue[gt_rec["image"].values]
    gt_rec["status"] = gt_rec["status"].map(dict(enumerate(GT_STATUS)))
    pred_rec = pd.DataFrame(np.concatenate([np.c_[np.full(len(r["pred_records"]), i), r["pred_records"]]
                                            for i, r in enumerate(res)]),
                            columns=["image", "cls", "area", "status"])
    pred_rec["tissue"] = tissue[pred_rec["image"].values]
    pred_rec["status"] = pred_rec["status"].map(dict(enumerate(PRED_STATUS)))

    gt_err = gt_rec.groupby("status").size().reindex(GT_STATUS, fill_value=0)
    pred_err = pred_rec.groupby("status").size().reindex(PRED_STATUS, fill_value=0)
    matched = gt_rec[gt_rec.status == "matched"]
    confusion = pd.crosstab(matched["cls"], matched["pred_cls"]).reindex(
        index=range(1, 6), columns=range(0, 6), fill_value=0)

    summary = {
        "n_images": int(n),
        "official": {"mPQ": mpq, "bPQ": bpq_m},
        "strict": {"mPQ": mpq_s},
        "plus": {"mPQ+": float(np.nanmean(list(pq_plus.values()))), "bPQ+": float(_pq(*bin_pooled))},
        "per_class_PQ": per_class_pq,
        "per_class_PQ_strict": per_class_pq_strict,
        "per_class_PQ+": pq_plus, "per_class_DQ+": dq_plus, "per_class_SQ+": sq_plus,
        "per_tissue": {t: {"mPQ": mpq_t[t], "bPQ": bpq_t[t], "mPQ_strict": mpq_s_t[t]} for t in mpq_t},
        "tissue_class_PQ": tissue_class,
        "AJI": float(np.nanmean([r["aji"] for r in res])),
        "AJI+": float(np.nanmean([r["aji_plus"] for r in res])),
        "detection": {"F_d": f_d, "P_d": prec_d, "R_d": rec_d,
                      "type_acc": float((pt == pp).mean()) if len(pt) else np.nan, "F_c": f_c},
        "errors": {"gt": {k: int(v) for k, v in gt_err.items()},
                   "pred": {k: int(v) for k, v in pred_err.items()},
                   "gt_rate": {k: float(v / max(len(gt_rec), 1)) for k, v in gt_err.items()}},
        "class_confusion_matched": {"rows_gt_1to5": confusion.values.tolist(), "cols_pred_0to5": list(range(6))},
    }
    return {"summary": summary, "gt_records": gt_rec, "pred_records": pred_rec,
            "image_mPQ": img_mpq, "image_bPQ": bpq, "image_class_PQ": cls_pq}


def save_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2))
    result["gt_records"].to_csv(out_dir / "gt_records.csv.gz", index=False)
    result["pred_records"].to_csv(out_dir / "pred_records.csv.gz", index=False)
    np.savez_compressed(out_dir / "per_image.npz", mPQ=result["image_mPQ"], bPQ=result["image_bPQ"],
                        class_PQ=result["image_class_PQ"])
    gt = result["gt_records"]
    tab = pd.crosstab([gt["tissue"]], gt["status"], normalize="index").reindex(columns=GT_STATUS, fill_value=0)
    tab.to_csv(out_dir / "errors_by_tissue.csv")
    pd.crosstab(gt["cls"].map(dict(enumerate(["bg"] + CLASS_NAMES))), gt["status"], normalize="index") \
        .reindex(columns=GT_STATUS, fill_value=0).to_csv(out_dir / "errors_by_class.csv")
    gt["touch_bin"] = pd.cut(gt["n_touch"], [-1, 0, 1, 2, 4, 100], labels=["0", "1", "2", "3-4", "5+"])
    pd.crosstab(gt["touch_bin"], gt["status"], normalize="index").reindex(columns=GT_STATUS, fill_value=0) \
        .to_csv(out_dir / "errors_by_touching.csv")


def format_summary(s: dict) -> str:
    lines = [f"mPQ {s['official']['mPQ']:.4f}  bPQ {s['official']['bPQ']:.4f}  "
             f"mPQ(strict) {s['strict']['mPQ']:.4f}  mPQ+ {s['plus']['mPQ+']:.4f}  "
             f"AJI {s['AJI']:.4f}  AJI+ {s['AJI+']:.4f}",
             "PQ   " + "  ".join(f"{k[:5]} {v:.3f}" for k, v in s["per_class_PQ"].items()),
             f"F_d {s['detection']['F_d']:.3f} (P {s['detection']['P_d']:.3f} R {s['detection']['R_d']:.3f})  "
             + "F_c " + "  ".join(f"{k[:5]} {v:.3f}" for k, v in s["detection"]["F_c"].items()),
             "GT   " + "  ".join(f"{k} {v:.3f}" for k, v in s["errors"]["gt_rate"].items())]
    return "\n".join(lines)
