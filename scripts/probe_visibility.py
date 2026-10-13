#!/usr/bin/env python
"""E4 visibility probe: controlled FOV-crop conditions on the frozen split1 checkpoint.

Spec docs/superpowers/specs/2026-10-13-visibility-probe-e4-design.md (user-approved
2026-10-13): no training, frozen-checkpoint inference only, fold2 (split1 val) read-only,
pre-registered gates H1/H2/H3 deciding whether a visibility-supervision training round may
be drafted. Gate numbers are frozen; verdicts are recomputable from probe.json.

    python scripts/probe_visibility.py [--run runs/cellvit_uni/split1] [--fold 2] \
        [--out runs/analysis/visibility_probe_20261013]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lever_postproc import file_provenance  # noqa: E402

from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.hovernet.data import hv_targets  # noqa: E402
from nucseg.hovernet.engine import _post  # noqa: E402
from nucseg.cellvit.engine import (TrainConfig, _forward_probs, build_model,  # noqa: E402
                                   run_dead_expert, run_widen)
from nucseg.cellvit.model import UNI_MEAN, UNI_STD  # noqa: E402
from nucseg.constants import NUM_CLASSES  # noqa: E402
from nucseg.metrics.errors import classify_errors  # noqa: E402
from nucseg.metrics.instance import overlap  # noqa: E402
from nucseg.metrics.pannuke_eval import evaluate, instance_classes  # noqa: E402
from nucseg.visibility_probe import (bin_area, bin_area_native, crop_group,  # noqa: E402
                                     max_iou_table, miss_table, original_full,
                                     rate_gap_bootstrap, sample_windows, shift_pad)

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RUN = (ROOT / "runs/cellvit_uni/split1").resolve()
ALLOWED_CKPT = ALLOWED_RUN / "final.pth"
ALLOWED_PRED = ALLOWED_RUN / "pred_fold2.npz"
FROZEN_EVAL = ALLOWED_RUN / "eval_fold2/summary.json"
WINDOW_SEED = 20261013
BOOT_SEED = 20261013
N_BOOT = 1000
SHIFT = 12
GATE_H1, GATE_H2, GATE_H3 = 0.05, 0.03, 0.05  # spec §2 frozen numbers


def gt_class_map(inst: np.ndarray, typ: np.ndarray) -> dict[int, int]:
    """Original id -> majority GT class (official vote; ties -> smallest class)."""
    _, cls = instance_classes(inst, typ)
    ids = np.unique(inst)
    ids = ids[ids > 0]
    return {int(i): int(c) for i, c in zip(ids, cls)}


def _resample_rows(img: np.ndarray, tissue: np.ndarray, rng) -> np.ndarray:
    """Image rows of one draw: images resampled with replacement WITHIN each tissue."""
    rows = []
    for t in dict.fromkeys(tissue):
        s = np.nonzero(tissue == t)[0]
        imgs = np.unique(img[s])
        picks = rng.choice(imgs, len(imgs), replace=True)
        rows.append(np.concatenate([np.nonzero(img == i)[0] for i in picks]))
    return np.concatenate(rows) if rows else np.zeros(0, int)


def did_bootstrap(tr_ck, tr_c0, fu_ck, fu_c0, img, tissue, n_boot, seed) -> dict:
    """DiD with image-clustered, tissue-stratified bootstrap (draws mirror the point:
    pooled group rates on the drawn multiset; same draw serves all four groups)."""
    def rates(rows):
        with np.errstate(all="ignore"):
            return (np.nanmean(tr_ck[rows]), np.nanmean(tr_c0[rows]),
                    np.nanmean(fu_ck[rows]), np.nanmean(fu_c0[rows]))

    rng = np.random.default_rng(seed)
    deltas = np.full(n_boot, np.nan)
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        for k in range(n_boot):
            rows = _resample_rows(img, tissue, rng)
            if len(rows):
                t_ck, t_c0, f_ck, f_c0 = rates(rows)
                deltas[k] = (t_ck - t_c0) - (f_ck - f_c0)
        lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    t_ck, t_c0, f_ck, f_c0 = rates(np.arange(len(img)))
    return {"did": float((t_ck - t_c0) - (f_ck - f_c0)), "lo": float(lo), "hi": float(hi)}


def analyze_records(rec: list[dict], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Pure gate arithmetic over per-nucleus records (cond/group/frag/miss/image/tissue).
    H1: C0 Dead border-vs-interior gap (pooled per-nucleus rates, clustered CI).
    H2: DiD per crop condition vs C0 baseline, paired by (image, gt_id).
    H3: C2 truncated <30 px vs >=100 px fragment-bin gap."""
    def arr(rows, key):
        return np.array([r[key] for r in rows], float)

    c0 = [r for r in rec if r["cond"] == "C0"]
    dead = [r for r in c0 if r["cls"] == 4]
    a = np.array([r["miss"] if r["group"] == "border" else np.nan for r in dead], float)
    b = np.array([r["miss"] if r["group"] == "interior" else np.nan for r in dead], float)
    h1 = rate_gap_bootstrap(a, b, arr(dead, "image"), np.asarray([r["tissue"] for r in dead]),
                            n_boot, seed)
    h1_pass = h1["gap"] >= GATE_H1 and h1["lo"] > 0

    c0_by_key = {(r["image"], r["gt_id"]): r for r in c0 if r["original_full"]}
    dids = {}
    for cond in ("C-int", "C2", "C3"):
        rows = [r for r in rec if r["cond"] == cond and r["group"] in ("truncated", "stays_full")]
        pairs = [r for r in rows if (r["image"], r["gt_id"]) in c0_by_key]
        n_all = len(pairs)
        tr_ck_a = np.full(n_all, np.nan); tr_c0_a = np.full(n_all, np.nan)
        fu_ck_a = np.full(n_all, np.nan); fu_c0_a = np.full(n_all, np.nan)
        for k, r in enumerate(pairs):
            base = c0_by_key[(r["image"], r["gt_id"])]
            if r["group"] == "truncated":
                tr_ck_a[k], tr_c0_a[k] = r["miss"], base["miss"]
            else:
                fu_ck_a[k], fu_c0_a[k] = r["miss"], base["miss"]
        dids[cond] = did_bootstrap(tr_ck_a, tr_c0_a, fu_ck_a, fu_c0_a,
                                   arr(pairs, "image"),
                                   np.asarray([r["tissue"] for r in pairs]), n_boot, seed)
    h2 = dids["C2"]
    h2_pass = (h2["did"] >= GATE_H2 and h2["lo"] > 0 and abs(dids["C-int"]["did"]) < h2["did"])

    c2t = [r for r in rec if r["cond"] == "C2" and r["group"] == "truncated"]
    a3 = np.array([r["miss"] if r["frag_area_bin"] == "<30" else np.nan for r in c2t], float)
    b3 = np.array([r["miss"] if r["frag_area_bin"] == ">=100" else np.nan for r in c2t], float)
    h3 = rate_gap_bootstrap(a3, b3, arr(c2t, "image"),
                            np.asarray([r["tissue"] for r in c2t]), n_boot, seed)
    h3_pass = h3["gap"] >= GATE_H3 and h3["lo"] > 0

    # descriptive: per-class border/interior + per-condition miss rates + frag bins
    desc: dict = {"h1_by_class": {}, "miss_by_cond": {}, "frag_bins_C2": {}, "frag_bins_C3": {}}
    for cls in range(1, NUM_CLASSES + 1):
        rows = [r for r in c0 if r["cls"] == cls]
        for grp in ("border", "interior"):
            sel = [r["miss"] for r in rows if r["group"] == grp]
            desc["h1_by_class"][f"cls{cls}_{grp}"] = {"rate": float(np.mean(sel)) if sel else None,
                                                      "n": len(sel)}
    for cond in sorted({r["cond"] for r in rec}):
        sel = [r["miss"] for r in rec if r["cond"] == cond]
        desc["miss_by_cond"][cond] = float(np.mean(sel)) if sel else None
    for cond, key in (("C2", "frag_bins_C2"), ("C3", "frag_bins_C3")):
        for b in ("<30", "30-100", ">=100"):
            sel = [r["miss"] for r in rec if r["cond"] == cond and r["group"] == "truncated"
                   and r["frag_area_bin"] == b]
            desc[key][b] = {"rate": float(np.mean(sel)) if sel else None, "n": len(sel)}

    return {"gates": {"H1": {"dead_gap": h1["gap"], "lo": h1["lo"], "hi": h1["hi"],
                             "pass": bool(h1_pass)},
                      "H2": {"did_c2": h2["did"], "lo": h2["lo"], "hi": h2["hi"],
                             "did_c_int": dids["C-int"]["did"],
                             "did_c_int_lo": dids["C-int"]["lo"], "did_c_int_hi": dids["C-int"]["hi"],
                             "did_c3": dids["C3"]["did"], "did_c3_lo": dids["C3"]["lo"],
                             "did_c3_hi": dids["C3"]["hi"], "pass": bool(h2_pass)},
                      "H3": {**h3, "pass": bool(h3_pass)}},
            "descriptive": desc,
            "n_records": len(rec)}


def predict_frames(model, frames: np.ndarray, device="cuda", batch=32, workers=16):
    """(inst, type, hv) at the 256 working scale for (N,256,256) uint8 frames —
    predict_fold's forward+decode path, no TTA, decode constants untouched."""
    n = len(frames)
    inst_out = np.empty((n, 256, 256), np.int32)
    type_out = np.empty((n, 256, 256), np.uint8)
    hv_out = np.empty((n, 256, 256, 2), np.float32)
    with Pool(workers) as pool:
        for s in range(0, n, batch):
            x = torch.from_numpy(frames[s:s + batch]).to(device)
            p = _forward_probs(model, x, (UNI_MEAN, UNI_STD))
            tp = p["tp"].argmax(-1, keepdim=True).float()
            maps = torch.cat([tp, p["np"][..., 1:], p["hv"]], -1).cpu().numpy()
            res = pool.map(_post, ((m, 1.0, 1.0) for m in maps), chunksize=2)
            for b, r in enumerate(res):
                inst_out[s + b], type_out[s + b] = r
                hv_out[s + b] = maps[b][..., 2:4]
    return inst_out, type_out, hv_out


def _resize_maps(inst, typ, hv, size):
    hw = (size, size)
    ri = cv2.resize(inst, hw, interpolation=cv2.INTER_NEAREST).astype(np.int32)
    rt = cv2.resize(typ, hw, interpolation=cv2.INTER_NEAREST)
    rh = np.stack([cv2.resize(hv[..., k], hw, interpolation=cv2.INTER_LINEAR)
                   for k in range(2)], -1) if size != 256 else hv
    return ri, rt, rh


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, default=ALLOWED_RUN)
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--out", type=Path, default=ROOT / "runs/analysis/visibility_probe_20261013")
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args(argv)
    if a.fold != 2:
        raise SystemExit(f"--fold {a.fold}: E4 reads fold2 only (split1 val; spec §5)")
    if a.run.resolve() != ALLOWED_RUN:
        raise SystemExit(f"--run {a.run} is not the allowlisted {ALLOWED_RUN}")
    cfg = json.loads((a.run / "config.json").read_text())
    if cfg.get("split") != 1:
        raise SystemExit(f"{a.run}/config.json split={cfg.get('split')} != 1; refusing")
    if a.out.exists():
        raise SystemExit(f"{a.out} exists — refusing to overwrite a round directory")

    f = PanNukeFold(2)
    n = len(f)
    pred0 = dict(np.load(ALLOWED_PRED))
    frozen = json.loads(FROZEN_EVAL.read_text())
    t0 = time.time()

    # ---- C0 binding: identity re-eval == frozen eval_fold2 mPQ (bit-exact)
    c0 = evaluate(f.gt_channels, np.asarray(f.inst), np.asarray(f.type), f.tissue,
                  pred0["inst"], pred0["type"], workers=16)["summary"]
    if c0["official"]["mPQ"] != frozen["official"]["mPQ"]:
        raise SystemExit(f"C0 binding failed: {c0['official']['mPQ']!r} != "
                         f"{frozen['official']['mPQ']!r}")

    # ---- condition frames (pure image ops; GT crops are id-preserving)
    w192 = sample_windows(n, 192, WINDOW_SEED)
    w160 = sample_windows(n, 160, WINDOW_SEED)
    frames: dict[str, np.ndarray] = {"C1": np.empty((n, 256, 256), np.uint8),
                                     "C-int": np.empty((n, 256, 256), np.uint8),
                                     "C2": np.empty((n, 256, 256), np.uint8),
                                     "C3": np.empty((n, 256, 256), np.uint8)}
    imgs = np.asarray(f.images)
    for j in range(n):
        im = imgs[j]
        frames["C1"][j] = shift_pad(im, SHIFT)
        small = cv2.resize(im, (192, 192), interpolation=cv2.INTER_LINEAR)
        frames["C-int"][j] = cv2.resize(small, (256, 256), interpolation=cv2.INTER_LINEAR)
        y, x = w192[j]
        frames["C2"][j] = cv2.resize(np.ascontiguousarray(im[y:y + 192, x:x + 192]),
                                     (256, 256), interpolation=cv2.INTER_LINEAR)
        y, x = w160[j]
        frames["C3"][j] = cv2.resize(np.ascontiguousarray(im[y:y + 160, x:x + 160]),
                                     (256, 256), interpolation=cv2.INTER_LINEAR)

    # ---- inference on the frozen checkpoint (no TTA, decode constants untouched)
    ck_cfg = TrainConfig(split=1, out_dir=str(a.run), dead_expert=run_dead_expert(a.run),
                         widen=run_widen(a.run))
    model = build_model(ck_cfg, pretrained=False).cuda()
    model.load_state_dict(torch.load(ALLOWED_CKPT, map_location="cpu", weights_only=False)["model"])
    preds = {cond: predict_frames(model, fr, batch=a.batch_size)
             for cond, fr in frames.items()}
    del model
    torch.cuda.empty_cache()

    # ---- per-nucleus records
    rec: list[dict] = []
    inst_all = np.asarray(f.inst)
    type_all = np.asarray(f.type)
    sec: dict = {"split_merge": {}, "type_acc": {}, "mpq_bpq": {}, "hv_err": {}}
    for j in range(n):
        tis = str(f.tissue[j])
        inst_j = np.asarray(inst_all[j])
        cls_map = gt_class_map(inst_j, np.asarray(type_all[j]))
        areas = np.bincount(inst_j.ravel())
        ofull = original_full(inst_j)
        # C0 (frozen predictions)
        miss0 = miss_table(inst_j, pred0["inst"][j])
        for i, m in miss0.items():
            rec.append(dict(image=j, tissue=tis, gt_id=i, cls=cls_map[i],
                            area_native=int(areas[i]), original_full=bool(ofull[i]),
                            cond="C0", group="interior" if ofull[i] else "border",
                            frag_area_bin="n/a", miss=bool(m)))
        # C1 (phase shift; same frame, same nuclei)
        sh_inst = shift_pad(inst_j, SHIFT)
        miss1 = miss_table(sh_inst, preds["C1"][0][j])
        for i, m in miss1.items():
            rec.append(dict(image=j, tissue=tis, gt_id=i, cls=cls_map[i],
                            area_native=int(areas[i]), original_full=bool(ofull[i]),
                            cond="C1", group="interior" if ofull[i] else "border",
                            frag_area_bin="n/a", miss=bool(m)))
        # window groupings (original-full nuclei only enter H2/H3 rows)
        g192 = crop_group(inst_j, (int(w192[j][0]), int(w192[j][1]), 192))
        g160 = crop_group(inst_j, (int(w160[j][0]), int(w160[j][1]), 160))
        # C-int (interpolation floor; groups from the 192 window per spec H2 calibration)
        missi = miss_table(inst_j, preds["C-int"][0][j])
        for i, grp in g192.items():
            if not ofull[i] or grp["group"] == "outside":
                continue
            rec.append(dict(image=j, tissue=tis, gt_id=i, cls=cls_map[i],
                            area_native=int(areas[i]), original_full=True,
                            cond="C-int", group=grp["group"], frag_area_bin="n/a",
                            miss=bool(missi[i])))
        # C2 / C3 (crop-frame evaluation, id-preserving GT crops)
        for cond, wk, gmap, size in (("C2", w192, g192, 192), ("C3", w160, g160, 160)):
            y, x = wk[j]
            gt_crop = inst_j[y:y + size, x:x + size]
            pi, pt, ph = _resize_maps(preds[cond][0][j], preds[cond][1][j], preds[cond][2][j], size)
            missk = miss_table(gt_crop, pi)
            u = max_iou_table(gt_crop, pi)
            for i, grp in gmap.items():
                if not ofull[i] or grp["group"] == "outside":
                    continue
                rec.append(dict(image=j, tissue=tis, gt_id=i, cls=cls_map[i],
                                area_native=int(areas[i]), original_full=True,
                                cond=cond, group=grp["group"],
                                frag_area_bin=bin_area(grp["area"]) if grp["group"] == "truncated"
                                else "n/a", miss=bool(missk[i])))
            # secondary: hv error inside truncated fragments (diagnostic only)
            ideal = hv_targets(gt_crop, min_size=0)
            frag = np.isin(gt_crop, [i for i, gp in gmap.items()
                                     if gp["group"] == "truncated" and ofull[i]])
            if frag.any():
                sec["hv_err"].setdefault(cond, []).append(float(np.abs(ph[frag] - ideal[frag]).mean()))
            # secondary: split/merge + matched type accuracy in the crop frame
            ov = overlap(gt_crop, pi)
            gs, ps, _ = classify_errors(ov)
            sec["split_merge"].setdefault(cond, {}).setdefault("split", 0)
            sec["split_merge"][cond]["split"] += int((gs == 2).sum())
            sec["split_merge"][cond]["merged"] = sec["split_merge"][cond].get("merged", 0) + int((gs == 1).sum())
            _, p_cls = instance_classes(pi, pt)
            iou = ov.iou
            gi, pii = np.nonzero(iou > 0.5)
            if len(gi):
                g_ids = np.unique(gt_crop); g_ids = g_ids[g_ids > 0]
                acc = [cls_map[int(g_ids[g])] == int(p_cls[pii[k]]) for k, g in enumerate(gi)]
                sec["type_acc"].setdefault(cond, []).append(float(np.mean(acc)))

    # secondary: C0/C1/C-int split/merge + type acc on the 256 frame
    for cond, gt_i, pi, pt in (("C0", None, pred0["inst"], pred0["type"]),
                               ("C1", "shift", preds["C1"][0], preds["C1"][1]),
                               ("C-int", None, preds["C-int"][0], preds["C-int"][1])):
        sp = me = 0
        accs = []
        for j in range(n):
            gtj = shift_pad(np.asarray(inst_all[j]), SHIFT) if gt_i == "shift" else np.asarray(inst_all[j])
            clsj = gt_class_map(gtj, np.asarray(type_all[j]))
            ov = overlap(gtj, pi[j])
            gs, _, _ = classify_errors(ov)
            sp += int((gs == 2).sum()); me += int((gs == 1).sum())
            _, p_cls = instance_classes(pi[j], pt[j])
            gi, pii = np.nonzero(ov.iou > 0.5)
            if len(gi):
                g_ids = np.unique(gtj); g_ids = g_ids[g_ids > 0]
                accs += [clsj[int(g_ids[g])] == int(p_cls[pii[k]]) for k, g in enumerate(gi)]
        sec["split_merge"][cond] = {"split": sp, "merged": me}
        sec["type_acc"][cond] = float(np.mean(accs)) if accs else None

    # secondary: per-condition image-level mPQ/bPQ (crop frame for C2/C3)
    for cond in ("C1", "C-int", "C2", "C3"):
        if cond == "C1":
            gt_ch = np.stack([shift_pad(np.asarray(f.gt_channels[..., c]), SHIFT)
                              for c in range(NUM_CLASSES)], -1)
            gi, gt_, pi, pt = shift_pad(inst_all, SHIFT), shift_pad(type_all, SHIFT), \
                preds["C1"][0], preds["C1"][1]
        elif cond == "C-int":
            gt_ch, gi, gt_, pi, pt = f.gt_channels, inst_all, type_all, \
                preds["C-int"][0], preds["C-int"][1]
        else:
            wk = w192 if cond == "C2" else w160
            size = 192 if cond == "C2" else 160
            gt_ch = np.empty((n, size, size, NUM_CLASSES), np.int32)
            gi = np.empty((n, size, size), np.int32)
            gt_ = np.empty((n, size, size), np.uint8)
            pi = np.empty((n, size, size), np.int32)
            pt = np.empty((n, size, size), np.uint8)
            for j in range(n):
                y, x = wk[j]
                gt_ch[j] = np.asarray(f.gt_channels[j])[y:y + size, x:x + size]
                gi[j] = inst_all[j][y:y + size, x:x + size]
                gt_[j] = type_all[j][y:y + size, x:x + size]
                pi[j], pt[j], _ = _resize_maps(preds[cond][0][j], preds[cond][1][j],
                                               preds[cond][2][j], size)
        s = evaluate(gt_ch, gi, gt_, f.tissue, np.asarray(pi), np.asarray(pt),
                     workers=16)["summary"]
        sec["mpq_bpq"][cond] = {"mPQ": s["official"]["mPQ"], "bPQ": s["official"]["bPQ"]}
        print(f"{cond}: mPQ {s['official']['mPQ']:.4f} bPQ {s['official']['bPQ']:.4f}", flush=True)
    sec["mpq_bpq"]["C0"] = {"mPQ": c0["official"]["mPQ"], "bPQ": c0["official"]["bPQ"]}
    sec["hv_err"] = {k: float(np.mean(v)) for k, v in sec["hv_err"].items()}
    sec["type_acc"] = {k: (float(np.mean(v)) if isinstance(v, list) else v)
                       for k, v in sec["type_acc"].items()}

    out = analyze_records(rec, N_BOOT, BOOT_SEED)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "probe.json").write_text(json.dumps(
        {"gates": out["gates"], "descriptive": out["descriptive"], "secondary": sec,
         "n_records": out["n_records"], "windows": {"192": w192.tolist(), "160": w160.tolist()},
         "window_seed": WINDOW_SEED, "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
         "gate_numbers": {"H1": GATE_H1, "H2": GATE_H2, "H3": GATE_H3},
         "provenance": {"ckpt": file_provenance(ALLOWED_CKPT),
                        "pred_fold2": file_provenance(ALLOWED_PRED),
                        "frozen_eval": file_provenance(FROZEN_EVAL),
                        "gt_fold2_inst": file_provenance(f.dir / "inst.npy"),
                        "script": file_provenance(Path(__file__))["sha256"],
                        "module": file_provenance(ROOT / "src/nucseg/visibility_probe.py")["sha256"]}},
        indent=2))
    lines = ["# E4 visibility probe (spec 2026-10-13; gates frozen)", "",
             f"H1 Dead border-interior gap {out['gates']['H1']['gap']:+.4f} "
             f"[{out['gates']['H1']['lo']:+.4f},{out['gates']['H1']['hi']:+.4f}] "
             f"pass={out['gates']['H1']['pass']}",
             f"H2 DiD(C2) {out['gates']['H2']['did_c2']:+.4f} "
             f"[{out['gates']['H2']['lo']:+.4f},{out['gates']['H2']['hi']:+.4f}] "
             f"| DiD(C-int) {out['gates']['H2']['did_c_int']:+.4f} "
             f"| DiD(C3) {out['gates']['H2']['did_c3']:+.4f} pass={out['gates']['H2']['pass']}",
             f"H3 <30 vs >=100 px gap {out['gates']['H3']['gap']:+.4f} "
             f"[{out['gates']['H3']['lo']:+.4f},{out['gates']['H3']['hi']:+.4f}] "
             f"pass={out['gates']['H3']['pass']}", "",
             "ALL THREE PASS -> visibility-supervision training round may be drafted;",
             "ANY FAIL -> route A closes per pre-registered stop rule.", ""]
    (a.out / "tables.md").write_text("\n".join(lines) + "\n")
    (a.out / "commands.md").write_text(
        f"# commands\n\n- host: {__import__('socket').gethostname()}\n"
        f"- cmd: `python scripts/probe_visibility.py --run {a.run} --fold 2 --out {a.out}`\n"
        f"- env: ~/.conda/envs/nuclei (numpy {np.__version__}, torch {torch.__version__})\n"
        f"- wall: {time.time() - t0:.1f}s\n- C0 binding: mPQ bit-exact vs eval_fold2\n"
        f"- decode constants untouched (decode_u=1, marker_u=1, min_size 10 at 256);\n"
        f"  C2/C3 zoom makes them relatively smaller in tissue terms (disclosed, no du\n"
        f"  compensation — DiD's stays-full control shares the same decode)\n")
    print("\n".join(lines))
    print(f"\nwrote {a.out}/probe.json")


if __name__ == "__main__":
    main()
