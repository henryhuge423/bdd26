#!/usr/bin/env python
"""E3 候选质量学习 dev 轮 runner（spec 2026-10-11，冻结命令见 plan Task 5）。

五臂菜单 A0/A1/A2/A3/A4 在 split1_seed1 冻结 fold2 候选池上重放；A1 行与冻结 P2
selection.json 的 selected 行硬断言（≤1e-12）；判定门数字随 spec §5 冻结。fold3 零读取；
训练只用 fold1 OOF 行（split2 三种子）。全部输入 SHA + 代码指纹入 provenance。
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fold_guard import ensure_dev_fold  # noqa: E402
from lever_postproc import file_provenance  # noqa: E402
from matched_seed_stats import bootstrap as ms_bootstrap  # noqa: E402
from run_p0_controls import check_prediction  # noqa: E402
from run_scale_fusion import image_confidence  # noqa: E402

from nucseg.candidate_quality.features import (FEATURE_NAMES, design_matrix,  # noqa: E402
                                               select_training_rows)
from nucseg.candidate_quality.menu import a1_allowed, build_menu, selection_rows  # noqa: E402
from nucseg.candidate_quality.model import (fit_logistic, predict, reliability,  # noqa: E402
                                            torch_head_directions)
from nucseg.candidate_quality.support import (effective_denominator, jury_paths_fold1,  # noqa: E402
                                              support_matrix)
from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.metrics import light  # noqa: E402
from nucseg.metrics.pannuke_eval import _pq  # noqa: E402
from nucseg.postproc.existence import existence_pass  # noqa: E402
from nucseg.postproc.scale_fusion import (_gt_facts, candidate_info, fuse,  # noqa: E402
                                          probability_rows, select_fusion, strict_dead_pq)
from nucseg.postproc import scale_fusion as scale_fusion_mod  # noqa: E402

PAIR = "split1_seed1"
SEED = 20261011
N_BOOT = 2000
EF_FLOOR = 30
EF_VARIANTS = {"off": (0, False), "a30": (30, False), "a30d": (30, True), "a60": (60, False)}
PRIMARY = {"d_dead": 2e-3, "d_bpq": -5e-4, "d_strict": -2e-3}
SECONDARY = {"d_dead_abs": 1e-3, "d_bpq": 5e-4, "unmatched_frac": 0.75}
GATED_ARMS = ("A2", "A3", "A4")
_G: dict = {}


def build_menu_loaded(rec_apply: dict, p_low: np.ndarray, p_sup: np.ndarray,
                      p2cfg: dict, ef_floor: int = EF_FLOOR,
                      dead_exempt: bool = False) -> list[dict]:
    """menu.build_menu + 注入 A1 的现任接受集（EF 变体语义）。"""
    rows = build_menu(rec_apply, p_low, p_sup)
    allowed, _ = a1_allowed(rec_apply, p2cfg, ef_floor=ef_floor, dead_exempt=dead_exempt)
    for r in rows:
        if r["name"] == "a1_incumbent":
            r["allowed"] = allowed
    return rows


def sweep(menu: list[dict], bi, bt, xi, xt, prob, gc, gi, gt, tissue, p2cfg: dict,
          workers: int = 1, ef_floor: int = EF_FLOOR, dead_exempt: bool = False) -> list[dict]:
    """逐图重放 18 行菜单，返回每行端点（含 per-image stats 供 bootstrap）。"""
    per_img = []
    for row in menu:
        d: dict[int, set] = {}
        for im, i in row["allowed"]:
            d.setdefault(im, set()).add(i)
        per_img.append(d)
    _G.update(menu=menu, per_img=per_img, bi=bi, bt=bt, xi=xi, xt=xt, prob=prob,
              gc=gc, gi=gi, gt=gt, p2cfg=p2cfg, ef_floor=ef_floor, dead_exempt=dead_exempt)
    try:
        if workers > 1:
            with Pool(workers) as pool:
                results = pool.map(_sweep_image, range(len(gt)), chunksize=4)
        else:
            results = [_sweep_image(j) for j in range(len(gt))]
    finally:
        _G.clear()
    rows = []
    for k, row in enumerate(menu):
        stats = np.stack([r[k][0] for r in results])
        s = light.summarize(stats, tissue)
        rows.append({**{kk: row[kk] for kk in ("name", "arm")},
                     "mPQ": s["mPQ"], "bPQ": s["bPQ"], "Dead_PQ": s["per_class_PQ"]["Dead"],
                     "per_class_PQ": s["per_class_PQ"],
                     "strict_dead": strict_dead_pq(stats),
                     "interior_matched": int(sum(r[k][1] for r in results)),
                     "added": int(sum(r[k][2] for r in results)),
                     "_stats": stats})
    return rows


def _sweep_image(j: int):
    bi, bt, xi, xt = (_G[k][j] for k in ("bi", "bt", "xi", "xt"))
    conf = image_confidence(xi, xt, _G["prob"].get(j, {}))
    cands = candidate_info(bi, xi, xt)
    out = []
    for k, row in enumerate(_G["menu"]):
        name = row["name"]
        if name == "identity":
            pi, pt, added = bi, bt, []
        elif name == "a1_incumbent":
            cfg = _G["p2cfg"]
            allowed = {c["id"] for c in cands
                       if existence_pass(c, _G["ef_floor"], _G["dead_exempt"])}
            pi, pt, added = fuse(bi, bt, xi, xt, conf, cfg["max_area"], cfg["min_prob"],
                                 cfg["interior_only"], allowed_ids=allowed)
        else:
            allowed = _G["per_img"][k].get(j)
            pi, pt, added = fuse(bi, bt, xi, xt, conf, None, 0, False,
                                 allowed_ids=allowed if allowed is not None else set())
        stats = light.image_stats(_G["gc"][j], _G["gi"][j], _G["gt"][j], pi, pt)
        counts = _gt_facts(_G["gi"][j], _G["gt"][j], pi, pt)[-1]
        out.append((stats, counts["interior_matched"], len(added)))
    return out


def _per_image_triples(stats: np.ndarray) -> np.ndarray:
    """(n_img, 3)：[mPQ 口径类均值, 二值 PQ, strict 口径 Dead PQ]。

    mPQ 列按官方聚合：仅平均 GT 在场类（s[4]>0）的 PQ；GT 缺席即使有预测也不计入
    （官方对缺类置 NaN 后排除，非计 0）。Dead 列沿用 strict_dead_pq 口径：GT 缺席但
    有预测计 0（这正是 strict 惩罚所在）。"""
    n = stats.shape[0]
    trip = np.full((n, 3), np.nan)
    for j in range(n):
        per_cls = [float(_pq(*s[:4])) for s in stats[j][1:] if s[4] > 0]
        if per_cls:
            trip[j, 0] = float(np.mean(per_cls))
        b = stats[j][0]
        if b[0] + b[1] + b[2] > 0:
            trip[j, 1] = float(_pq(*b[:4]))
        s = stats[j][4]
        if s[4] > 0:
            trip[j, 2] = float(_pq(*s[:4]))
        elif s[0] + s[1] > 0:
            trip[j, 2] = 0.0
    return trip


def evaluate_gates(sel_rows: dict, a1: dict) -> dict:
    """spec §5：主门（Dead 增益）与次门（FP 精简），全部相对 A1 选中行。"""
    out = {}
    for arm, r in sel_rows.items():
        d_dead = r["Dead_PQ"] - a1["Dead_PQ"]
        d_bpq = r["bPQ"] - a1["bPQ"]
        d_strict = r["strict_dead"] - a1["strict_dead"]
        out[arm] = {
            "deltas": {"Dead_PQ": d_dead, "bPQ": d_bpq, "strict_dead": d_strict,
                       "mPQ": r["mPQ"] - a1["mPQ"]},
            "primary": bool(d_dead >= PRIMARY["d_dead"] and d_bpq >= PRIMARY["d_bpq"]
                            and d_strict >= PRIMARY["d_strict"]),
            "secondary": bool(abs(d_dead) <= SECONDARY["d_dead_abs"]
                              and d_bpq >= SECONDARY["d_bpq"]
                              and r["unmatched"] <= SECONDARY["unmatched_frac"] * a1["unmatched"]),
        }
    return out


def _unmatched_of(rec_apply: dict, allowed: set) -> int:
    keys = list(zip(np.asarray(rec_apply["image"]).tolist(), np.asarray(rec_apply["id"]).tolist()))
    matched = np.asarray(rec_apply["matched"], bool)
    return int(sum(1 for k, m in zip(keys, matched) if k in allowed and not m))


def _real_loader(a) -> dict:
    rec_full = dict(np.load(a.records))
    sel = np.asarray(rec_full["pair"]) == PAIR
    rec_apply = dict(np.load(a.apply_records))
    for col in ("image", "id", "area", "pmax", "border", "class", "matched"):
        if not np.array_equal(np.asarray(rec_apply[col]), np.asarray(rec_full[col])[sel]):
            raise SystemExit(f"apply records column {col} does not bind to the v2 subset")
    rec_train = select_training_rows(rec_full)
    if len(rec_train["pair"]) != 13332:
        raise SystemExit(f"expected 13,332 fold1 training rows, got {len(rec_train['pair'])}")
    jury = jury_paths_fold1(a.round, a.hn_fold1)
    sup_train, excl = support_matrix(rec_train, a.round, jury)
    den = effective_denominator(excl)
    with np.load(a.fold2_support) as z:
        iou13, jury_names = z["iou"], [str(s) for s in z["jury_names"]]
    if iou13.shape[0] != len(rec_apply["pair"]):
        raise SystemExit("fold2 support rows do not align with apply records")
    sup_apply = (iou13 >= 0.3).sum(1)

    fold = PanNukeFold(2)
    gi, gt, gc, tissue = np.asarray(fold.inst), np.asarray(fold.type), fold.gt_channels, fold.tissue
    with np.load(a.base_val) as b, np.load(a.x2_val) as x:
        bi, bt = b["inst"], b["type"]
        xi, xt = x["inst"], x["type"]
        check_prediction(bi, bt, fold, b)
        check_prediction(xi, xt, fold, x)
        prob = probability_rows(x)
    return {"bi": bi, "bt": bt, "xi": xi, "xt": xt, "gi": gi, "gt": gt, "gc": gc,
            "tissue": tissue, "prob": prob, "rec_train": rec_train,
            "sup_train": sup_train, "sup_excl_den": den, "rec_apply": rec_apply,
            "sup_apply": sup_apply, "jury_names_fold2": jury_names,
            "jury_fold1": [j["name"] for j in jury]}


ROOT = Path(__file__).resolve().parents[1]


def code_fingerprints() -> dict:
    """SHA256 of every project module the round's numbers depend on (beyond this script)."""
    import nucseg.candidate_quality.features as cq_features
    import nucseg.candidate_quality.support as cq_support
    import nucseg.candidate_quality.model as cq_model
    import nucseg.candidate_quality.menu as cq_menu
    import nucseg.postproc.existence as pp_existence
    mods = (cq_features, cq_support, cq_model, cq_menu, scale_fusion_mod, pp_existence)
    out = {}
    for m in mods:
        p = Path(m.__file__)
        out[str(p.relative_to(ROOT))] = file_provenance(p)["sha256"]
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def main(argv=None, loader=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--pair", default=PAIR)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--apply-records", type=Path, required=True, dest="apply_records")
    ap.add_argument("--base-val", type=Path, required=True, dest="base_val")
    ap.add_argument("--x2-val", type=Path, required=True, dest="x2_val")
    ap.add_argument("--p2-selection", type=Path, required=True, dest="p2_selection")
    ap.add_argument("--ef-selection", type=Path, required=True, dest="ef_selection")
    ap.add_argument("--fold2-support", type=Path, required=True, dest="fold2_support")
    ap.add_argument("--hn-fold1", type=Path, default=None, dest="hn_fold1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=N_BOOT, dest="n_boot")
    a = ap.parse_args(argv)
    if a.pair != PAIR:
        raise SystemExit(f"only the frozen pair {PAIR} is supported")
    ensure_dev_fold(2, 1)
    if (a.out / "results.json").exists():
        raise SystemExit(f"{a.out / 'results.json'} exists; refusing to overwrite")
    p2 = json.loads(a.p2_selection.read_text())
    p2cfg = p2["selected"]
    if p2cfg.get("name") == "identity" or "max_area" not in p2cfg:
        raise SystemExit("frozen P2 selection is identity — E3 dev round undefined")
    ef = json.loads(a.ef_selection.read_text())
    ef_sel = ef["selected"]
    if ef_sel["name"] not in EF_VARIANTS:
        raise SystemExit(f"unsupported frozen EF variant {ef_sel['name']}")
    ef_floor, dead_exempt = EF_VARIANTS[ef_sel["name"]]

    data = (loader or _real_loader)(a)
    rec_apply = data["rec_apply"]
    if data.get("rec_train") is not None:
        X, y = design_matrix(data["rec_train"])
        groups = np.asarray(data["rec_train"]["image"])
        sup = data["sup_train"].sum(1).astype(float)
        frac = sup / data["sup_excl_den"]
        X_sup = np.column_stack([X, sup, frac])
        fit_low = fit_logistic(X, y, groups)
        fit_sup = fit_logistic(X_sup, y, groups)
        Xa, _ = design_matrix(rec_apply)
        sa = data["sup_apply"].astype(float)
        p_low, p_sup = predict(fit_low, Xa), predict(fit_sup, np.column_stack([Xa, sa, sa / len(data["jury_names_fold2"])]))
        model_meta = {"C_low": fit_low["C"], "cv_auc_low": fit_low["cv_auc"],
                      "C_sup": fit_sup["C"], "cv_auc_sup": fit_sup["cv_auc"],
                      "features_low": FEATURE_NAMES,
                      "features_sup": FEATURE_NAMES + ["support", "support_frac"]}
    else:
        p_low, p_sup = data["p_low"], data["p_sup"]
        model_meta = {"mocked": True}

    menu = build_menu_loaded(rec_apply, p_low, p_sup, p2cfg, ef_floor, dead_exempt)
    rows = sweep(menu, data["bi"], data["bt"], data["xi"], data["xt"], data["prob"],
                 data["gc"], data["gi"], data["gt"], data["tissue"], p2cfg,
                 workers=a.workers, ef_floor=ef_floor, dead_exempt=dead_exempt)
    for r in rows:
        r["unmatched"] = _unmatched_of(rec_apply, next(m["allowed"] for m in menu if m["name"] == r["name"]))
    a1 = next(r for r in rows if r["name"] == "a1_incumbent")
    for k in ("mPQ", "bPQ", "strict_dead", "interior_matched", "added"):
        if abs(a1[k] - ef_sel[k]) > 1e-12:
            raise SystemExit(f"A1 replay drifted on '{k}': {a1[k]} vs frozen {ef_sel[k]}")

    sel = selection_rows(rows)
    gates = evaluate_gates({k: sel[k] for k in GATED_ARMS if k in sel}, a1)

    boot = {}
    for arm in GATED_ARMS:
        if arm not in sel:
            continue
        delta = _per_image_triples(sel[arm]["_stats"]) - _per_image_triples(a1["_stats"])
        boot[arm] = ms_bootstrap({"sel": delta}, data["tissue"], a.n_boot, SEED)

    diag = {"reliability_low": reliability(np.asarray(rec_apply["matched"], float), p_low),
            "reliability_sup": reliability(np.asarray(rec_apply["matched"], float), p_sup)}
    if data.get("rec_train") is not None:
        from scipy.stats import spearmanr
        outs = torch_head_directions(X, y, Xa, seeds=(0, 1))
        diag["torch_spearman_vs_sklearn"] = [float(spearmanr(o, p_low).statistic) for o in outs]
    a0 = next(r for r in rows if r["name"] == "identity")
    util = []
    cls = np.asarray(rec_apply["class"])
    iou_b = np.asarray(rec_apply["iou_best"], float)
    mat = np.asarray(rec_apply["matched"], bool)
    for c in range(1, 6):
        m = cls == c
        e_iou = float(iou_b[m & mat].mean()) if (m & mat).any() else float("nan")
        util.append({"class": int(c), "n": int(m.sum()), "mean_p_low": float(p_low[m].mean()),
                     "E_iou_given_matched": e_iou,
                     "p_times_E_iou": float(p_low[m].mean() * e_iou) if e_iou == e_iou else float("nan")})
    diag["utility_vs_q2"] = util
    mix = {}
    for arm in ("A1",) + GATED_ARMS:
        allowed = next(m["allowed"] for m in menu if m["name"] == sel[arm]["name"])
        keys = list(zip(np.asarray(rec_apply["image"]).tolist(), np.asarray(rec_apply["id"]).tolist()))
        idx = [k for k in keys if k in allowed]
        cm = np.asarray(rec_apply["class"]); mm = np.asarray(rec_apply["matched"], bool)
        sel_idx = [keys.index(k) for k in idx]
        mix[arm] = {"added": len(idx), "unmatched": int((~mm[sel_idx]).sum()) if sel_idx else 0,
                    "by_class": {int(c): int((cm[sel_idx] == c).sum()) for c in range(1, 6)} if sel_idx else {}}
    diag["added_class_mix"] = mix

    res = {"rows": [{k: v for k, v in r.items() if k != "_stats"} for r in rows],
           "selected": {k: {kk: vv for kk, vv in v.items() if kk != "_stats"} for k, v in sel.items()},
           "gates": gates, "bootstrap": boot, "model": model_meta, "diagnostics": diag,
           "params": {"seed": SEED, "n_boot": a.n_boot, "ef_variant": ef_sel["name"],
                      "ef_floor": ef_floor, "dead_exempt": dead_exempt,
                      "primary": PRIMARY, "secondary": SECONDARY, "pair": PAIR,
                      "identity_class_pq": a0["per_class_PQ"]},
           "provenance": ({"inputs": {kk: file_provenance(getattr(a, kk))["sha256"] for kk in
                            ("records", "apply_records", "base_val", "x2_val", "p2_selection",
                             "ef_selection", "fold2_support", "hn_fold1") if getattr(a, kk) is not None},
                          "script": file_provenance(Path(__file__))["sha256"],
                          "code": code_fingerprints(),
                          "spec": "docs/superpowers/specs/2026-10-11-e3-candidate-quality-design.md"}
                         if loader is None else {"mocked": True})}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(_jsonable(res), indent=1))
    verdict = {arm: (g["primary"] or g["secondary"]) for arm, g in gates.items()}
    print(f"E3 dev: gates={verdict} A1 added={a1['added']} deadPQ={a1['Dead_PQ']:.4f}", flush=True)


if __name__ == "__main__":
    main()
