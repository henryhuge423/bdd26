#!/usr/bin/env python
"""E1-T：跨检测器分型共识审计（spec 2026-10-11）。

    python scripts/e1t_typing_consensus.py \
        --round runs/analysis/matched_seed_20261004 \
        --extra-jury runs/analysis/e1_consensus_20261010/hovernet_split1_fold2/pred.npz:HoverNet:hovernet:split1 \
                     runs/analysis/e1_consensus_20261010/hovernet_split3_fold2/pred.npz:HoverNet:hovernet:split3 \
        --out runs/analysis/e1t_typing_20261011

基 = split1_seed1 val base identity（x1）。Jury = E1-C 13 席 − 基 run（split1_seed1-x1）= 12 席
（CellViT 10 + HoVer-Net 2；来源 run split1_seed1-x2 已不在枚举内）。只读 fold2；无新推理。
统计量与预注册解读键见 spec §2；全部输出为检测器共识证据，不作生物学断言。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fold_guard import ensure_dev_fold  # noqa: E402
from lever_postproc import file_provenance  # noqa: E402

from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.e1_typing import clustered_rate_bootstrap, jury_votes, majority_type  # noqa: E402
from nucseg.metrics.instance import overlap  # noqa: E402
from nucseg.metrics.pannuke_eval import instance_classes  # noqa: E402

SEED = 20261011
N_BOOT = 2000
PAIR_THR = 0.5   # base-GT 匹配（审计口径 IoU > .5）
VOTE_THR = 0.3   # jury 票口径（E1-C：IoU >= .3）
MIN_VOTES = 3
PAIR = "split1_seed1"


def builtin_jury(round_root: Path) -> list[dict]:
    """E1-C 的 12 席枚举减去来源 run（split1_seed1-x2）再减去基 run（split1_seed1-x1）。"""
    out = []
    for split in (1, 3):
        for seed in (19, 1, 2):
            for scale, arm in (("x1", "base"), ("x2", "x2")):
                if (split, seed) == (1, 1):
                    continue  # 基 run（x1）与来源 run（x2）同 pair，一并排除
                out.append({
                    "name": f"cellvit_s{split}_e{seed}_{scale}",
                    "path": str(round_root / "p0" / f"split{split}_seed{seed}" / "val" / arm
                             / "identity" / "pred.npz"),
                    "arch": "CellViT", "train_fold": split, "seed": seed, "scale": scale})
    return out


def pair_base_detailed(gt: np.ndarray, gt_type: np.ndarray, base: np.ndarray,
                        base_type: np.ndarray, thr: float = PAIR_THR) -> list[dict]:
    """GT id -> 匹配状态与两侧类别；并列最大（>thr）记 ambiguous，不悄悄取先者。"""
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    base_ids = np.unique(base)
    base_ids = base_ids[base_ids > 0]
    _, base_classes = instance_classes(base, base_type)
    ov = overlap(gt, base)
    rows = []
    for g, gid in enumerate(gt_ids):
        tvals = np.unique(gt_type[gt == gid])
        if len(tvals) != 1 or tvals[0] == 0:
            raise ValueError(f"GT instance {int(gid)} has invalid type values {tvals.tolist()}")
        row = {"gt_id": int(gid), "gt_class": int(tvals[0]), "base_id": None, "iou": 0.0,
               "base_class": None, "status": "unmatched"}
        col = ov.iou[g]
        if col.size and col.max() > thr:
            j = int(np.argmax(col))
            ties = np.flatnonzero(np.isclose(col, col[j], rtol=0.0, atol=1e-12))
            if len(ties) > 1:
                row["status"] = "ambiguous"
            else:
                # canonical pixel-majority class; 0 = untyped（合法的“分错”形态）
                row.update(status="matched", base_id=int(base_ids[j]), iou=float(col[j]),
                           base_class=int(base_classes[j]))
        rows.append(row)
    return rows


def _new_counters() -> dict:
    return {"n": 0, "maj_eq_gt": 0, "maj_eq_base": 0, "maj_other": 0,
            "maj_neq_gt": 0, "no_majority": 0}


def _tabulate(records: list[dict], folds: dict[int, int], min_votes: int) -> dict:
    overall = {"G1": _new_counters(), "G0": _new_counters()}
    by_pair: dict[str, dict] = {}
    for r in records:
        maj, _ = majority_type(r["votes"], folds, min_votes=min_votes)
        if r["status"] != "matched":
            continue
        g1 = r["gt_class"] != r["base_class"]
        cells = [overall["G1" if g1 else "G0"]]
        if g1:
            key = f"{r['gt_class']},{r['base_class']}"
            by_pair.setdefault(key, _new_counters())
            cells.append(by_pair[key])
        for c in cells:
            c["n"] += 1
            if maj is None:
                c["no_majority"] += 1
            elif g1:
                if maj == r["gt_class"]:
                    c["maj_eq_gt"] += 1
                elif maj == r["base_class"]:
                    c["maj_eq_base"] += 1
                else:
                    c["maj_other"] += 1
            else:
                if maj == r["gt_class"]:
                    c["maj_eq_gt"] += 1
                else:
                    c["maj_neq_gt"] += 1
    return {"overall": overall, "by_pair": by_pair}


def _bootstrap_rates(records: list[dict], n_boot: int, seed: int) -> dict:
    def rate_ci(group, event):
        # 条件率：组内事件 1 / 组内非事件 0 / 组外 NaN（不入分母）
        ind = np.array([float(bool(event(r))) if group(r) else np.nan for r in records])
        img = np.array([r["img"] for r in records])
        if np.isnan(ind).all():
            return (float("nan"), float("nan"))
        return clustered_rate_bootstrap(ind, img, n_boot=n_boot, seed=seed)

    g1 = lambda r: r["status"] == "matched" and r["gt_class"] != r["base_class"]
    g0 = lambda r: r["status"] == "matched" and r["gt_class"] == r["base_class"]
    out = {
        "G1_maj_eq_base": rate_ci(g1, lambda r: r["_maj"] == r["base_class"]),
        "G1_maj_eq_gt": rate_ci(g1, lambda r: r["_maj"] == r["gt_class"]),
        "G0_maj_neq_gt": rate_ci(g0, lambda r: r["_maj"] not in (None, r["gt_class"])),
        "G1_no_majority": rate_ci(g1, lambda r: r["_maj"] is None),
    }
    for key in ("2,3", "3,2"):
        k = key
        out[f"pair_{key}_maj_eq_base"] = rate_ci(
            lambda r: g1(r) and f"{r['gt_class']},{r['base_class']}" == k,
            lambda r: r["_maj"] == r["base_class"])
    dead = lambda r: g1(r) and (r["gt_class"] == 4 or r["base_class"] == 4)
    out["pair_dead_maj_eq_base"] = rate_ci(dead, lambda r: r["_maj"] == r["base_class"])
    out["pair_dead_maj_eq_gt"] = rate_ci(dead, lambda r: r["_maj"] == r["gt_class"])
    return out


def analyze(gt_inst: np.ndarray, gt_type: np.ndarray, base_inst: np.ndarray,
            base_type: np.ndarray, jury: list[dict], *, pair_thr: float = PAIR_THR,
            vote_thr: float = VOTE_THR, min_votes: int = MIN_VOTES,
            n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """End-to-end 统计：records → 四格计数 + by_pair + 聚类 bootstrap + 敏感性三套。"""
    n_img = gt_inst.shape[0]
    records: list[dict] = []
    for j in range(n_img):
        gt = gt_inst[j]
        rows = pair_base_detailed(gt, gt_type[j], base_inst[j], base_type[j], pair_thr)
        votes_j: dict[int, dict[int, int]] = {}
        for s, seat in enumerate(jury):
            for gid, vote in jury_votes(gt, seat["inst"][j], seat["type"][j], thr=vote_thr).items():
                if vote is not None:
                    votes_j.setdefault(gid, {})[s] = vote[1]
        for r in rows:
            r["img"] = j
            r["votes"] = votes_j.get(r["gt_id"], {})
        records.extend(rows)
    folds = {s: seat["train_fold"] for s, seat in enumerate(jury)}
    for r in records:
        r["_maj"], _ = majority_type(r["votes"], folds, min_votes=min_votes)

    tab = _tabulate(records, folds, min_votes)
    groups = {"G1": tab["overall"]["G1"]["n"], "G0": tab["overall"]["G0"]["n"],
              "ambiguous": sum(r["status"] == "ambiguous" for r in records),
              "unmatched_gt": sum(r["status"] == "unmatched" for r in records)}
    boot = _bootstrap_rates(records, n_boot, seed)

    sens: dict = {}
    hn_idx = {s for s, seat in enumerate(jury) if seat["arch"] != "CellViT"}
    if len(hn_idx) >= 2:
        hn_folds = {s: jury[s]["train_fold"] for s in hn_idx}
        t = _tabulate([{**r, "votes": {k: v for k, v in r["votes"].items() if k in hn_idx}}
                       for r in records], hn_folds, min_votes=2)
        sens["hn_only"] = {"overall": t["overall"], "by_pair": t["by_pair"], "min_votes": 2}
    else:
        sens["hn_only"] = {"skipped": "fewer than two non-CellViT seats"}
    for name, thr, mv in (("iou_thr_05", 0.5, min_votes), ("min_votes_5", vote_thr, 5)):
        recs = []
        for j in range(n_img):
            votes_j = {}
            for s, seat in enumerate(jury):
                for gid, vote in jury_votes(gt_inst[j], seat["inst"][j], seat["type"][j],
                                            thr=thr).items():
                    if vote is not None:
                        votes_j.setdefault(gid, {})[s] = vote[1]
            for r in (x for x in records if x["img"] == j):
                recs.append({**r, "votes": votes_j.get(r["gt_id"], {})})
        t = _tabulate(recs, folds, mv)
        sens[name] = {"overall": t["overall"], "by_pair": t["by_pair"]}
    return {"groups": groups, "overall": tab["overall"], "by_pair": tab["by_pair"],
            "bootstrap": {k: list(v) for k, v in boot.items()}, "sensitivity": sens,
            "params": {"pair_thr": pair_thr, "vote_thr": vote_thr, "min_votes": min_votes,
                       "n_boot": n_boot, "seed": seed}}


def _render_md(res: dict, jury: list[dict]) -> str:
    lines = ["# E1-T 跨检测器分型共识（检测器共识证据，非生物学断言）", "",
             f"jury {len(jury)} 席：{', '.join(s['name'] for s in jury)}", "",
             "| group | n | maj=GT | maj=base | maj=other | maj≠GT | no majority |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    o = res["overall"]
    lines.append(f"| G1 基分错 | {o['G1']['n']} | {o['G1']['maj_eq_gt']} | {o['G1']['maj_eq_base']} "
                 f"| {o['G1']['maj_other']} | — | {o['G1']['no_majority']} |")
    lines.append(f"| G0 基分对 | {o['G0']['n']} | {o['G0']['maj_eq_gt']} | — | — "
                 f"| {o['G0']['maj_neq_gt']} | {o['G0']['no_majority']} |")
    lines += ["", "## bootstrap 95% CI（图像聚类）"]
    for k, (lo, hi) in res["bootstrap"].items():
        lines.append(f"- {k}: [{lo:.3f}, {hi:.3f}]")
    lines += ["", "## by_pair（G1；gt_class,base_class → maj=base / maj=GT / other / no-maj）"]
    for k in sorted(res["by_pair"], key=lambda s: tuple(map(int, s.split(",")))):
        c = res["by_pair"][k]
        lines.append(f"- ({k}): n={c['n']} base={c['maj_eq_base']} gt={c['maj_eq_gt']} "
                     f"other={c['maj_other']} nomaj={c['no_majority']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--pair", default=PAIR)
    ap.add_argument("--extra-jury", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    if a.pair != PAIR:
        raise SystemExit(f"only the frozen pair {PAIR} is supported")
    ensure_dev_fold(2, 1)
    if (a.out / "stats.json").exists():
        raise SystemExit(f"{a.out / 'stats.json'} exists; refusing to overwrite")

    jury = builtin_jury(a.round)
    for spec in a.extra_jury:
        path, name, arch, tf = spec.split(":")
        jury.append({"name": name, "path": path, "arch": arch, "train_fold": int(tf[5:]),
                     "seed": None, "scale": "x1"})
    base_path = a.round / "p0" / PAIR / "val" / "base" / "identity" / "pred.npz"

    fold = PanNukeFold(2)
    gt_inst, gt_type = np.asarray(fold.inst), np.asarray(fold.type)
    with np.load(base_path) as z:
        base_inst, base_type = z["inst"], z["type"]
    for seat in jury:
        with np.load(seat["path"]) as z:
            seat["inst"], seat["type"] = z["inst"], z["type"]
        if seat["inst"].shape != gt_inst.shape:
            raise SystemExit(f"jury {seat['name']} shape mismatch vs fold2 GT")

    res = analyze(gt_inst, gt_type, base_inst, base_type, jury, n_boot=a.n_boot)
    res["provenance"] = {
        "base": file_provenance(base_path),
        "jury": [{"name": s["name"], "arch": s["arch"], "train_fold": s["train_fold"],
                  "path": s["path"], "sha256": file_provenance(Path(s["path"]))["sha256"]}
                 for s in jury],
        "script": file_provenance(Path(__file__))["sha256"],
        "round": str(a.round.resolve()),
        "spec": "docs/superpowers/specs/2026-10-11-e1t-typing-consensus-design.md",
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "stats.json").write_text(json.dumps(res, indent=1, default=float))
    (a.out / "tables.md").write_text(_render_md(res, jury))
    print(f"E1-T: G1={res['groups']['G1']} G0={res['groups']['G0']} "
          f"ambiguous={res['groups']['ambiguous']} unmatched={res['groups']['unmatched_gt']}")


if __name__ == "__main__":
    main()
