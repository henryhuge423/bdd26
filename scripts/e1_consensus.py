#!/usr/bin/env python
"""E1-C：跨检测器共识分析（spec §7.1，2026-10-10 批准）。

    python scripts/e1_consensus.py \
        --round runs/analysis/matched_seed_20261004 --pair split1_seed1 \
        --records runs/analysis/e1_input_check_20261010_split1_seed1/records.npz \
        --packet runs/analysis/e1_packet_20261010 \
        --extra-jury runs/analysis/e1_consensus_20261010/hovernet_split1_fold2/pred.npz:HoverNet:hovernet:split1 \
        --out runs/analysis/e1_consensus_20261010

Jury 固定 12 席（split1/split3 × seed{19,1,2} × {x1,x2} 的 fold2 identity 预测）剔除来源
run（split1_seed1-x2）后 11 席 + --extra-jury（HoVer-Net split1/split3 fold2）。全部成员
训练折 ∈ {1,3}，fold2 从未参与训练。支持度 = 与候选 IoU ≥ 0.3 的成员数（0.2/0.5 记录为
敏感性）。输出检测器共识证据，不作生物学断言。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fold_guard import ensure_dev_fold  # noqa: E402
from audit_existence_candidates import audit_pair  # noqa: E402

from nucseg.data.pannuke import PanNukeFold  # noqa: E402
from nucseg.metrics.instance import overlap  # noqa: E402

PRIMARY_THR = 0.3
SENS_THRS = (0.2, 0.3, 0.5)


def support_for_candidate(cand: np.ndarray, jury_maps, thr: float = PRIMARY_THR) -> int:
    """Jury members with an instance at IoU >= thr vs the single-label candidate mask."""
    n = 0
    for m in jury_maps:
        if not np.any(m) or not np.any(cand):
            continue
        ov = overlap(cand.astype(np.int32), m.astype(np.int32))
        if ov.np_ and float(ov.iou[0].max()) >= thr:
            n += 1
    return n


def window_instance_hits(inst: np.ndarray, cy: int, cx: int, size: int = 160) -> int:
    """Instances whose centroid falls inside the size×size window centered (clamped) at (cy,cx)."""
    ids = np.unique(inst); ids = ids[ids > 0]
    oy = min(max(cy - size // 2, 0), inst.shape[0] - size)
    ox = min(max(cx - size // 2, 0), inst.shape[1] - size)
    hits = 0
    for i in ids:
        ys, xs = np.nonzero(inst == i)
        if oy <= ys.mean() < oy + size and ox <= xs.mean() < ox + size:
            hits += 1
    return hits


def builtin_jury(round_root: Path) -> list[dict]:
    out = []
    for split in (1, 3):
        for seed in (19, 1, 2):
            for scale, arm in (("x1", "base"), ("x2", "x2")):
                if (split, seed, scale) == (1, 1, "x2"):
                    continue  # source run excluded (spec §7.1)
                out.append({
                    "name": f"cellvit_s{split}_e{seed}_{scale}",
                    "path": str(round_root / "p0" / f"split{split}_seed{seed}" / "val" / arm
                             / "identity" / "pred.npz"),
                    "arch": "CellViT", "train_fold": split, "seed": seed, "scale": scale})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--pair", default="split1_seed1")
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--extra-jury", nargs="*", default=[],
                    help="PATH:NAME:ARCH:TRAINFOLD entries (e.g. the two HoVer-Net fold2 runs)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if (a.out / "consensus.json").exists():
        raise SystemExit("consensus.json exists; refusing to overwrite")
    ensure_dev_fold(2, 1)
    a.out.mkdir(parents=True, exist_ok=True)

    jury = builtin_jury(a.round)
    for spec in a.extra_jury:
        path, name, arch, tf = spec.split(":")
        jury.append({"name": name, "path": path, "arch": arch, "train_fold": int(tf[5:]),
                     "seed": None, "scale": "x1"})

    rows = audit_pair(a.round, a.pair, 1, 1, a.out, 4)
    z = np.load(a.records)
    sel = z["pair"] == a.pair
    for col in ("image", "id", "added", "matched", "area", "border"):
        mine = np.array([r[col] for r in rows])
        ref = np.asarray(z[col])[sel]
        if not np.array_equal(mine.astype(ref.dtype), ref):
            raise SystemExit(f"binding failed on column {col}")

    selection = json.loads((a.round / "p2" / a.pair / "selection.json").read_text())
    x2_val = Path(selection["provenance"]["inputs"]["x2"]["path"])
    src = np.load(x2_val)
    src_inst = src["inst"]

    by_img: dict[int, list[int]] = {}
    for k, r in enumerate(rows):
        by_img.setdefault(r["image"], []).append(k)

    n = len(rows)
    iou_mat = np.zeros((n, len(jury)), np.float32)     # per candidate × member max IoU
    for j, member in enumerate(jury):
        inst = np.load(member["path"])["inst"]          # materialize ONCE per member
        for img, idxs in by_img.items():
            cmap = np.zeros((256, 256), np.int32)
            for k in idxs:
                cmap[src_inst[img] == rows[k]["id"]] = k + 1
            if not np.any(cmap):
                continue
            ov = overlap(cmap, inst[img].astype(np.int32))
            best = ov.iou.max(1) if ov.np_ else np.zeros(len(idxs))
            iou_mat[idxs, j] = best
        print(f"jury {member['name']}: done")
        del inst

    # ---- S4 background windows（材料包 sample.json）
    sample = json.loads((a.packet / "KEY" / "sample.json").read_text())
    s4 = [it for it in sample["items"].values() if it["source"] == "S4"]
    s4_hits = np.zeros((len(s4), len(jury)), np.int32)
    for j, member in enumerate(jury):
        inst = np.load(member["path"])["inst"]
        for w, it in enumerate(s4):
            s4_hits[w, j] = window_instance_hits(inst[it["image"]], it["cy"], it["cx"])
        print(f"jury {member['name']}: S4 done")
        del inst

    # ---- 汇总
    matched = np.array([r["matched"] for r in rows])
    added = np.array([r["added"] for r in rows])
    area = np.array([r["area"] for r in rows])
    border = np.array([r["border"] for r in rows])
    arch_arr = np.array([m["arch"] for m in jury])
    fold_arr = np.array([m["train_fold"] for m in jury])

    def summarize(mask, label):
        out = {"n": int(mask.sum())}
        for thr in SENS_THRS:
            sup = (iou_mat[mask] >= thr).sum(1)
            out[f"support_hist_thr{thr}"] = np.bincount(sup, minlength=len(jury) + 1).tolist()
        sup = (iou_mat[mask] >= PRIMARY_THR)
        cnt = sup.sum(1)
        crossfold = ((sup & (fold_arr[None, :] == 3)).sum(1) > 0) | \
                    ((sup & (arch_arr[None, :] != "CellViT")).sum(1) > 0)
        out.update({
            "mean_support": float(cnt.mean()),
            "matched_rate_by_support": {str(k): float(matched[mask][cnt == k].mean())
                                        for k in range(len(jury) + 1) if (cnt == k).any()},
            "unmatched_ge3_crossfold": int((~matched[mask] & (cnt >= 3) & crossfold).sum()),
            "unmatched_le1": int((~matched[mask] & (cnt <= 1)).sum()),
        })
        return label, out

    per_source = dict([summarize(added, "S2_added"), summarize(~added, "S3_rejected")])
    area_bins = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 10 ** 9)]
    per_area = {}
    for lo, hi in area_bins:
        for s in (True, False):
            m = (added == s) & (area >= lo) & (area < hi)
            if m.any():
                per_area[f"{'S2' if s else 'S3'}_area[{lo},{hi})"] = summarize(m, "")[1]

    payload = {
        "spec": "docs/superpowers/specs/2026-10-10-e1-blind-audit-design.md#7",
        "pair": a.pair, "n_candidates": n, "jury": jury, "primary_thr": PRIMARY_THR,
        "per_source": per_source, "per_area_bin": per_area,
        "border_rate_by_source": {"S2": float(border[added].mean()),
                                  "S3": float(border[~added].mean())},
        "s4_windows": {"n": len(s4),
                       "mean_hits_per_window": s4_hits.mean(1).round(3).tolist(),
                       "mean_hits_by_member": {jury[j]["name"]: float(s4_hits[:, j].mean())
                                               for j in range(len(jury))}},
        "caveats": ["detector-consensus evidence only, no biological adjudication",
                    "jury members share the CellViT architecture for 11/13 seats; HoVer-Net "
                    "covers 2 — cross-architecture support is recorded explicitly",
                    "all jury training folds are 1 or 3; fold2 was never trained on"],
    }
    (a.out / "consensus.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(a.out / "per_candidate.npz",
                        iou=iou_mat,
                        jury_names=np.array([m["name"] for m in jury]),
                        jury_fold=np.array([m["train_fold"] for m in jury]),
                        jury_arch=np.array([m["arch"] for m in jury]))
    print(f"wrote {a.out}/consensus.json")
    for s in ("S2_added", "S3_rejected"):
        d = per_source[s]
        print(f"{s}: n={d['n']} mean_support={d['mean_support']:.2f} "
              f"unmatched_ge3_crossfold={d['unmatched_ge3_crossfold']} "
              f"unmatched_le1={d['unmatched_le1']}")


if __name__ == "__main__":
    main()
