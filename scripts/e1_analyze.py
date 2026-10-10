#!/usr/bin/env python
"""E1 盲评分析（在任何评审答案存在之前冻结——spec §3）。

    python scripts/e1_analyze.py --packet runs/analysis/e1_packet_20261010 \
        --r1 forms/reviewer1_stage1.csv --r2 forms/reviewer2_stage1.csv \
        [--r1s2 forms/reviewer1_stage2.csv --r2s2 forms/reviewer2_stage2.csv] \
        --out runs/analysis/e1_packet_20261010/analysis

输出：每源 HT 估计 P(真实核)、两评审一致率与 Cohen's kappa（uncertain 单列，不并入
非核）、评审判断 vs 冻结 `matched`（GT IoU>.5）交叉表、不确定比例。HT 权重 w=1/π；
uncertain 不并入假阳性。S4 的分析单位是窗口（π 按选图×均匀中心两步设计）。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ht_estimate(labels, probs) -> float:
    """Horvitz-Thompson mean of `labels` with inclusion probabilities `probs` (w = 1/pi)."""
    y = np.asarray(labels, float)
    w = 1.0 / np.asarray(probs, float)
    return float((w * y).sum() / w.sum())


def kappa(a, b) -> float:
    """Cohen's kappa on rows where neither reviewer is uncertain (-1)."""
    a, b = np.asarray(a), np.asarray(b)
    keep = (a != -1) & (b != -1)
    a, b = a[keep], b[keep]
    if not len(a) or len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
        return float("nan") if not len(a) else 1.0 if (a == b).all() else 0.0
    po = float((a == b).mean())
    pe = float(sum((a == v).mean() * (b == v).mean() for v in set(a.tolist()) | set(b.tolist())))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _read_form(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[row["id"].strip()] = {k: v.strip() for k, v in row.items() if k != "id"}
    return out


def _binarize(ans: dict) -> int:
    """real -> 1, non_nucleus -> 0, uncertain -> -1（parse 容错：大小写/前缀）。"""
    v = (ans.get("is_nucleus") or "").lower()
    if v.startswith("real"):
        return 1
    if v.startswith("non"):
        return 0
    return -1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--r1", type=Path, required=True)
    ap.add_argument("--r2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if (a.out).with_suffix(".json").exists():
        raise SystemExit(f"{a.out}.json exists; refusing to overwrite")

    sample = json.loads((a.packet / "sample.json").read_text())
    items = sample["items"]
    f1, f2 = _read_form(a.r1), _read_form(a.r2)
    missing = [nid for nid in items if nid not in f1 or nid not in f2]
    if missing:
        raise SystemExit(f"{len(missing)} windows lack answers (e.g. {missing[:5]})")

    report = {"per_source": {}, "agreement": {}, "uncertain": {}}
    srcs = sorted({it["source"] for it in items.values()})
    for s in srcs:
        nids = [nid for nid, it in items.items() if it["source"] == s]
        y1 = np.array([_binarize(f1[nid]) for nid in nids])
        y2 = np.array([_binarize(f2[nid]) for nid in nids])
        probs = [items[nid]["prob"] for nid in nids]
        both = (y1 != -1) & (y2 != -1)
        rep = {
            "n": len(nids),
            "ht_real_r1": ht_estimate(y1[both], np.asarray(probs)[both]),
            "ht_real_r2": ht_estimate(y2[both], np.asarray(probs)[both]),
            "raw_agreement": float((y1 == y2).mean()),
            "kappa": kappa(y1, y2),
            "uncertain_rate_r1": float((y1 == -1).mean()),
            "uncertain_rate_r2": float((y2 == -1).mean()),
        }
        # 评审 vs 冻结 GT 匹配标记（S2/S3 有 matched；uncertain 单列）
        m = np.array([items[nid].get("matched") for nid in nids], object)
        have = np.array([v is not None and v != "" for v in m]) & both
        if have.any():
            mv = np.array([bool(v) for v in m[have]], int)
            for name, y in (("r1", y1[have]), ("r2", y2[have])):
                rep[f"vs_matched_{name}"] = {
                    "both_real": int(((y == 1) & (mv == 1)).sum()),
                    "real_but_unmatched": int(((y == 1) & (mv == 0)).sum()),
                    "unmatched_but_non_nucleus": int(((y == 0) & (mv == 0)).sum()),
                    "matched_but_non_nucleus": int(((y == 0) & (mv == 1)).sum()),
                }
        report["per_source"][s] = rep

    report["notes"] = [
        "HT weights w=1/pi within each source; window-level for S4 (not comparable denominators)",
        "uncertain never merged into non-nucleus; kappa drops uncertain rows pairwise",
        "200 windows check audit feasibility only; rare-class subgroups underpowered (spec §3)",
    ]
    out_json = a.out.with_suffix(".json")
    out_json.write_text(json.dumps(report, indent=2))
    (a.out.with_suffix(".md")).write_text(
        "| source | n | HT real (r1) | HT real (r2) | raw agr | kappa | unc. r1 | unc. r2 |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(
            f"| {s} | {r['n']} | {r['ht_real_r1']:.3f} | {r['ht_real_r2']:.3f} "
            f"| {r['raw_agreement']:.2f} | {r['kappa']:.2f} | {r['uncertain_rate_r1']:.2f} "
            f"| {r['uncertain_rate_r2']:.2f} |" for s, r in report["per_source"].items()))
    print(json.dumps({s: round(r["ht_real_r1"], 3) for s, r in report["per_source"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
