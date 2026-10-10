#!/usr/bin/env python
"""E1 盲评分析（在任何评审答案存在之前冻结——spec §3）。

    python scripts/e1_analyze.py --packet runs/analysis/e1_packet_20261010 \
        --r1 KEY/reviewer1_stage1.csv --r2 KEY/reviewer2_stage1.csv \
        [--r1s2 KEY/reviewer1_stage2.csv --r2s2 KEY/reviewer2_stage2.csv] \
        --out runs/analysis/e1_packet_20261010/analysis

输出：每源 HT 估计 P(真实核)（各评审只按自己非 uncertain 的行估计，不按对方答案条件化）、
两评审一致率与 Cohen's kappa（uncertain 单列，不并入非核）、评审判断 vs 冻结 `matched`
（GT IoU>.5）交叉表、不确定比例；阶段二（可选）：每源 relation 分布、S4"疑似漏标"计数。
表头匹配：构建器生成的列名带括号说明（如 is_nucleus(real/…)），读取时按前缀归一化。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

_PREFIXES = ("is_nucleus", "relation", "unlabelled_suspect", "completeness")


def _read_form(path: Path) -> dict[str, dict]:
    """Read a form CSV, normalizing parenthetical header suffixes (builder writes them)."""
    out: dict[str, dict] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            ans = {}
            for k, v in row.items():
                if k is None or k == "id":
                    continue
                key = next((p for p in _PREFIXES if k.strip().startswith(p)), k.strip())
                ans[key] = (v or "").strip()
            out[row["id"].strip()] = ans
    return out


def _binarize(ans: dict) -> int:
    """real -> 1, non_nucleus -> 0, uncertain -> -1."""
    v = (ans.get("is_nucleus") or "").lower()
    if v.startswith("real"):
        return 1
    if v.startswith("non"):
        return 0
    return -1


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


def _relation_tally(rel: dict) -> dict:
    """Stage-2 relation distribution (consistent/boundary_offset/merged/oversegmented/...)."""
    out: dict[str, int] = {}
    for v in rel.values():
        if v:
            out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _s4_unlabelled_tally(q: dict) -> dict:
    """S4 extra question: unlabelled-suspect yes/no/uncertain counts."""
    return _relation_tally(q)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--r1", type=Path, required=True)
    ap.add_argument("--r2", type=Path, required=True)
    ap.add_argument("--r1s2", type=Path, default=None, help="stage-2 answers (optional)")
    ap.add_argument("--r2s2", type=Path, default=None, help="stage-2 answers (optional)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    out_json = a.out.with_suffix(".json")
    if out_json.exists():
        raise SystemExit(f"{out_json} exists; refusing to overwrite")

    sample = json.loads((a.packet / "KEY" / "sample.json").read_text())
    items = sample["items"]
    f1, f2 = _read_form(a.r1), _read_form(a.r2)
    missing = [nid for nid in items if nid not in f1 or nid not in f2]
    if missing:
        raise SystemExit(f"{len(missing)} windows lack stage-1 answers (e.g. {missing[:5]})")
    s1r, s2r = (_read_form(a.r1s2), _read_form(a.r2s2)) if a.r1s2 and a.r2s2 else (None, None)

    report = {"per_source": {}, "notes": [
        "HT weights w=1/pi within each source; window-level for S4 (not comparable denominators)",
        "per-reviewer HT uses THAT reviewer's non-uncertain rows only (no cross-conditioning);",
        "kappa/agreement drop uncertain rows pairwise; uncertain never merged into non-nucleus",
        "200 windows check audit feasibility only; rare-class subgroups underpowered (spec §3)",
    ]}
    srcs = sorted({it["source"] for it in items.values()})
    for s in srcs:
        nids = [nid for nid, it in items.items() if it["source"] == s]
        y1 = np.array([_binarize(f1[nid]) for nid in nids])
        y2 = np.array([_binarize(f2[nid]) for nid in nids])
        probs = np.asarray([items[nid]["prob"] for nid in nids], float)
        ok1, ok2 = y1 != -1, y2 != -1     # per-reviewer masks (review fix #4)
        both = ok1 & ok2
        rep = {
            "n": len(nids),
            "n_certain_r1": int(ok1.sum()), "n_certain_r2": int(ok2.sum()),
            "ht_real_r1": ht_estimate(y1[ok1], probs[ok1]) if ok1.any() else None,
            "ht_real_r2": ht_estimate(y2[ok2], probs[ok2]) if ok2.any() else None,
            "raw_agreement": float((y1 == y2).mean()),
            "kappa": kappa(y1, y2),
            "uncertain_rate_r1": float((y1 == -1).mean()),
            "uncertain_rate_r2": float((y2 == -1).mean()),
        }
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
        if s1r is not None:
            rep["stage2_relation_r1"] = _relation_tally({n: s1r[n].get("relation", "")
                                                         for n in nids if n in s1r})
            rep["stage2_relation_r2"] = _relation_tally({n: s2r[n].get("relation", "")
                                                         for n in nids if n in s2r})
            if s == "S4":
                rep["s4_unlabelled_r1"] = _s4_unlabelled_tally(
                    {n: s1r[n].get("unlabelled_suspect", "") for n in nids if n in s1r})
                rep["s4_unlabelled_r2"] = _s4_unlabelled_tally(
                    {n: s2r[n].get("unlabelled_suspect", "") for n in nids if n in s2r})
        report["per_source"][s] = rep

    out_json.write_text(json.dumps(report, indent=2))
    a.out.with_suffix(".md").write_text(
        "| source | n | HT real (r1) | HT real (r2) | raw agr | kappa | unc. r1 | unc. r2 |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(
            f"| {s} | {r['n']} | {r['ht_real_r1']:.3f} | {r['ht_real_r2']:.3f} "
            f"| {r['raw_agreement']:.2f} | {r['kappa']:.2f} | {r['uncertain_rate_r1']:.2f} "
            f"| {r['uncertain_rate_r2']:.2f} |" for s, r in report["per_source"].items()))
    print(json.dumps({s: round(r["ht_real_r1"], 3) for s, r in report["per_source"].items()
                      if r["ht_real_r1"] is not None}, indent=2))


if __name__ == "__main__":
    main()
