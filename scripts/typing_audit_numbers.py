#!/usr/bin/env python
"""Aggregate the six typing-audit JSONs into the summary tables used in docs/findings.md.

Refuses missing splits/roles rather than averaging what happens to exist. Descriptive only.

    python scripts/typing_audit_numbers.py --root runs/analysis/typing_audit_20261003 \
        --out-json runs/analysis/typing_audit_20261003/summary.json \
        --out-markdown runs/analysis/typing_audit_20261003/summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = (1, 2, 3)
ROLES = ("val", "test")
ARMS = ("x1", "x2")
CLASS_NAMES = {1: "Neo", 2: "Inf", 3: "Con", 4: "Dead", 5: "Epi"}


def load(root):
    out = {}
    for split in SPLITS:
        for role in ROLES:
            path = root / f"split{split}_{role}.json"
            if not path.exists():
                raise SystemExit(f"missing audit output {path}; run the audit first")
            d = json.loads(path.read_text())
            if d["split"] != split or d["role"] != role:
                raise SystemExit(f"{path} claims split/role {d['split']}/{d['role']}")
            out[(split, role)] = d
    return out


def bin_table(cal):
    """(correct, matched, n) per confidence bin pooled over classes; Dead-only alongside."""
    pooled = {}
    dead = {}
    for b, cells in cal["conf_bins"].items():
        c = sum(v["correct"] for v in cells.values())
        m = sum(v["matched"] for v in cells.values())
        n = sum(v["n"] for v in cells.values())
        pooled[b] = (c, m, n)
        d = cells["4"]
        dead[b] = (d["correct"], d["matched"], d["n"])
    return pooled, dead


def arm_summary(data):
    """Per arm: overall accuracy among matched, Dead accuracy among matched, argmax stats."""
    rows = {}
    for arm in ARMS:
        key = f"{arm}_calibration"
        if key not in data:
            continue
        cal, cls = data[key], data[f"{arm}_class_summary"]
        tot_c = sum(v["correct"] for v in cls.values())
        tot_m = sum(v["matched"] for v in cls.values())
        tot_n = sum(v["n"] for v in cls.values())
        d = cls["4"]
        ag = cal["row_argmax_agreement"]
        dis = ag["disagree"]
        rows[arm] = {"n_instances": tot_n, "matched": tot_m, "correct": tot_c,
                     "acc_given_matched": tot_c / tot_m if tot_m else None,
                     "dead_matched": d["matched"], "dead_correct": d["correct"],
                     "dead_acc_given_matched": d["correct"] / d["matched"] if d["matched"] else None,
                     "argmax_disagree": dis,
                     "argmax_disagree_rate": dis / (dis + ag["agree"]) if dis + ag["agree"] else None,
                     "argmax_disagree_acc": ag["disagree_correct"] / dis if dis else None,
                     "argmax_agree_acc": ag["agree_correct"] / ag["agree"] if ag["agree"] else None}
    return rows


def mean3(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "runs/analysis/typing_audit_20261003")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-markdown", type=Path, required=True)
    a = p.parse_args()
    for out in (a.out_json, a.out_markdown):
        if out.exists():
            raise SystemExit(f"refusing to overwrite existing output {out}")
    data = load(a.root)

    summary = {"scope": "descriptive aggregation of typing_audit JSONs; no thresholds selected",
               "per_split": {}, "three_split_mean": {}}
    for split in SPLITS:
        for role in ROLES:
            d = data[(split, role)]
            entry = {"arms": arm_summary(d)}
            pooled = {}
            for arm in ARMS:
                if f"{arm}_calibration" in d:
                    pooled[arm], pooled[f"{arm}_dead"] = bin_table(d[f"{arm}_calibration"])
            entry["conf_bins"] = pooled
            ag = d["agreement"]["conditioned"]
            atr = d["agreement"]["disagreement_attribution"]
            br, xr = atr["base_right_x2_wrong"], atr["base_wrong_x2_right"]
            entry["agreement"] = {
                "stable_pairs": d["agreement"]["stable_pairs"],
                "agree_rate": ag["agree"]["n"] / d["agreement"]["stable_pairs"],
                "agree_base_acc": ag["agree"]["base_correct"] / ag["agree"]["n"],
                "agree_x2_acc": ag["agree"]["x2_correct"] / ag["agree"]["n"],
                "disagree_n": ag["disagree"]["n"],
                "disagree_base_acc": ag["disagree"]["base_correct"] / ag["disagree"]["n"],
                "disagree_x2_acc": ag["disagree"]["x2_correct"] / ag["disagree"]["n"],
                "base_right_x2_wrong": br, "base_wrong_x2_right": xr, "both_wrong": atr["both_wrong"],
                "mean_conf_base_right": atr["base_right_x2_wrong_conf_sum"] / br if br else None,
                "mean_conf_x2_right": atr["base_wrong_x2_right_conf_sum"] / xr if xr else None}
            if "p2_added" in d:
                by = d["p2_added"]["by_class"]
                entry["p2_added"] = {"n": d["p2_added"]["n"],
                                     **{CLASS_NAMES[int(c)]: v for c, v in by.items() if int(c) > 0}}
                entry["p2_rejected"] = {"n": d["p2_rejected"]["n"], "reasons": d["p2_rejected"]["reasons"]}
            if "anchor_unfiltered_vs_raw_x1" in d:
                entry["anchor"] = {"n": d["anchor_unfiltered_vs_raw_x1"]["n"],
                                   "matched": d["anchor_unfiltered_vs_raw_x1"]["by_class"]}
            if "repro_sanity" in d:
                entry["repro_sanity"] = d["repro_sanity"]["aggregate"]
            summary["per_split"][f"split{split}_{role}"] = entry

    for role in ROLES:
        entries = [summary["per_split"][f"split{s}_{role}"] for s in SPLITS]
        mean = {"agreement": {}, "arms": {}}
        for k in ("stable_pairs", "agree_rate", "agree_base_acc", "agree_x2_acc", "disagree_n",
                  "disagree_base_acc", "disagree_x2_acc", "base_right_x2_wrong",
                  "base_wrong_x2_right", "both_wrong", "mean_conf_base_right", "mean_conf_x2_right"):
            mean["agreement"][k] = mean3([e["agreement"][k] for e in entries])
        for arm in ARMS:
            if arm in entries[0]["arms"]:
                mean["arms"][arm] = {k: mean3([e["arms"][arm].get(k) for e in entries])
                                     for k in entries[0]["arms"][arm]}
        mean["p2_added"] = {k: mean3([e.get("p2_added", {}).get(k) for e in entries])
                            for k in ("n",) }
        summary["three_split_mean"][role] = mean

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(summary, indent=2) + "\n")

    md = ["# Typing-confidence audit summary (descriptive)", "",
          "Per-split details in summary.json; all numbers derived from the audit JSONs.", ""]
    for role in ROLES:
        m = summary["three_split_mean"][role]
        md += [f"## 3-split mean ({role})", "",
               "| arm | acc\\|matched | Dead acc\\|matched | argmax disagree | disagree acc | agree acc |",
               "|---|---:|---:|---:|---:|---:|"]
        for arm in ("x1", "x2"):
            r = m["arms"].get(arm)
            if r:
                md.append(f"| {arm} | {r['acc_given_matched']:.4f} | {r['dead_acc_given_matched']:.4f} "
                          f"| {r['argmax_disagree_rate']:.4f} | {r['argmax_disagree_acc']:.4f} "
                          f"| {r['argmax_agree_acc']:.4f} |")
        ag = m["agreement"]
        md += ["", f"stable pairs {ag['stable_pairs']:.0f}, agree rate {ag['agree_rate']:.4f}; "
               f"disagree: base right {ag['base_right_x2_wrong']:.0f} vs x2 right "
               f"{ag['base_wrong_x2_right']:.0f} (both wrong {ag['both_wrong']:.0f}); "
               f"mean x2 conf {ag['mean_conf_base_right']:.3f} vs {ag['mean_conf_x2_right']:.3f}", ""]
    a.out_markdown.write_text("\n".join(md) + "\n")
    print(f"wrote {a.out_json} and {a.out_markdown}")


if __name__ == "__main__":
    main()
