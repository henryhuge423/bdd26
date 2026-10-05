"""Cross-check stage report III against raw evaluation summaries and audit rows.

Run from the repository root. Prints portable JSON; redirect stdout to a new file
if a verification snapshot is needed. Does not modify any experiment artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ENDPOINTS = ("mPQ", "bPQ", "mPQ+", "strict mPQ", "Dead PQ", "strict Dead PQ")


def load(path):
    return json.loads(path.read_text())


def scores(path):
    d = load(path)
    return {"mPQ": d["official"]["mPQ"], "bPQ": d["official"]["bPQ"],
            "mPQ+": d["plus"]["mPQ+"], "strict mPQ": d["strict"]["mPQ"],
            "Dead PQ": d["per_class_PQ"]["Dead"],
            "strict Dead PQ": d["per_class_PQ_strict"]["Dead"]}


def close(a, b):
    if not np.isclose(a, b, rtol=0, atol=1e-12):
        raise ValueError(f"Mismatch: {a!r} != {b!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/analysis"))
    args = parser.parse_args()
    ms = args.root / "matched_seed_20261004"
    ef = args.root / "existence_ef_20261005"
    pair_names = [f"split{s}_seed{k}" for s in (1, 2, 3) for k in (19, 1, 2)]
    result = {"scope": "Raw summaries, frozen selections and validation audit records; no new evaluation",
              "arms": {}, "source_sha256": {}, "verified_endpoint_deltas": 0}
    for label, root, folder in (("P2", ms, "p2"), ("EF-P2", ef, "p2ef")):
        stat_path = root / "stats/stats.json"
        stat = load(stat_path)
        if set(stat["pairs"]) != set(pair_names):
            raise ValueError("Expected exactly nine distinct split/seed pairs")
        result["source_sha256"][str(stat_path)] = hashlib.sha256(stat_path.read_bytes()).hexdigest()
        absolute = []
        base_absolute = []
        deltas = []
        selections = {}
        for pair in pair_names:
            baseline = scores(ms / f"p0/{pair}/test/base/ms/eval/summary.json")
            method = scores(root / f"{folder}/{pair}/test/eval/summary.json")
            delta = {ep: method[ep] - baseline[ep] for ep in ENDPOINTS}
            for ep in ENDPOINTS:
                close(baseline[ep], stat["pairs"][pair]["baseline"][ep])
                close(method[ep], stat["pairs"][pair]["p2"][ep])
                close(delta[ep], stat["pairs"][pair]["delta"][ep])
                result["verified_endpoint_deltas"] += 1
            deltas.append(delta)
            absolute.append(method)
            base_absolute.append(baseline)
            selection = load(root / f"{folder}/{pair}/selection.json")
            selections[pair] = selection["selected"]["name"]
            if label == "EF-P2":
                off = next(row for row in selection["rows"] if row["name"] == "off")
                original = load(ms / f"p2/{pair}/selection.json")["selected"]
                for key in ("mPQ", "bPQ", "strict_dead", "interior_matched", "added"):
                    close(off[key], original[key])
        mean = {ep: float(np.mean([d[ep] for d in deltas])) for ep in ENDPOINTS}
        max_std = {ep: max(float(np.std([d[ep] for d in deltas[i:i+3]], ddof=0))
                           for i in (0, 3, 6)) for ep in ENDPOINTS}
        signs = {ep: {"positive": sum(d[ep] > 0 for d in deltas),
                      "zero": sum(d[ep] == 0 for d in deltas),
                      "negative": sum(d[ep] < 0 for d in deltas)} for ep in ENDPOINTS}
        for ep in ENDPOINTS:
            close(mean[ep], stat["overall"]["three_split_mean"][ep])
            close(max_std[ep], stat["overall"]["max_split_seed_std"][ep])
        decision = (mean["Dead PQ"] >= max_std["Dead PQ"]
                    and signs["Dead PQ"]["negative"] <= 2
                    and mean["mPQ"] >= -max_std["mPQ"]
                    and mean["bPQ"] >= -max_std["bPQ"])
        if decision != stat["decision"]["robust_improvement"]:
            raise ValueError(f"Decision mismatch: {label}")
        result["arms"][label] = {"mean": {ep: float(np.mean([d[ep] for d in absolute])) for ep in ENDPOINTS},
                                 "delta": mean, "max_paired_delta_sd": max_std,
                                 "sign_counts": signs, "decision": bool(decision), "selections": selections}
        result["baseline"] = {ep: float(np.mean([d[ep] for d in base_absolute])) for ep in ENDPOINTS}

    audit_root = args.root / "existence_audit_20261005_v2"
    summary = load(audit_root / "summary.json")
    with np.load(audit_root / "records.npz") as data:
        added = data["added"].astype(bool)
        matched = data["matched"].astype(bool)
        result["audit"] = {"eligible": len(added), "added": int(added.sum()),
                           "matched": int((added & matched).sum()), "area": [], "border": []}
        for axis, bins in (("area", [(0, 30), (30, 60), (60, 100), (100, 150), (150, 200), (200, 10**9)]),
                           ("border", [(0, 1), (1, 2)])):
            for index, (low, high) in enumerate(bins):
                mask = added & (data[axis] >= low) & (data[axis] < high)
                row = {"bin": [low, high], "n": int(mask.sum()),
                       "matched": int((mask & matched).sum()),
                       "typed": int((mask & data["typed"]).sum()),
                       "dead_typed": int((mask & data["typed"] & (data["class"] == 4)).sum())}
                table = summary["tables"][f"added_by_{axis}"]
                expected = (next(entry for entry in table if entry["bin"] == str(bool(low)))
                            if axis == "border" else table[index])
                for key in ("n", "matched", "typed", "dead_typed"):
                    close(row[key], expected[key])
                row["match_rate"] = row["matched"] / row["n"] if row["n"] else None
                result["audit"][axis].append(row)
    result["dead_gain_retained"] = result["arms"]["EF-P2"]["delta"]["Dead PQ"] / result["arms"]["P2"]["delta"]["Dead PQ"]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
