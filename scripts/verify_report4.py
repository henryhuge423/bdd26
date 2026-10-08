#!/usr/bin/env python
"""Independent verification of stage-report-IV numbers from the rawest artifacts.

Recomputes the report's headline numbers from raw evaluation summaries and the
frozen gate/cost JSONs, compares them against the frozen snapshots, and emits
one portable JSON snapshot to stdout:

    python scripts/verify_report4.py > docs/report/stage_report_4/verified_numbers.json

Sources (relative to the repository root):
  runs/hv_threshold_controls_20261006/<arm>_seed<seed>/audit_val_fold2/eval/summary.json
  runs/analysis/hv_threshold_controls_20261006/verified_numbers.json   (bootstrap CIs, protocol)
  runs/analysis/dsb_gates_20261007/{gate_a,gate_b,gate_b_v2,gate_c,gate_d}.json
  runs/analysis/inference_cost_20261008/cost.json
  runs/cellvit_uni/split{1,2,3}/eval_test_fold{3,3,1}/summary.json     (audit re-check)
  runs/cellvit_uni_x2/split2_seed2/eval_test_fold3_du2{,lev}/summary.json (audit re-check)

Does not modify any experiment artifact. Raises ValueError on any mismatch
beyond the stated tolerance.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ENDPOINTS = {
    "mPQ": ("official", "mPQ"),
    "bPQ": ("official", "bPQ"),
    "mPQ+": ("plus", "mPQ+"),
    "strict mPQ": ("strict", "mPQ"),
    "Dead PQ": ("per_class_PQ", "Dead"),
    "strict Dead PQ": ("per_class_PQ_strict", "Dead"),
}
ARMS = ["x1_hv30", "x1_hv8", "x2_hv30", "x2_hv120"]
SEEDS = [19, 1]
GATE_MODULES = ["encoder", "skips", "np_branch", "hv_branch", "tp_branch", "decoder"]
EPS = 1e-12


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scores(summary_path):
    d = load(summary_path)
    out = {}
    for name, (block, key) in ENDPOINTS.items():
        out[name] = float(d[block][key])
    return out


def close(a, b, eps=EPS, what=""):
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=eps):
        raise ValueError(f"mismatch {what}: {a!r} != {b!r} (eps {eps})")
    return a


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--hv-snapshot", type=Path,
                    default=Path("runs/analysis/hv_threshold_controls_20261006/verified_numbers.json"))
    args = ap.parse_args()

    src_sha = {}

    # ---- E2a: recompute all endpoints from the raw audit evaluation summaries ----
    hv_runs_root = args.root / "hv_threshold_controls_20261006"
    per_run, arm_mean = {}, {a: {} for a in ARMS}
    cells = 0
    for arm in ARMS:
        for seed in SEEDS:
            sp = hv_runs_root / f"{arm}_seed{seed}" / "audit_val_fold2" / "eval" / "summary.json"
            per_run[f"{arm}_seed{seed}"] = scores(sp)
            src_sha[str(sp)] = sha256(sp)
    snap = load(args.hv_snapshot)
    src_sha[str(args.hv_snapshot)] = sha256(args.hv_snapshot)
    for key, eps_ in per_run.items():
        for e, v in eps_.items():
            close(v, snap["runs"][key]["endpoints"][e], what=f"e2a {key} {e}")
            cells += 1
    for arm in ARMS:
        for e in ENDPOINTS:
            arm_mean[arm][e] = sum(per_run[f"{arm}_seed{s}"][e] for s in SEEDS) / len(SEEDS)
            close(arm_mean[arm][e], snap["arms"][arm]["mean"][e], what=f"e2a mean {arm} {e}")
            cells += 1
    audit_zero = all(
        all(v == 0.0 for v in snap["runs"][k]["audit_minus_original"].values()) for k in snap["runs"]
    )

    # contrast means re-derived as mean of per-seed differences from the raw summaries
    contrast_means = {}
    for key, c in snap["contrasts"].items():
        contrast_means[key] = {}
        for e in ENDPOINTS:
            redone = sum(
                per_run[f"{c['method']}_seed{s}"][e] - per_run[f"{c['reference']}_seed{s}"][e]
                for s in SEEDS
            ) / len(SEEDS)
            close(redone, c["mean"][e], what=f"contrast {key} {e}")
            contrast_means[key][e] = redone
            cells += 1

    # ---- Gate A: rule arithmetic from the recomputed E2a means ----
    g = args.root / "analysis" / "dsb_gates_20261007"
    ga = load(g / "gate_a.json")
    src_sha[str(g / "gate_a.json")] = sha256(g / "gate_a.json")
    r_ = arm_mean["x1_hv8"]["Dead PQ"] - arm_mean["x1_hv30"]["Dead PQ"]
    R_ = arm_mean["x2_hv30"]["Dead PQ"] - arm_mean["x1_hv30"]["Dead PQ"]
    ratio = r_ / R_
    close(r_, ga["r_x1_hv8_minus_x1_hv30"], eps=1e-9, what="gate_a r")
    close(R_, ga["R_x2_hv30_minus_x1_hv30"], eps=1e-9, what="gate_a R")
    close(ratio, ga["ratio"], eps=1e-9, what="gate_a ratio")
    close(ga["bpq_tax_x1_hv8"], contrast_means["lower_support_x1"]["bPQ"], eps=1e-9,
          what="gate_a bPQ tax vs contrast")

    # ---- Gate B v2: recompute decoder/module statistics from serialized cosines ----
    gb2 = load(g / "gate_b_v2.json")
    src_sha[str(g / "gate_b_v2.json")] = sha256(g / "gate_b_v2.json")
    cos = gb2["per_batch_cosines"]
    gate_b_check = {"decoder": {}, "modules": {}, "per_layer_t": {}}
    dec = cos["decoder"]
    mean_dec = sum(dec) / len(dec)
    neg_dec = sum(c < 0 for c in dec)
    close(mean_dec, gb2["decoder"]["mean_cosine"], what="gate_b decoder mean")
    close(neg_dec / len(dec), gb2["decoder"]["conflict_fraction"], what="gate_b decoder conflict")
    gate_b_check["decoder"] = {
        "mean_cosine": mean_dec, "min": min(dec), "max": max(dec),
        "median": statistics.median(dec), "negative_batches": neg_dec, "n_batches": len(dec),
    }
    for mod in GATE_MODULES:
        lst = cos[mod]
        m = sum(lst) / len(lst)
        neg = sum(c < 0 for c in lst)
        close(m, gb2["modules"][mod]["mean_cosine"], what=f"gate_b {mod} mean")
        close(neg / len(lst), gb2["modules"][mod]["conflict_fraction"], what=f"gate_b {mod} conflict")
        # one-sample t statistic, ddof=1, as pre-registered in the amendment
        sd = statistics.stdev(lst)
        t = m / (sd / math.sqrt(len(lst)))
        close(t, gb2["per_layer"]["modules"][mod]["t"], eps=1e-6, what=f"gate_b {mod} t")
        gate_b_check["modules"][mod] = {"mean_cosine": m, "negative_batches": neg,
                                        "conflict_fraction": neg / len(lst)}
        gate_b_check["per_layer_t"][mod] = t
    gb_void = load(g / "gate_b.json")
    src_sha[str(g / "gate_b.json")] = sha256(g / "gate_b.json")

    # ---- Gate C / Gate D ----
    gc = load(g / "gate_c.json")
    gd = load(g / "gate_d.json")
    src_sha[str(g / "gate_c.json")] = sha256(g / "gate_c.json")
    src_sha[str(g / "gate_d.json")] = sha256(g / "gate_d.json")
    base = gc["rows"][0]
    for row in gc["rows"][1:]:
        close(row["dead_pq"] - base["dead_pq"], row["d_dead"], eps=1e-12, what="gate_c d_dead")
        close(row["bpq"] - base["bpq"], row["d_bpq"], eps=1e-12, what="gate_c d_bpq")
    gate_c_peak = max(gc["rows"][1:], key=lambda r_: r_["d_dead"])
    d_rows = gd["rows"]
    gate_d_deltas = {
        v: {k: d_rows[v][k] - d_rows["base"][k]
            for k in ("dead_pq", "bpq", "mpq", "strict_dead_pq")}
        for v in ("shrunk", "exact")
    }

    # ---- E0: recompute the verdict ratios from per-rep timings ----
    cost = load(args.root / "analysis" / "inference_cost_20261008" / "cost.json")
    cost_path = args.root / "analysis" / "inference_cost_20261008" / "cost.json"
    src_sha[str(cost_path)] = sha256(cost_path)
    best = {a: min(rep["mean_ms"] for rep in spec["reps"]) for a, spec in cost["arms"].items()}
    fuse_wall = min(rep["wall"]["mean_ms"] for rep in cost["fusion"]["reps"])
    fuse_cpu = min(rep["cpu"]["mean_ms"] for rep in cost["fusion"]["reps"])
    e0 = {
        "x1_best_ms": best["x1"],
        "x2_du2_best_ms": best["x2_du2"],
        "x1_tta_best_ms": best["x1_tta"],
        "fuse_wall_ms": fuse_wall,
        "fuse_cpu_ms": fuse_cpu,
        "two_model_over_x1_wall": (best["x1"] + best["x2_du2"] + fuse_wall) / best["x1"],
        "two_model_over_x1_fuse_cpu": (best["x1"] + best["x2_du2"] + fuse_cpu) / best["x1"],
        "x1_tta_over_x1": best["x1_tta"] / best["x1"],
        "x2_du2_over_x1": best["x2_du2"] / best["x1"],
    }
    vi = cost["verdict_inputs"]
    close(e0["two_model_over_x1_wall"], vi["two_model_over_x1_wall"], what="e0 two-model wall")
    close(e0["two_model_over_x1_fuse_cpu"], vi["two_model_over_x1_fuse_cpu"], what="e0 two-model cpu")
    close(e0["x1_tta_over_x1"], vi["x1_tta_over_x1"], what="e0 tta ratio")
    close(e0["fuse_wall_ms"], vi["fuse_wall_ms_per_image"], what="e0 fuse wall")
    close(e0["fuse_cpu_ms"], vi["fuse_cpu_ms_per_image"], what="e0 fuse cpu")

    # ---- Audit re-checks ----
    splits = [("split1", "eval_test_fold3"), ("split2", "eval_test_fold3"), ("split3", "eval_test_fold1")]
    bpq = []
    for sp, ev in splits:
        p = args.root / "cellvit_uni" / sp / ev / "summary.json"
        bpq.append(load(p)["official"]["bPQ"])
        src_sha[str(p)] = sha256(p)
    cellvit_bpq_mean = sum(bpq) / len(bpq)
    x2s2 = args.root / "cellvit_uni_x2" / "split2_seed2"
    du2 = scores(x2s2 / "eval_test_fold3_du2" / "summary.json")
    lev = scores(x2s2 / "eval_test_fold3_du2lev" / "summary.json")
    src_sha[str(x2s2 / "eval_test_fold3_du2" / "summary.json")] = sha256(x2s2 / "eval_test_fold3_du2" / "summary.json")
    src_sha[str(x2s2 / "eval_test_fold3_du2lev" / "summary.json")] = sha256(x2s2 / "eval_test_fold3_du2lev" / "summary.json")
    lever_delta = {e: lev[e] - du2[e] for e in ("mPQ", "bPQ", "Dead PQ")}
    sidecar = x2s2 / "pred_fold3_x2_du2_lev.npz.json"
    src_sha[str(sidecar)] = sha256(sidecar)

    result = {
        "scope": ("Raw audit evaluation summaries, frozen gate JSONs and the frozen cost/E2a "
                  "snapshots; no new training, inference or evaluation"),
        "e2a": {
            "protocol": {k: snap["protocol"][k] for k in
                         ("split", "training_fold", "validation_fold", "seeds", "source_commit",
                          "bootstrap_replicates", "bootstrap_seed")},
            "per_run": per_run,
            "arm_means": arm_mean,
            "contrast_means": contrast_means,
            "contrast_ci95": {k: {e: c["image_bootstrap"][e]["ci95"]
                                  for e in ("mPQ", "bPQ", "Dead PQ")}
                              for k, c in snap["contrasts"].items()},
            "audit_minus_original_all_zero": audit_zero,
            "training_loop_hours_total": round(
                sum(snap["runs"][k]["training"]["training_loop_hours"] for k in snap["runs"]), 2),
        },
        "gate_a": {"r": r_, "R": R_, "ratio": ratio, "bpq_tax_x1_hv8": ga["bpq_tax_x1_hv8"],
                   "verdict": ga["verdict"], "base": ga["base"],
                   "thresholds": {"switch": 0.5, "downgrade": 0.9, "tax_guard": -0.001}},
        "gate_b": {
            "first_run": {"status": "VOID (sequential sampling: 1 Dead-positive batch of 64)",
                          "decoder_mean_cosine": gb_void["decoder"]["mean_cosine"]},
            "amended_rerun": gate_b_check,
            "conflict_bar": gb2["threshold_conflict_fraction"],
            "n_active": gb2["n_active"], "min_active": gb2["min_active"],
            "n_dead_positive_images": gb2["sampling"]["n_dead_positive_images"],
            "per_layer": {"alpha": gb2["per_layer"]["alpha"],
                          "any_significant": gb2["per_layer"]["any_significant"]},
            "verdict": gb2["verdict"],
        },
        "gate_c": {"menu": gc["menu"], "peak_d_dead": gate_c_peak["d_dead"],
                   "peak_tau": gate_c_peak["tau_dead"], "peak_d_bpq": gate_c_peak["d_bpq"],
                   "close_line_d_dead": gc["gain_threshold"], "verdict": gc["verdict"],
                   "base_dead_pq": base["dead_pq"]},
        "gate_d": {"n_added": d_rows["exact"]["n_added"], "deltas": gate_d_deltas,
                   "base": d_rows["base"], "shrunk": d_rows["shrunk"], "exact": d_rows["exact"],
                   "threshold_d_dead": gd["threshold_d_dead"], "verdict": gd["verdict"]},
        "e0_cost": e0,
        "audit_2026_10_08": {
            "cellvit_uni_seed19_3split_bpq": {"per_split": bpq, "exact_mean": cellvit_bpq_mean,
                                              "reported": 0.6653},
            "split2_seed2_regenerated": {"du2": {e: du2[e] for e in ("mPQ", "bPQ", "Dead PQ")},
                                         "du2lev": {e: lev[e] for e in ("mPQ", "bPQ", "Dead PQ")},
                                         "lever_delta": lever_delta},
        },
        "verified_endpoint_cells": cells,
        "source_sha256": src_sha,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
