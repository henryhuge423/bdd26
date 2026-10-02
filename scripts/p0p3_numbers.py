#!/usr/bin/env python
"""Generate P0–P3 tables strictly from complete saved artifacts, with provenance checks."""
from __future__ import annotations
import argparse
import csv
import gzip
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mPQ", "bPQ", "mPQ+", "DeadPQ", "strictDeadPQ", "strictmPQ", "DeadFc")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def mean_complete(rows):
    if set(rows) != {1, 2, 3}:
        raise ValueError("three complete official splits are required; no partial mean")
    keys = set(rows[1])
    if any(set(r) != keys for r in rows.values()):
        raise ValueError("inconsistent metric columns")
    if any(not math.isfinite(v) for r in rows.values() for v in r.values()):
        raise ValueError("nonfinite metric in three-split mean")
    return {k: mean(rows[s][k] for s in (1, 2, 3)) for k in rows[1]}


def seed_from_run(run):
    config = Path(run) / "config.json"
    return load(config).get("seed") if config.exists() else None


def seed_coverage(records):
    grouped = {}
    for r in records:
        grouped.setdefault(r["family"], {}).setdefault(str(r["split"]), []).append(r)
    return {family: {split: {"runs": len(rs), "distinct_seeds": sorted(set(r["seed"] for r in rs)),
                            "duplicate_seeds": sorted(s for s, n in Counter(r["seed"] for r in rs).items() if n > 1),
                            "run_names": [r["run"] for r in rs]}
                     for split, rs in splits.items()} for family, splits in grouped.items()}


def actual_seed_records(root):
    records = {}
    snapshot = root / "remote_run_inventory.json"
    for entry in load(snapshot)["runs"]:
        cfg = entry["config"]
        records[entry["run"]] = {"run": entry["run"], "family": "x2" if "cellvit_uni_x2" in entry["run"] else "base",
                                  "split": cfg["split"], "seed": cfg["seed"]}
    patterns = ("runs/cellvit_uni/split*/config.json", "runs/cellvit_abl/split*_seed*/config.json",
                "runs/cellvit_uni_x2/split*/config.json")
    for pattern in patterns:
        for p in ROOT.glob(pattern):
            cfg = load(p); run = str(p.parent.relative_to(ROOT))
            row = {"run": run, "family": "x2" if "cellvit_uni_x2" in run else "base",
                   "split": cfg["split"], "seed": cfg["seed"]}
            if run in records and records[run] != row:
                raise ValueError(f"local/remote provenance conflict: {run}")
            records[run] = row
    return list(records.values())


def metrics(d):
    return dict(zip(METRICS, (d["official"]["mPQ"], d["official"]["bPQ"], d["plus"]["mPQ+"],
                              d["per_class_PQ"]["Dead"], d["per_class_PQ_strict"]["Dead"],
                              d["strict"]["mPQ"], d["detection"]["F_c"]["Dead"])))


def validate_selection_link(selection_path, manifest, split, stage):
    selection = load(selection_path)
    val_fold, test_fold = {1: (2, 3), 2: (1, 3), 3: (2, 1)}[split]
    if manifest.get("selection_sha256") != sha256(selection_path):
        raise ValueError("selection file differs from the executed test's fingerprint")
    if selection.get("split") != split or manifest.get("split") != split or manifest.get("fold") != test_fold:
        raise ValueError("selection/test split or fold mismatch")
    if stage == "p2":
        if selection.get("fold") != val_fold or selection.get("role") != "validation selection":
            raise ValueError("P2 selection is not from the required validation fold")
        if manifest.get("role") != "frozen test; identity fallback if validation NO_GO" or manifest.get("selected") != selection.get("selected"):
            raise ValueError("reported P2 choice differs from executed test parameters")
    elif stage == "p0":
        if selection.get("validation_fold") != val_fold or manifest.get("role") != "test":
            raise ValueError("P0 selection/test role mismatch")
        arm = manifest["arm"]
        selected = selection["arms"][arm]["selected"]
        if manifest["variant"] not in ("ms", "identity"):
            raise ValueError("unknown P0 variant")
        params = {k: selected[k] for k in ("frac", "a_min")} if manifest["variant"] == "ms" else {"frac": None, "a_min": 0}
        if manifest.get("parameters") != params or manifest.get("actual_seed") != selection["arms"][arm]["actual_seed"]:
            raise ValueError("reported P0 parameters/seed differ from executed test")
        if manifest["code"]["files"] != selection["code"]["files"]:
            raise ValueError("P0 selection/test code fingerprints differ")
    else:
        raise ValueError("unknown selection stage")
    return selection


def verified_summary(directory):
    manifest = load(directory / "manifest.json")
    if manifest["prediction"]["sha256"] != sha256(directory / "pred.npz"):
        raise ValueError(f"saved prediction differs from manifest: {directory}")
    return metrics(load(directory / "eval/summary.json"))


def class_counts(path):
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    dead = [r for r in rows if int(r["cls"]) == 4]
    return {"gt_nuclei": len(rows), "dead_nuclei": len(dead), "dead_images": len({r["image"] for r in dead})}


def validate_bootstrap(data, root, split, base, fusion):
    """Bind reported intervals to this split's evaluated reference and candidate."""
    if data.get("fold") != {1: 3, 2: 3, 3: 1}[split] or data.get("reference") != "base_ms":
        raise ValueError("bootstrap fold/reference mismatch")
    if data.get("boot") != 2000:
        raise ValueError("bootstrap replicate count differs from the reported protocol")
    keys = {"mPQ": ("mPQ", 0), "bPQ": ("bPQ", 1), "Dead": ("DeadPQ", 5)}
    for name, path, expected in (("base_ms", f"p0/split{split}/test/base/ms/eval", base),
                                 ("fusion", f"p2/split{split}/test/eval", fusion)):
        group = data["groups"][name]
        runs = [p if p.is_absolute() else ROOT / p for p in map(Path, group["runs"])]
        if [p.resolve() for p in runs] != [(Path(root) / path).resolve()]:
            raise ValueError("bootstrap runs differ from the reported P0/P2 comparison")
        if len(group["per_run"]) != 1 or len(group["per_run"][0]) != 7:
            raise ValueError("bootstrap must use one checkpoint per split")
        for key, (metric, index) in keys.items():
            for point in (group["mean"][key], group["per_run"][0][index]):
                if not math.isclose(point, expected[metric], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("bootstrap point estimate differs from canonical evaluation")
    group = data["groups"]["fusion"]
    for key, (metric, _) in keys.items():
        if not math.isclose(group["delta"][key], fusion[metric] - base[metric], rel_tol=0, abs_tol=1e-12):
            raise ValueError("bootstrap delta differs from the reported comparison")
        ci = group["ci95"][key]
        if len(ci) != 2 or not all(math.isfinite(v) for v in ci) or ci[0] > ci[1]:
            raise ValueError("invalid bootstrap confidence interval")


def collect(root):
    root = Path(root)
    p0, choices = {}, {}
    for arm in ("base", "x2"):
        for variant in ("identity", "ms"):
            rows = {}
            for s in (1, 2, 3):
                directory = root / f"p0/split{s}/test/{arm}/{variant}"
                manifest = load(directory / "manifest.json")
                if manifest.get("arm") != arm or manifest.get("variant") != variant:
                    raise ValueError("P0 directory/manifest arm mismatch")
                validate_selection_link(root / f"p0/split{s}/selection.json", manifest, s, "p0")
                rows[s] = verified_summary(directory)
            p0[f"{arm}_{variant}"] = {"splits": rows, "mean": mean_complete(rows)}
    for s in (1, 2, 3):
        choices[s] = {k: v["selected"] for k, v in load(root / f"p0/split{s}/selection.json")["arms"].items()}
    p1 = {str(s): {role: load(root / f"p1/split{s}_{role}.json") for role in ("val", "test")}
          for s in (1, 2, 3)}
    # Match diagnostic light-stat summaries to independent canonical evaluations of identical inputs.
    for s in (1, 2, 3):
        for role in ("val", "test"):
            diag = p1[str(s)][role]
            if diag["split"] != s or diag["role"] != role:
                raise ValueError("P1 artifact role/split mismatch")
            for name, variant in (("base", "identity"), ("x2", "ms")):
                expected = metrics(load(root / f"p0/split{s}/{role}/{name}/{variant}/eval/summary.json"))
                actual = diag["scores"][name]
                for key in ("mPQ", "bPQ"):
                    if abs(actual[key] - expected[key]) > 1e-12:
                        raise ValueError(f"P1/canonical mismatch: split{s} {role} {name} {key}")
    p2_rows, p2_choices = {}, {}
    for s in (1, 2, 3):
        directory = root / f"p2/split{s}/test"
        manifest = load(directory / "manifest.json")
        choice = validate_selection_link(root / f"p2/split{s}/selection.json", manifest, s, "p2")
        for role, evidence in (("test", manifest["provenance"]), ("val", choice["provenance"])):
            base_path = root / f"p0/split{s}/{role}/base/ms/pred.npz"
            if evidence["inputs"]["base"]["sha256"] != sha256(base_path):
                raise ValueError("P2 reference is not the reported corrected P0 base/MS")
        p2_rows[s] = verified_summary(directory)
        p2_choices[s] = choice
    p2 = {"splits": p2_rows, "mean": mean_complete(p2_rows), "selections": p2_choices}
    p3 = {}
    manifest = root / "p3/indices_fold2.json"
    for train in (1, 2):
        for infer in (1, 2):
            d = root / f"p3/train{train}_infer{infer}"
            m = load(d / "metadata.json")
            if (m["train_upscale"], m["infer_upscale"], m["n_images"], m["fold"], m["split"], m["seed"]) != (train, infer, 128, 2, 1, 19):
                raise ValueError("P3 scope metadata mismatch")
            if m["indices_sha256"] != sha256(manifest) or m["prediction_sha256"] != sha256(d / "pred.npz"):
                raise ValueError("P3 manifest/prediction hash mismatch")
            p3[f"train{train}_infer{infer}"] = {"metrics": metrics(load(d / "eval/summary.json")), "metadata": m,
                                                 "sample_counts": class_counts(d / "eval/gt_records.csv.gz")}
    diagnostic_summary = {
        "base_interior_dead_matched_rate": mean(p1[str(s)]["test"]["diagnostics"]["base_dead"]["interior_matched"] /
                                                 p1[str(s)]["test"]["diagnostics"]["base_dead"]["interior_gt"] for s in (1, 2, 3)),
        "x2_interior_dead_matched_rate": mean(p1[str(s)]["test"]["diagnostics"]["x2_dead"]["interior_matched"] /
                                               p1[str(s)]["test"]["diagnostics"]["x2_dead"]["interior_gt"] for s in (1, 2, 3)),
        "type_swap_mpq": {name: mean(p1[str(s)]["test"]["scores"][name]["mPQ"] for s in (1, 2, 3))
                          for name in ("base", "x2", "base_geometry_x2_types", "x2_geometry_base_types")}}
    bootstraps = {}
    for s in (1, 2, 3):
        p = root / f"p2/bootstrap_split{s}.json"
        b = load(p)
        validate_bootstrap(b, root, s, p0["base_ms"]["splits"][s], p2_rows[s])
        bootstraps[s] = {"delta": b["groups"]["fusion"]["delta"],
                         "ci95": b["groups"]["fusion"]["ci95"], "n_boot": b["boot"],
                         "source_sha256": sha256(p), "scope": "paired image sampling, not training-seed uncertainty"}
    return {"scope": "P0/P1/P2 seed19 three official splits; P3 split1 validation-only 128 images; no training",
            "seed_coverage": seed_coverage(actual_seed_records(root)), "p0": p0, "p0_selections": choices,
            "p1": p1, "p1_summary": diagnostic_summary, "p2": p2, "p2_bootstrap": bootstraps,
            "p2_delta_vs_base_ms": {k: p2["mean"][k] - p0["base_ms"]["mean"][k] for k in METRICS}, "p3": p3}


def render_tables(d):
    lines = ["# P0–P3 artifact-derived tables", "", d["scope"], "",
             "## P0: corrected controls (three-split mean)", "",
             "| Arm | mPQ | bPQ | mPQ+ | Dead PQ | strict Dead PQ |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in d["p0"].items():
        r = row["mean"]
        lines.append(f"| {name} | " + " | ".join(f"{r[k]:.4f}" for k in METRICS[:5]) + " |")
    lines += ["", "## P0: validation-selected parameters", "", "| Split | base frac / area | x2 frac / area |", "|---|---|---|"]
    for s, choices in d["p0_selections"].items():
        lines.append(f"| {s} | {choices['base']['frac']} / {choices['base']['a_min']} | {choices['x2']['frac']} / {choices['x2']['a_min']} |")
    lines += ["", "## P1: validation candidate pool (not deployment results)", "",
              "| Split | Disjoint | Typed TP | Typed FP | Predicted Dead | Typed Dead TP | Oracle ΔmPQ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s, pair in d["p1"].items():
        v = pair["val"]; c = v["diagnostics"]["candidates"]
        lines.append(f"| {s} | {c['all_disjoint']} | {c['typed_tp']} | {c['typed_fp']} | {c['predicted_dead']} | {c['typed_dead_tp']} | {v['scores']['oracle_add']['mPQ']-v['scores']['base']['mPQ']:+.4f} |")
    lines += ["", "## P2: frozen choices and test results", "",
              "| Split | Validation verdict | Choice | Test mPQ | Test bPQ | Test Dead PQ | strict Dead PQ |", "|---|---|---|---:|---:|---:|---:|"]
    for s in (1, 2, 3):
        sel = d["p2"]["selections"][s]; r = d["p2"]["splits"][s]
        lines.append(f"| {s} | {sel['status']} | {sel['selected']['name']} | {r['mPQ']:.4f} | {r['bPQ']:.4f} | {r['DeadPQ']:.4f} | {r['strictDeadPQ']:.4f} |")
    r = d["p2"]["mean"]
    lines.append("| Mean | — | — | " + " | ".join(f"{r[k]:.4f}" for k in ("mPQ", "bPQ", "DeadPQ", "strictDeadPQ")) + " |")
    lines += ["", "P2 reference is corrected base/MS; candidates use original du2 probability tables. NO_GO selects identity, not a test-tuned fallback.",
              "", "| Split | ΔmPQ 95% CI | ΔbPQ 95% CI | ΔDead PQ 95% CI |", "|---|---|---|---|"]
    for s, b in d["p2_bootstrap"].items():
        lines.append(f"| {s} | " + " | ".join(f"[{b['ci95'][k][0]:+.4f}, {b['ci95'][k][1]:+.4f}]" for k in ("mPQ", "bPQ", "Dead")) + " |")
    lines += ["", "CIs condition on one checkpoint per split; no pooled-independent-split or training-seed inference is made.",
              "", "## P3: fixed 128-image validation subset", "",
              "| Train / infer scale | mPQ | bPQ | Dead PQ | strict Dead PQ | GPU allocated peak MiB |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in d["p3"].items():
        r = row["metrics"]
        lines.append(f"| {name} | {r['mPQ']:.4f} | {r['bPQ']:.4f} | {r['DeadPQ']:.4f} | {r['strictDeadPQ']:.4f} | {row['metadata']['peak_cuda_allocated_bytes']/2**20:.1f} |")
    counts = next(iter(d["p3"].values()))["sample_counts"]
    lines += ["", f"Subset: {counts['gt_nuclei']} GT nuclei; {counts['dead_nuclei']} Dead nuclei in {counts['dead_images']} images.",
              "P3 is not a full-fold test or a three-split result. Class-specific numbers on this small subset are descriptive only.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "runs/analysis/p0p3_20261002_v2")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-markdown", type=Path, required=True)
    a = p.parse_args()
    if a.out_json.exists() or a.out_markdown.exists():
        raise FileExistsError("refusing to overwrite generated result files")
    data = collect(a.root)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    with a.out_json.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    with a.out_markdown.open("x") as f:
        f.write(render_tables(data))
    print(a.out_json); print(a.out_markdown)


if __name__ == "__main__":
    main()
