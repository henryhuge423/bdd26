import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FILE = Path(__file__)
SPLITS = {1: 3, 2: 3, 3: 1}  # split -> test fold (official 1/2/3, 2/1/3, 3/2/1 mapping)
SEEDS = (19, 1, 2)
BASE = {"mPQ": .50, "bPQ": .66, "mPQ+": .515, "strict mPQ": .45, "Dead PQ": .15, "strict Dead PQ": .11}


def write_eval(directory: Path, values: dict, n_images: int = 6, dead_images=(0, 3)):
    directory.mkdir(parents=True)
    summary = {"n_images": n_images, "official": {"mPQ": values["mPQ"], "bPQ": values["bPQ"]},
               "strict": {"mPQ": values["strict mPQ"]}, "plus": {"mPQ+": values["mPQ+"]},
               "per_class_PQ": {"Dead": values["Dead PQ"]},
               "per_class_PQ_strict": {"Dead": values["strict Dead PQ"]}}
    (directory / "summary.json").write_text(json.dumps(summary))
    m = np.full(n_images, values["mPQ"])
    b = np.full(n_images, values["bPQ"])
    dead = np.full(n_images, np.nan)
    dead[list(dead_images)] = values["Dead PQ"]
    np.savez_compressed(directory / "per_image.npz", mPQ=m, bPQ=b, class_PQ=np.column_stack([dead] * 5))


def make_root(base: Path, dead_delta, mpq_delta=0., bpq_delta=0.):
    """dead_delta: {(split, seed): value}; other endpoints shift uniformly."""
    root = base / "matched"
    for split, fold in SPLITS.items():
        d = base / "data" / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        n = 6
        np.save(d / "inst.npy", np.zeros((n, 8, 8), np.int32))
        np.save(d / "type.npy", np.zeros((n, 8, 8), np.int32))
        np.save(d / "tissue.npy", np.array(["Lung"] * 3 + ["Colon"] * 3))
        np.savez_compressed(d / "gt_channels.npz", gt=np.zeros((n, 8, 8, 5), np.int32))
        for seed in SEEDS:
            pair = f"split{split}_seed{seed}"
            write_eval(root / "p0" / pair / "test" / "base" / "identity" / "eval", BASE)
            write_eval(root / "p0" / pair / "test" / "base" / "ms" / "eval", BASE)
            p2 = dict(BASE, **{"mPQ": BASE["mPQ"] + mpq_delta, "bPQ": BASE["bPQ"] + bpq_delta,
                               "Dead PQ": BASE["Dead PQ"] + dead_delta[(split, seed)]})
            write_eval(root / "p2" / pair / "test" / "eval", p2)
    return root


def run_script(root: Path, base: Path, out: Path = None, boot=40):
    out = out or (base / "stats")
    env = {**os.environ, "PANNUKE_ROOT": str(base / "data"), "CUDA_VISIBLE_DEVICES": ""}
    script = FILE.parents[1] / "scripts" / "matched_seed_stats.py"
    r = subprocess.run([sys.executable, str(script), "--root", str(root), "--out", str(out),
                        "--boot", str(boot)], capture_output=True, text=True, env=env, timeout=120)
    return r, out


def test_stats_tables_deltas_sign_counts_and_bootstrap_determinism(tmp_path):
    dead = {(1, 19): .01, (1, 1): .012, (1, 2): .011, (2, 19): -.004, (2, 1): .006,
            (2, 2): .005, (3, 19): .02, (3, 1): .021, (3, 2): .019}
    root = make_root(tmp_path, dead)
    r, out = run_script(root, tmp_path)
    assert r.returncode == 0, r.stderr
    result = json.loads((out / "stats.json").read_text())
    # per-pair deltas
    assert result["pairs"]["split1_seed19"]["baseline"] == BASE
    assert result["pairs"]["split1_seed19"]["delta"]["Dead PQ"] == pytest.approx(.01)
    assert result["pairs"]["split2_seed19"]["delta"]["Dead PQ"] == pytest.approx(-.004)
    # per-split seed means / stds / sign counts (population std over 3 seeds)
    s1 = result["splits"]["1"]
    assert s1["seed_mean"]["Dead PQ"] == pytest.approx(np.mean([.01, .012, .011]))
    assert s1["seed_std"]["Dead PQ"] == pytest.approx(np.std([.01, .012, .011]))
    assert s1["sign_counts"]["Dead PQ"] == {"nonneg": 3, "neg": 0}
    assert result["splits"]["2"]["sign_counts"]["Dead PQ"] == {"nonneg": 2, "neg": 1}
    assert result["overall"]["sign_counts"]["Dead PQ"] == {"nonneg": 8, "neg": 1}
    # decision rule: Dead mean .01156 >= max std .000943..., 8/9 >= 0, mPQ/bPQ deltas 0
    assert result["decision"]["robust_improvement"] is True
    # bootstrap deterministic + contains point estimate
    b1 = json.loads((out / "bootstrap_split1.json").read_text())
    assert b1["Dead PQ"]["ci95"][0] <= b1["Dead PQ"]["point"] <= b1["Dead PQ"]["ci95"][1]
    r2, out2 = run_script(root, tmp_path, out=tmp_path / "stats_again")
    b1b = json.loads((out2 / "bootstrap_split1.json").read_text())
    assert b1 == b1b
    assert r2.returncode == 0, r2.stderr


def test_decision_rule_fails_on_dead_std_or_sign_or_mpq_guard(tmp_path):
    # Dead gain large but one split very noisy -> fails the >= 1x max seed std arm
    noisy = {(1, 19): .01, (1, 1): .05, (1, 2): .012, (2, 19): .006, (2, 1): .005,
             (2, 2): .007, (3, 19): .02, (3, 1): .021, (3, 2): .019}
    root = make_root(tmp_path, noisy)
    r, out = run_script(root, tmp_path)
    assert r.returncode == 0, r.stderr
    d = json.loads((out / "stats.json").read_text())["decision"]
    assert d["robust_improvement"] is False and any("std" in f for f in d["failed"])
    # enough negative pairs -> sign arm fails
    signs = {(1, 19): -.01, (1, 1): .01, (1, 2): .01, (2, 19): -.01, (2, 1): .01,
             (2, 2): .01, (3, 19): -.01, (3, 1): .02, (3, 2): .019}
    root2 = make_root(tmp_path / "b", signs)
    r2, out2 = run_script(root2, tmp_path / "b")
    assert r2.returncode == 0, r2.stderr
    d2 = json.loads((out2 / "stats.json").read_text())["decision"]
    assert d2["robust_improvement"] is False and any("7/9" in f for f in d2["failed"])
    # mPQ guard: uniform mPQ cost beyond -1x its (zero) std
    root3 = make_root(tmp_path / "c", {(s, e): .02 for s in (1, 2, 3) for e in SEEDS}, mpq_delta=-.001)
    r3, out3 = run_script(root3, tmp_path / "c")
    assert r3.returncode == 0, r3.stderr
    d3 = json.loads((out3 / "stats.json").read_text())["decision"]
    assert d3["robust_improvement"] is False and any("mPQ" in f for f in d3["failed"])


def test_bootstrap_resamples_absolute_indices_within_tissue():
    """Unequal tissue sizes and levels: resampling positions instead of absolute
    image indices silently reads the wrong tissue and shifts the bootstrap center."""
    from scripts.matched_seed_stats import bootstrap
    import warnings
    tissue = np.array(["Lung"] * 4 + ["Colon"] * 2)
    # mPQ deltas: Lung images vary around mean +.02, Colon around -.04; bPQ/Dead unused
    d = np.zeros((6, 3))
    d[:4, 0] = [.04, .03, .02, -.01]
    d[4:, 0] = [.00, -.08]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = bootstrap({19: d, 1: d}, tissue, boot=2000, seed=7)
    # the official pooled delta weights tissues equally: mean(+.02, -.04) = -.01
    series_mean = (out["mPQ"]["ci95"][0] + out["mPQ"]["ci95"][1]) / 2
    assert abs(series_mean - (-.01)) < .01
    ci = out["mPQ"]["ci95"]
    assert ci[0] <= -.01 <= ci[1]


def test_stats_refuse_missing_pair_or_inconsistent_summary(tmp_path):
    dead = {(s, e): .01 for s in (1, 2, 3) for e in SEEDS}
    root = make_root(tmp_path, dead)
    r, out = run_script(root, tmp_path)
    assert r.returncode == 0
    # a second run over the same root must refuse to overwrite
    r_again, _ = run_script(root, tmp_path, out=tmp_path / "stats")
    assert r_again.returncode != 0
    # missing pair directory
    import shutil
    shutil.rmtree(root / "p2" / "split3_seed2")
    r_missing, _ = run_script(root, tmp_path, out=tmp_path / "stats_missing")
    assert r_missing.returncode != 0 and "split3_seed2" in r_missing.stderr
    # summary that disagrees with its per_image arrays
    root3 = make_root(tmp_path / "d", dead)
    p = root3 / "p2" / "split1_seed19" / "test" / "eval"
    s = json.loads((p / "summary.json").read_text())
    s["official"]["mPQ"] += .1
    (p / "summary.json").write_text(json.dumps(s))
    r_bad, _ = run_script(root3, tmp_path / "d", out=tmp_path / "stats_bad")
    assert r_bad.returncode != 0 and "mPQ" in r_bad.stderr
