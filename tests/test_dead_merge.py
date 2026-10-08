"""EF-style dead-candidate merge and the frozen dev-fold menu selection (DSB plan Task 6)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nucseg.postproc.dead_merge import merge_dead

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _maps():
    base = np.zeros((16, 16), np.int32)
    base[0:4, 0:4] = 1                                      # base instance
    btyp = np.zeros((16, 16), np.uint8)
    btyp[base == 1] = 2
    dead = np.zeros((16, 16), np.int32)
    dead[8:11, 8:11] = 1                                    # disjoint candidate, area 9
    dead[12:15, 2:4] = 2                                    # disjoint candidate, area 6
    dead[0:3, 0:3] = 3                                      # overlaps the base -> always rejected
    return base, btyp, dead


def test_merge_dead_rules():
    base, btyp, dead = _maps()
    out, typ, chosen = merge_dead(base, btyp, dead, min_area=0)
    assert (out[base > 0] == base[base > 0]).all()          # base pixels byte-identical
    assert typ[base > 0].tolist() == btyp[base > 0].tolist()
    assert 3 not in chosen and set(chosen) == {1, 2}        # id 3 overlaps the base
    new = set(np.unique(out)) - set(np.unique(base))
    assert new == {2, 3}                                    # renumbered above base.max()
    assert (typ[out == 2] == 4).all() and (typ[out == 3] == 4).all()  # candidates typed Dead
    _, _, chosen30 = merge_dead(base, btyp, dead, min_area=30)
    assert chosen30 == []                                   # both candidates below the floor


def test_selection_rule_frozen():
    cli = _load("dsb_dev_eval")
    rows = [{"name": "identity", "d_dead": 0.0, "d_bpq": 0.0},
            {"name": "a30", "d_dead": 0.006, "d_bpq": -0.0008},
            {"name": "a60", "d_dead": 0.008, "d_bpq": -0.002}]
    assert cli.select_menu(rows) == "a30"   # a60's larger Dead gain fails the -0.001 bPQ guard
    rows[1]["d_dead"] = -0.001
    assert cli.select_menu(rows) == "identity"  # nothing beats identity -> identity fallback


def _dev_fixture(tmp_path):
    root = tmp_path / "data"
    d = root / "fold2"
    d.mkdir(parents=True)
    gi = np.zeros((2, 32, 32), np.int32)
    gt = np.zeros((2, 32, 32), np.uint8)
    gi[0, 4:8, 4:8] = 1; gt[0, 4:8, 4:8] = 4        # img0 Dead GT, missed by base (area 16)
    gi[0, 20:28, 20:28] = 2; gt[0, 20:28, 20:28] = 2
    gi[1, 4:10, 4:10] = 1; gt[1, 4:10, 4:10] = 4    # img1 Dead GT, missed by base (area 36)
    np.save(d / "inst.npy", gi)
    np.save(d / "type.npy", gt)
    np.save(d / "tissue.npy", np.array(["Colon", "Colon"]))
    np.savez_compressed(d / "gt_channels.npz",
                        gt=np.stack([np.where(gt == c, gi, 0) for c in range(1, 6)], -1))
    bi = np.zeros((2, 32, 32), np.int32)
    bt = np.zeros((2, 32, 32), np.uint8)
    bi[0, 20:28, 20:28] = 1; bt[0, 20:28, 20:28] = 2  # base matches the non-Dead GT only
    dead = np.zeros((2, 32, 32), np.int32)
    dead[0, 4:8, 4:8] = 1                            # area 16: true recovery
    dead[0, 12:19, 12:19] = 2                        # area 49: false positive
    dead[1, 4:10, 4:10] = 1                          # area 36: true recovery
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"split": 1}))
    np.savez_compressed(run / "pred_fold2.npz", inst=bi, type=bt)
    np.savez_compressed(run / "pred_fold2_dead.npz", dead_inst=dead)
    return root, run


def test_dev_eval_cli_end_to_end(tmp_path):
    root, run = _dev_fixture(tmp_path)
    cmd = [sys.executable, str(SCRIPTS / "dsb_dev_eval.py"), "--run", str(run), "--fold", "2"]
    env = {**os.environ, "PANNUKE_ROOT": str(root), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads((run / "dsb_dev_eval" / "menu.json").read_text())
    rows = {row["name"]: row for row in out["rows"]}
    assert rows["identity"]["n_added"] == 0
    assert rows["a30"]["n_added"] == 2      # the area-36 recovery + the area-49 FP survive the floor
    assert rows["a60"]["n_added"] == 0      # both recoveries are below 60
    assert rows["a30"]["d_dead"] > 0 and rows["a30"]["d_bpq"] >= -0.001
    assert out["selection"] == "a30"
    assert set(out["inputs"]) == {"base", "dead"}                     # sha256 provenance bound
    assert all(len(v["sha256"]) == 64 for v in out["inputs"].values())
    first = (run / "dsb_dev_eval" / "menu.json").read_bytes()
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    assert r2.returncode != 0 and "frozen" in r2.stderr               # refuse-overwrite guard
    assert (run / "dsb_dev_eval" / "menu.json").read_bytes() == first
    shutil.rmtree(run / "dsb_dev_eval")
    subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120, check=True)
    assert (run / "dsb_dev_eval" / "menu.json").read_bytes() == first  # bit-exact regeneration
