import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FILE = Path(__file__)
SCRIPTS = FILE.parents[1] / "scripts"


def maps():
    base = np.zeros((12, 12), np.int32)
    base[2:5, 2:5] = 8
    bt = np.where(base > 0, 1, 0).astype(np.uint8)
    x = base.copy()
    x[7:9, 7:9] = 23  # 4 px Dead addition, interior
    xt = np.where(x == 23, 4, bt).astype(np.uint8)
    return base, bt, x, xt


def make_fixture(tmp_path):
    b, bt, x, xt = maps()
    gi, gt = np.stack([x, b]), np.stack([xt, bt])  # image 1 has no addition (prob table is sparse)
    bi, btyp = np.stack([b, b]), np.stack([bt, bt])
    data = tmp_path / "data"
    for fold in (2, 3):
        d = data / f"fold{fold}"
        d.mkdir(parents=True)
        np.save(d / "inst.npy", gi)
        np.save(d / "type.npy", gt)
        np.save(d / "tissue.npy", np.array(["Lung", "Colon"]))
        np.savez_compressed(d / "gt_channels.npz",
                            gt=np.stack([np.where(gt == c, gi, 0) for c in range(1, 6)], -1))
    inputs = {}
    for name, inst, typ in (("base", bi, btyp), ("x2", gi, gt)):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"split": 1, "seed": 19}))
        for role in ("val", "test"):
            path = d / f"{role}.npz"
            if name == "x2":
                np.savez_compressed(path, inst=inst, type=typ, inst_img=[0, 0, 1],
                                    inst_id=[8, 23, 8], inst_prob=np.eye(6)[[1, 4, 1]])
            else:
                np.savez_compressed(path, inst=inst, type=typ)
            inputs[f"{name}-{role}"] = path
    return data, inputs


def ef_cmd(inputs, out, selection, stage, extra=()):
    cmd = [sys.executable, str(SCRIPTS / "run_existence_ef.py"), "--split", "1", "--workers", "1",
           "--p2-selection", str(selection), "--out", str(out), "--stage", stage, *extra]
    for k, v in inputs.items():
        cmd += ["--" + k, str(v)]
    return cmd


def run_p2_val(data, inputs, tmp_path):
    out = tmp_path / "p2"
    cmd = [sys.executable, str(SCRIPTS / "run_scale_fusion.py"), "--split", "1",
           "--workers", "1", "--out", str(out), "--stage", "val"]
    for k, v in inputs.items():
        cmd += ["--" + k, str(v)]
    env = {**os.environ, "PANNUKE_ROOT": str(data), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    return out, env


def test_ef_cli_end_to_end_menu_off_guard_and_frozen_input_binding(tmp_path):
    data, inputs = make_fixture(tmp_path)
    p2, env = run_p2_val(data, inputs, tmp_path)
    p2sel = json.loads((p2 / "selection.json").read_text())
    assert p2sel["status"] == "GO"  # the frozen P2 menu keeps the 4 px Dead addition

    out = tmp_path / "ef"
    r = subprocess.run(ef_cmd(inputs, out, p2 / "selection.json", "val"),
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    sel = json.loads((out / "selection.json").read_text())
    rows = {row["name"]: row for row in sel["rows"]}
    # off must reproduce the frozen stage-1 row bit-for-bit (stage-1 reimplementation guard)
    for k in ("mPQ", "bPQ", "strict_dead", "interior_matched", "added"):
        assert rows["off"][k] == pytest.approx(p2sel["selected"][k], abs=1e-12), k
    # the 4 px Dead addition: kept by off/a30d (Dead-exempt), dropped by a30/a60/identity
    assert rows["identity"]["added"] == 0
    assert rows["a30"]["added"] == 0
    assert rows["a60"]["added"] == 0
    assert rows["a30d"]["added"] == 1 and rows["off"]["added"] == 1
    # only rows that still recover the Dead nucleus pass the gate; equal (mPQ, added) -> name
    assert sel["selected"]["name"] == "off"

    original = inputs["x2-val"].read_bytes()
    inputs["x2-val"].write_bytes(original + b"changed")
    bad = subprocess.run(ef_cmd(inputs, out, p2 / "selection.json", "test"),
                         capture_output=True, text=True, env=env, timeout=60)
    assert bad.returncode != 0 and not (out / "test").exists()
    inputs["x2-val"].write_bytes(original)
    good = subprocess.run(ef_cmd(inputs, out, p2 / "selection.json", "test"),
                          capture_output=True, text=True, env=env, timeout=60)
    assert good.returncode == 0, good.stderr
    score = json.loads((out / "test/eval/summary.json").read_text())
    assert score["official"]["mPQ"] == pytest.approx(1, abs=1e-6)


def test_ef_val_refuses_doctored_stage1_row(tmp_path):
    data, inputs = make_fixture(tmp_path)
    p2, env = run_p2_val(data, inputs, tmp_path)
    path = p2 / "selection.json"
    record = json.loads(path.read_text())
    record["selected"]["mPQ"] += 0.01  # a stage-1 row the inputs cannot reproduce
    path.write_text(json.dumps(record))
    r = subprocess.run(ef_cmd(inputs, tmp_path / "ef", path, "val"),
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode != 0 and "off" in r.stderr
