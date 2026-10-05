"""Scale fusion cannot change base instances or use a shifted probability-table row."""
import importlib
from pathlib import Path

import numpy as np
import pytest

FILE = Path(__file__).resolve().parents[1] / "src/nucseg/postproc/scale_fusion.py"


def module():
    assert FILE.exists(), "scale fusion implementation missing"
    return importlib.import_module("nucseg.postproc.scale_fusion")


def maps():
    base = np.zeros((12, 12), np.int32)
    base[2:5, 2:5] = 8
    bt = np.where(base > 0, 1, 0).astype(np.uint8)
    x = base.copy()
    x[7:9, 7:9] = 23
    xt = np.where(x == 23, 4, bt).astype(np.uint8)
    return base, bt, x, xt


def test_addition_preserves_every_base_pixel_and_class():
    b, bt, x, xt = maps()
    out, typ, chosen = module().fuse(b, bt, x, xt, {23: .8, 8: .9}, 100, .7, True)
    assert np.array_equal(out[b > 0], b[b > 0])
    assert np.array_equal(typ[b > 0], bt[b > 0])
    assert np.all(out[x == 23] > 0) and np.all(typ[x == 23] == 4)
    assert chosen == [23]


def test_candidate_overlapping_base_is_never_clipped_or_added():
    b, bt, x, xt = maps()
    x[4:8, 4:8] = 23
    out, _, chosen = module().fuse(b, bt, x, xt, {23: .99, 8: .99}, None, 0, False)
    assert np.array_equal(out, b) and chosen == []


@pytest.mark.parametrize("area,prob,interior", [(3, .7, True), (100, .9, True)])
def test_area_and_actual_id_confidence_are_applied(area, prob, interior):
    b, bt, x, xt = maps()
    out, _, chosen = module().fuse(b, bt, x, xt, {23: .8, 8: .99}, area, prob, interior)
    assert np.array_equal(out, b) and chosen == []


def test_border_filter_and_empty_maps():
    m = module()
    z = np.zeros((8, 8), np.int32)
    x = z.copy(); x[0:2, 3:5] = 99
    typ = np.where(x > 0, 4, 0).astype(np.uint8)
    assert m.fuse(z, z, x, typ, {99: .9}, 100, .7, True)[2] == []
    assert m.fuse(z, z, x, typ, {99: .9}, 100, .7, False)[2] == [99]
    assert np.array_equal(m.fuse(z, z, z, z, {}, None, 0, False)[0], z)


def test_confidence_table_is_keyed_by_image_and_sparse_id():
    m = module()
    table = {"inst_img": np.array([1, 0, 0]), "inst_id": np.array([23, 23, 8]),
             "inst_prob": np.array([[.1,.1,.1,.1,.5,.1], [.1,.1,.1,.1,.5,.1], [.1,.5,.1,.1,.1,.1]])}
    rows = m.probability_rows(table)
    assert np.array_equal(rows[0][23], table["inst_prob"][1])
    bad = {**table, "inst_img": np.array([0, 0, 0])}
    with pytest.raises(ValueError):
        m.probability_rows(bad)


def test_type_transfer_never_changes_geometry_or_unmatched_type():
    b, bt, x, xt = maps()
    xt[x == 8] = 2
    out, coverage = module().transfer_types(b, bt, x, xt)
    assert np.all(out[b == 8] == 2) and coverage == 1
    assert np.all(out[b == 0] == 0)


def test_oracle_adds_only_type_correct_gt_matches():
    b, bt, x, xt = maps()
    gt = x.copy(); gt_type = xt.copy()
    out, typ, chosen = module().oracle_additions(b, bt, x, xt, gt, gt_type)
    assert chosen == [23]
    wrong = xt.copy(); wrong[x == 23] = 1
    assert module().oracle_additions(b, bt, x, wrong, gt, gt_type)[2] == []


def test_strict_dead_summary_counts_class_absent_predictions():
    m = module()
    stats = np.zeros((2, 6, 5))
    stats[0, 4] = [1, 0, 0, 1, 1]
    stats[1, 4] = [0, 1, 0, 0, 0]
    assert m.strict_dead_pq(stats) == pytest.approx(.5, abs=1e-6)


def test_duplicate_seed_labels_are_not_needed_for_per_image_ops():
    b, bt, x, xt = maps()
    m = module()
    a = m.fuse(b, bt, x, xt, {23: .8, 8: .9}, 100, .7, True)
    x[x == 23] = 3
    c = m.fuse(b, bt, x, xt, {3: .8, 8: .9}, 100, .7, True)
    assert np.array_equal(a[0] > 0, c[0] > 0)
    assert np.array_equal(a[1], c[1])


def test_diagnostics_find_interior_dead_recovery_and_negative_image_fp():
    b, bt, x, xt = maps()
    m = module()
    d = m.image_diagnostics(x, xt, b, bt, x, xt)
    assert d["base_dead"]["interior_gt"] == 1
    assert d["base_dead"]["interior_matched"] == 0
    assert d["x2_dead"]["interior_matched"] == 1
    assert d["candidates"]["typed_dead_tp"] == 1
    negative = m.image_diagnostics(b, bt, b, bt, x, xt)
    assert negative["x2_dead"]["negative_images"] == 1
    assert negative["x2_dead"]["dead_predictions_on_negative"] == 1


def test_deployment_selection_requires_strict_score_and_recall_guards():
    m = module()
    base = {"name": "identity", "mPQ": .5, "bPQ": .6, "strict_dead": .2,
            "interior_matched": 10, "added": 0}
    harmful = {**base, "name": "bad", "mPQ": .51, "strict_dead": .19,
               "interior_matched": 11, "added": 2}
    good = {**base, "name": "good", "mPQ": .501, "interior_matched": 11, "added": 1}
    assert m.select_fusion([base, harmful])["name"] == "identity"
    assert m.select_fusion([base, harmful, good])["name"] == "good"


def test_p1_scores_match_canonical_per_image_statistics():
    from scripts.analyze_scale_complementarity import run_image
    from nucseg.metrics.pannuke_eval import _eval_image
    b, bt, x, xt = maps()
    ch = np.stack([np.where(xt == c, x, 0) for c in range(1, 6)], -1)
    diag, stats, coverage = run_image(x, xt, ch, b, bt, x, xt)
    canonical = _eval_image((ch, x, xt, x, xt))
    expected = np.vstack([canonical["bin"][:5], np.asarray(canonical["cls"])[:, :5]])
    assert np.allclose(stats[1], expected)
    assert coverage[0] == 1


def test_p2_probability_mapping_uses_predicted_class_for_actual_id():
    from scripts.run_scale_fusion import image_confidence
    b, bt, x, xt = maps()
    rows = {8: np.array([.1,.5,.1,.1,.1,.1]), 23: np.array([.1,.1,.1,.1,.5,.1])}
    assert image_confidence(x, xt, rows) == {8: .5, 23: .5}
    with pytest.raises(ValueError):
        image_confidence(x, xt, {8: rows[8]})


def test_prediction_context_rejects_wrong_sidecar_hash(tmp_path):
    from scripts import analyze_scale_complementarity as a
    import json
    pred = tmp_path / "pred.npz"
    pred.write_bytes(b"example")
    context = {"actual_seed": 19, "split": 1, "prediction": {"sha256": "wrong"}}
    (tmp_path / "manifest.json").write_text(json.dumps(context))
    with pytest.raises(ValueError):
        a.prediction_context(pred)


def test_analysis_rejects_wrong_model_split_or_unknown_seed():
    from scripts import analyze_scale_complementarity as a
    good = {"run_context": {"base": {"split": 1, "actual_seed": 19}, "x2": {"split": 1, "actual_seed": 19}}}
    a.validate_run_context(good, 1)
    with pytest.raises(ValueError):
        a.validate_run_context(good, 2)
    good["run_context"]["x2"]["actual_seed"] = None
    with pytest.raises(ValueError):
        a.validate_run_context(good, 1)


def test_analysis_binds_run_context_to_declared_pair_seed():
    from scripts import analyze_scale_complementarity as a
    pair = {"run_context": {"base": {"split": 2, "actual_seed": 1}, "x2": {"split": 2, "actual_seed": 1}}}
    a.validate_run_context(pair, 2, expected_seed=1)
    with pytest.raises(ValueError):
        a.validate_run_context(pair, 2, expected_seed=2)
    pair["run_context"]["x2"]["actual_seed"] = 2
    with pytest.raises(ValueError):
        a.validate_run_context(pair, 2, expected_seed=1)


def test_p1_p2_cli_end_to_end_and_frozen_input_guard(tmp_path):
    import json
    import os
    import subprocess
    import sys
    b, bt, x, xt = maps()
    gi, gt = np.stack([x, b]), np.stack([xt, bt])
    bi, btyp = np.stack([b, b]), np.stack([bt, bt])
    data = tmp_path / "data"
    for fold in (2, 3):
        d = data / f"fold{fold}"; d.mkdir(parents=True)
        np.save(d / "inst.npy", gi); np.save(d / "type.npy", gt)
        np.save(d / "tissue.npy", np.array(["Lung", "Colon"]))
        np.savez_compressed(d / "gt_channels.npz", gt=np.stack([np.where(gt == c, gi, 0) for c in range(1, 6)], -1))
    inputs = {}
    for name, inst, typ in (("base", bi, btyp), ("x2", gi, gt)):
        d = tmp_path / name; d.mkdir()
        (d / "config.json").write_text(json.dumps({"split": 1, "seed": 19}))
        for role in ("val", "test"):
            path = d / f"{role}.npz"
            if name == "x2":
                np.savez_compressed(path, inst=inst, type=typ, inst_img=[0, 0, 1], inst_id=[8, 23, 8],
                                    inst_prob=np.eye(6)[[1, 4, 1]])
            else:
                np.savez_compressed(path, inst=inst, type=typ)
            inputs[f"{name}-{role}"] = path
    env = {**os.environ, "PANNUKE_ROOT": str(data), "CUDA_VISIBLE_DEVICES": ""}
    scripts = FILE.parents[3] / "scripts"
    p1 = subprocess.run([sys.executable, str(scripts / "analyze_scale_complementarity.py"),
                         "--base", str(inputs["base-val"]), "--x2", str(inputs["x2-val"]),
                         "--split", "1", "--role", "val", "--out", str(tmp_path / "p1.json"),
                         "--workers", "1"], capture_output=True, text=True, env=env, timeout=60)
    assert p1.returncode == 0, p1.stderr
    assert json.loads((tmp_path / "p1.json").read_text())["diagnostics"]["candidates"]["typed_dead_tp"] == 1
    out = tmp_path / "p2"
    cmd = [sys.executable, str(scripts / "run_scale_fusion.py"), "--split", "1", "--workers", "1", "--out", str(out)]
    for k, v in inputs.items():
        cmd += ["--" + k, str(v)]
    val = subprocess.run(cmd + ["--stage", "val"], capture_output=True, text=True, env=env, timeout=60)
    assert val.returncode == 0, val.stderr
    frozen = (out / "selection.json").read_bytes()
    assert json.loads(frozen)["status"] == "GO"
    original = inputs["x2-val"].read_bytes()
    inputs["x2-val"].write_bytes(original + b"changed")
    bad = subprocess.run(cmd + ["--stage", "test"], capture_output=True, text=True, env=env, timeout=60)
    assert bad.returncode != 0 and not (out / "test").exists()
    inputs["x2-val"].write_bytes(original)
    test = subprocess.run(cmd + ["--stage", "test"], capture_output=True, text=True, env=env, timeout=60)
    assert test.returncode == 0, test.stderr
    assert (out / "selection.json").read_bytes() == frozen
    score = json.loads((out / "test/eval/summary.json").read_text())
    assert score["official"]["mPQ"] == pytest.approx(1, abs=1e-6)
    # matched-seed round: arms trained with seed 1 need the declared pair seed
    for name in ("base", "x2"):
        (tmp_path / name / "config.json").write_text(json.dumps({"split": 1, "seed": 1}))
    ms_cmd = [sys.executable, str(scripts / "run_scale_fusion.py"), "--split", "1", "--workers", "1"]
    for k, v in inputs.items():
        ms_cmd += ["--" + k, str(v)]
    legacy = subprocess.run(ms_cmd + ["--out", str(tmp_path / "p2_default19"), "--stage", "val"],
                            capture_output=True, text=True, env=env, timeout=60)
    assert legacy.returncode != 0 and "seed19" in legacy.stderr
    ok = subprocess.run(ms_cmd + ["--out", str(tmp_path / "p2_ms"), "--expect-seed", "1", "--stage", "val"],
                        capture_output=True, text=True, env=env, timeout=60)
    assert ok.returncode == 0, ok.stderr
    ms = json.loads((tmp_path / "p2_ms" / "selection.json").read_text())
    assert ms["status"] == "GO" and ms["expected_seed"] == 1
