"""Frozen HV-control summaries reject incomplete/mismatched runs and wrong GT alignment."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/hv_control_numbers.py"


def _module():
    spec = importlib.util.spec_from_file_location("hv_control_numbers", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "runs"
    data = tmp_path / "data"
    fold = data / "fold2"
    fold.mkdir(parents=True)
    inst = np.zeros((3, 32, 32), np.uint16)
    inst[0, :4, :4] = 5
    inst[1, 10:16, 10:16] = 11
    inst[2, :4, 20:24] = 19
    typ = np.zeros_like(inst, dtype=np.uint8)
    typ[0, inst[0] > 0] = 1
    typ[1:, ...] = np.where(inst[1:] > 0, 4, 0)
    tissue = np.array(["Colon", "Colon", "Lung"])
    for name, array in (("inst", inst), ("type", typ), ("tissue", tissue)):
        np.save(fold / f"{name}.npy", array)
    arms = {"x1_hv30": (1, 30, [0, 0, 0]), "x1_hv8": (1, 8, [.02, 0, .04]),
            "x2_hv30": (2, 30, [.03, .02, .05]), "x2_hv120": (2, 120, [.01, .01, .02])}
    for arm, (scale, cutoff, offsets) in arms.items():
        for seed in (19, 1):
            run = root / f"{arm}_seed{seed}"
            ev = run / "audit_val_fold2/eval"
            ev.mkdir(parents=True)
            cfg = dict(split=1, seed=seed, upscale=scale, hv_min_size=cutoff,
                       epochs=130, unfreeze_epoch=25, batch_size=16, lr=.0003,
                       weight_decay=.0001, gamma=.85, sampling_gamma=.85, workers=8,
                       val_every=5, np_wce=0., dead_w=0., small_w=0., cp_prob=0.,
                       synth=None, synth_frac=0.)
            (run / "config.json").write_text(json.dumps(cfg))
            prov = dict(source_commit="c6e7e92aceea9a9bc28f85d77611f302af9b7387", arm=arm,
                        seed=seed, split=1, train_fold=1, val_fold=2, test_access=False,
                        upscale=scale, hv_min_size=cutoff, batch_size=16, epochs=130)
            (run / "provenance.json").write_text(json.dumps(prov))
            (run / "log.jsonl").write_text("\n".join(json.dumps({"epoch": ep, "train/loss": 1., "time": 1.})
                                                     for ep in range(130)))
            (run / "final.pth").write_bytes(b"unit-test fixture, not a model")
            (run / "last.pth").write_bytes(b"unit-test fixture, not a model")
            offset = np.array(offsets) + (0 if seed == 19 else .005)
            mpq = np.array([.2, .4, .9]) + offset
            bpq = np.array([.4, .6, .9]) + offset
            cls = np.full((3, 5), np.nan)
            cls[0, 0], cls[1, 3], cls[2, 3] = mpq
            np.savez_compressed(ev / "per_image.npz", mPQ=mpq, bPQ=bpq, class_PQ=cls)
            m = (mpq[:2].mean() + mpq[2]) / 2
            b = (bpq[:2].mean() + bpq[2]) / 2
            dead = mpq[1:].mean()
            summary = {"n_images": 3, "official": {"mPQ": m, "bPQ": b},
                       "strict": {"mPQ": m-.1}, "plus": {"mPQ+": m+.01},
                       "per_class_PQ": {"Dead": dead}, "per_class_PQ_strict": {"Dead": dead-.1}}
            (ev / "summary.json").write_text(json.dumps(summary))
            pd.DataFrame({"image": [0, 1, 2], "cls": [1, 4, 4], "area": [16, 36, 16],
                          "tissue": tissue, "status": ["matched"]*3, "pred_cls": [1, 4, 4]}) \
                .to_csv(ev / "gt_records.csv.gz", index=False)
            shutil.copytree(ev, run / "eval_val_fold2")
            pred = ev.parent / ("pred_fold2.npz" if scale == 1 else "pred_fold2_x2_du2.npz")
            pred.write_bytes(b"unit-test prediction fixture")
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            code_paths = ("scripts/predict_cellvit.py", "scripts/eval_pannuke.py",
                          "src/nucseg/cellvit/model.py", "src/nucseg/cellvit/engine.py",
                          "src/nucseg/hovernet/engine.py", "src/nucseg/metrics/pannuke_eval.py",
                          "src/nucseg/metrics/instance.py", "third_party/hover_net/models/hovernet/post_proc.py")
            manifest = dict(fold=2, tta=False, upscale=scale, decode_u=scale, marker_u=1,
                            batch_size=8, checkpoint="final.pth", complete=True,
                            checkpoint_sha256=sha(run / "final.pth"), prediction_file=pred.name,
                            prediction_sha256=sha(pred),
                            evaluation_sha256={name: sha(ev / name) for name in
                                               ("summary.json", "per_image.npz", "gt_records.csv.gz")},
                            code_sha256={name: sha(PATH.parents[1] / name) for name in code_paths})
            (ev.parent / "inference_provenance.json").write_text(json.dumps(manifest))
    return root, data


def test_frozen_contrasts_and_tissue_aggregation(study):
    module = _module()
    result = module.analyze(*study, boot=32)
    assert len(result["runs"]) == 8
    assert result["protocol"]["validation_fold"] == 2
    base = result["arms"]["x1_hv30"]["mean"]
    assert base["mPQ"] == pytest.approx(.6025)
    assert base["Dead PQ"] == pytest.approx(.6525)
    contrasts = result["contrasts"]
    assert contrasts["lower_support_x1"]["mean"]["mPQ"] == pytest.approx(.025)
    assert contrasts["lower_support_x2"]["mean"]["mPQ"] == pytest.approx(.0225)
    assert contrasts["scale_at_high_support"]["mean"]["mPQ"] == pytest.approx(.015)
    assert contrasts["scale_at_low_support"]["mean"]["mPQ"] == pytest.approx(.0125)
    assert contrasts["lower_support_x1"]["mean"]["Dead PQ"] == pytest.approx(.02)
    view = result["runs"]["x1_hv30_seed19"]["gt_views"]
    assert view["interior_dead"]["n"] == 1
    assert view["border_dead"]["n"] == 1
    assert view["native_area_lt30"]["n"] == 2
    assert view["interior_dead"]["matched_rate"] == 1
    json.dumps(result, allow_nan=False)


def test_incomplete_run_is_rejected(study):
    root, data = study
    (root / "x2_hv120_seed1/final.pth").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        _module().analyze(root, data, boot=8)


@pytest.mark.parametrize("field,value", [("hv_min_size", 30), ("seed", 19), ("batch_size", 8)])
def test_wrong_training_configuration_is_rejected(study, field, value):
    root, data = study
    p = root / "x1_hv8_seed1/config.json"
    cfg = json.loads(p.read_text()); cfg[field] = value
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=field):
        _module().analyze(root, data, boot=8)


def test_test_fold_artifacts_are_rejected(study):
    root, data = study
    (root / "x1_hv30_seed19/pred_fold3.npz").write_bytes(b"forbidden fixture")
    with pytest.raises(ValueError, match="fold3"):
        _module().analyze(root, data, boot=8)


def test_dead_summary_must_match_raw_per_image_values(study):
    root, data = study
    p = root / "x1_hv30_seed19/audit_val_fold2/eval/summary.json"
    s = json.loads(p.read_text()); s["per_class_PQ"]["Dead"] += .1
    p.write_text(json.dumps(s))
    manifest_path = p.parent.parent / "inference_provenance.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_sha256"]["summary.json"] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Dead"):
        _module().analyze(root, data, boot=8)


def test_gt_order_mismatch_is_not_silently_truncated(study):
    root, data = study
    p = root / "x1_hv30_seed19/audit_val_fold2/eval/gt_records.csv.gz"
    g = pd.read_csv(p).iloc[::-1]
    g.to_csv(p, index=False)
    manifest_path = p.parent.parent / "inference_provenance.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_sha256"]["gt_records.csv.gz"] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="GT alignment"):
        _module().analyze(root, data, boot=8)


def test_incomplete_training_log_is_rejected(study):
    root, data = study
    p = root / "x1_hv30_seed19/log.jsonl"
    p.write_text(json.dumps({"epoch": 0, "train/loss": 1., "time": 1.}))
    with pytest.raises(ValueError, match="epoch"):
        _module().analyze(root, data, boot=8)


def test_wrong_source_commit_is_rejected(study):
    root, data = study
    p = root / "x2_hv30_seed1/provenance.json"
    v = json.loads(p.read_text()); v["source_commit"] = "wrong"
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError, match="source_commit"):
        _module().analyze(root, data, boot=8)


@pytest.mark.parametrize("field,value", [("decode_u", 1), ("tta", True), ("checkpoint", "last.pth")])
def test_inference_settings_must_match_frozen_protocol(study, field, value):
    root, data = study
    p = root / "x2_hv30_seed1/audit_val_fold2/inference_provenance.json"
    manifest = json.loads(p.read_text()); manifest[field] = value
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        _module().analyze(root, data, boot=8)


def test_prediction_and_checkpoint_hashes_are_verified(study):
    root, data = study
    p = root / "x1_hv8_seed1/audit_val_fold2/pred_fold2.npz"
    p.write_bytes(b"different prediction")
    with pytest.raises(ValueError, match="prediction.*hash"):
        _module().analyze(root, data, boot=8)


def test_changed_final_checkpoint_is_rejected(study):
    root, data = study
    (root / "x1_hv8_seed1/final.pth").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="checkpoint.*hash"):
        _module().analyze(root, data, boot=8)
