"""End-to-end HV configuration, immutable run identity and a train-only CLI."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from nucseg.cellvit import engine
from nucseg.cellvit.data import PanNukeCellViT


@pytest.fixture
def tiny_fold(tmp_path, monkeypatch):
    root = tmp_path / "data"
    fold = root / "fold1"
    fold.mkdir(parents=True)
    inst = np.zeros((1, 256, 256), np.int32)
    inst[0, 20:24, 20:24] = 7
    inst[0, 50:56, 50:56] = 42
    typ = np.where(inst == 7, 4, np.where(inst == 42, 1, 0)).astype(np.uint8)
    np.save(fold / "images.npy", np.full((1, 256, 256, 3), 128, np.uint8))
    np.save(fold / "inst.npy", inst)
    np.save(fold / "type.npy", typ)
    np.save(fold / "tissue.npy", np.array(["Colon"]))
    monkeypatch.setenv("PANNUKE_ROOT", str(root))
    return inst[0], typ[0]


@pytest.mark.parametrize("upscale,cutoff,want_small", [(1, 30, False), (1, 8, True),
                                                      (2, 30, True), (2, 120, False)])
def test_dataset_cutoff_changes_only_hv(tiny_fold, upscale, cutoff, want_small):
    inst, typ = tiny_fold
    ds = PanNukeCellViT([1], train=False, upscale=upscale, hv_min_size=cutoff)
    sample = ds[0]
    up_inst = np.repeat(np.repeat(inst, upscale, axis=0), upscale, axis=1)
    up_type = np.repeat(np.repeat(typ, upscale, axis=0), upscale, axis=1)
    np.testing.assert_array_equal(sample["np_map"], up_inst > 0)
    np.testing.assert_array_equal(sample["tp_map"], up_type)
    assert bool(np.any(sample["hv_map"].numpy()[up_inst == 7])) == want_small
    assert np.any(sample["hv_map"].numpy()[up_inst == 42])


def test_training_and_validation_receive_the_same_cutoff(tiny_fold, tmp_path, monkeypatch):
    real_dataset = engine.PanNukeCellViT
    datasets = []

    def capture(*args, **kwargs):
        ds = real_dataset(*args, **kwargs)
        ds.augs = lambda **kw: kw  # isolate HV wiring from random geometric augmentation
        datasets.append(ds)
        return ds

    monkeypatch.setattr(engine, "PanNukeCellViT", capture)
    cfg = engine.TrainConfig(split=1, out_dir=str(tmp_path / "run"), epochs=0,
                             workers=1, hv_min_size=8)
    engine.train(cfg, [1], [1], device="cpu", model=torch.nn.Linear(1, 1))
    assert len(datasets) == 2
    for ds in datasets:
        sample = ds[0]
        assert sample["hv_map"][20:24, 20:24].abs().sum() > 0
        assert sample["np_map"].sum() == 52
    saved = json.loads((Path(cfg.out_dir) / "config.json").read_text())
    assert saved["hv_min_size"] == 8


def test_resolver_uses_existing_cutoff_and_legacy_default(tmp_path):
    assert engine.run_hv_min_size(tmp_path) == 30
    (tmp_path / "config.json").write_text(json.dumps({"split": 1, "hv_min_size": 8}))
    assert engine.run_hv_min_size(tmp_path) == 8
    assert engine.run_hv_min_size(tmp_path, 120) == 120
    (tmp_path / "config.json").write_text(json.dumps({"split": 1}))
    assert engine.run_hv_min_size(tmp_path) == 30


@pytest.mark.parametrize("key,value", [("hv_min_size", 8), ("upscale", 2),
                                       ("seed", 1), ("lr", 1e-4)])
def test_resume_refuses_changed_scientific_config_before_overwrite(tmp_path, key, value):
    cfg = engine.TrainConfig(split=1, out_dir=str(tmp_path))
    original = json.dumps(asdict(cfg))
    path = tmp_path / "config.json"
    path.write_text(original)
    changed = engine.TrainConfig(**{**asdict(cfg), key: value})
    with pytest.raises(ValueError, match=key):
        engine.prepare_train_config(changed)
    assert path.read_text() == original


def test_legacy_resume_defaults_to30_and_allows_runtime_paths(tmp_path):
    cfg = engine.TrainConfig(split=1, out_dir=str(tmp_path))
    old = asdict(cfg)
    old.pop("hv_min_size")
    old["out_dir"] = "old/machine/path"
    (tmp_path / "config.json").write_text(json.dumps(old))
    engine.prepare_train_config(cfg)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["hv_min_size"] == 30


def test_orphan_checkpoint_cannot_get_a_new_config(tmp_path):
    (tmp_path / "last.pth").write_bytes(b"unread checkpoint")
    cfg = engine.TrainConfig(split=1, out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="config"):
        engine.prepare_train_config(cfg)
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("bad", [0, -1, 7.5, True])
def test_train_config_rejects_invalid_hv_cutoff(bad, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        engine.TrainConfig(split=1, out_dir=str(tmp_path / "absent"), hv_min_size=bad)
    assert not (tmp_path / "absent").exists()


def _cli():
    path = Path(__file__).resolve().parents[1] / "scripts/train_cellvit.py"
    spec = importlib.util.spec_from_file_location("train_cellvit_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_only_stops_before_test_inference(tmp_path, monkeypatch):
    cli = _cli()
    seen = []

    def trainer(cfg, tr, va):
        seen.append((asdict(cfg), tr, va))

    def forbidden(*args, **kwargs):
        pytest.fail("train-only entered test inference")

    monkeypatch.setattr(cli, "train", trainer)
    monkeypatch.setattr(cli, "build_model", forbidden)
    monkeypatch.setattr(cli, "PanNukeFold", forbidden)
    monkeypatch.setattr(cli, "predict_fold", forbidden)
    cli.main(["--split", "1", "--out", str(tmp_path / "run"), "--seed", "1",
              "--upscale", "2", "--hv-min-size", "120", "--train-only"])
    assert len(seen) == 1
    cfg, tr, va = seen[0]
    assert (cfg["hv_min_size"], cfg["upscale"], cfg["seed"]) == (120, 2, 1)
    assert tr == [1] and va == [2]


def test_cli_rejects_bad_cutoff_before_writes(tmp_path):
    cli = _cli()
    with pytest.raises(SystemExit) as exc:
        cli.main(["--split", "1", "--out", str(tmp_path / "no"),
                  "--hv-min-size", "0", "--train-only"])
    assert exc.value.code == 2 and not (tmp_path / "no").exists()


def test_cli_rejects_skip_train_with_train_only(tmp_path):
    cli = _cli()
    with pytest.raises(SystemExit) as exc:
        cli.main(["--split", "1", "--out", str(tmp_path), "--skip-train", "--train-only"])
    assert exc.value.code == 2
