"""CLI wiring, resume freeze and dead-aware prediction artifacts (DSB plan Task 5)."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from nucseg.cellvit import engine
from nucseg.cellvit.engine import TrainConfig


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_fold(tmp_path, monkeypatch):
    root = tmp_path / "data"
    fold = root / "fold1"
    fold.mkdir(parents=True)
    inst = np.zeros((2, 256, 256), np.int32)
    inst[:, 20:24, 20:24] = 7
    typ = np.where(inst == 7, 4, 0).astype(np.uint8)
    np.save(fold / "images.npy", np.full((2, 256, 256, 3), 128, np.uint8))
    np.save(fold / "inst.npy", inst)
    np.save(fold / "type.npy", typ)
    np.save(fold / "tissue.npy", np.array(["Colon", "Colon"]))
    monkeypatch.setenv("PANNUKE_ROOT", str(root))
    return inst, typ


@pytest.mark.parametrize("key,value", [("dead_expert", True), ("dead_neg_w", 0.1), ("widen", 64)])
def test_resume_refuses_changed_dead_config(tmp_path, key, value):
    cfg = TrainConfig(split=1, out_dir=str(tmp_path))
    original = json.dumps(asdict(cfg))
    path = tmp_path / "config.json"
    path.write_text(original)
    changed = TrainConfig(**{**asdict(cfg), key: value})
    with pytest.raises(ValueError, match=key):
        engine.prepare_train_config(changed)
    assert path.read_text() == original


def test_cli_wires_dead_fields_to_config(tmp_path, monkeypatch):
    cli = _load("train_cellvit")
    seen = []
    monkeypatch.setattr(cli, "train", lambda cfg, tr, va: seen.append(asdict(cfg)))
    monkeypatch.setattr(cli, "build_model",
                        lambda *a, **k: pytest.fail("train-only must not build an inference model"))
    cli.main(["--split", "1", "--out", str(tmp_path / "r"), "--dead-expert",
              "--dead-neg-w", "0.1", "--widen", "64", "--train-only"])
    cfg = seen[0]
    assert (cfg["dead_expert"], cfg["dead_neg_w"], cfg["widen"]) == (True, 0.1, 64)


def test_run_resolvers_default_and_config(tmp_path):
    assert engine.run_dead_expert(tmp_path) is False and engine.run_widen(tmp_path) == 0
    (tmp_path / "config.json").write_text(json.dumps({"split": 1, "dead_expert": True, "widen": 64}))
    assert engine.run_dead_expert(tmp_path) is True and engine.run_widen(tmp_path) == 64
    assert engine.run_dead_expert(tmp_path, False) is False


class _StubDeadModel(torch.nn.Module):
    """No base foreground anywhere; one confident 12x12 dead-fg square per image on a star HV field."""

    def forward(self, x):
        b = x.shape[0]
        # instance-local HV like a trained expert would emit: zero outside the square, linear
        # toward its center (a whole-image ramp has constant Sobel response and decodes to nothing)
        ys = torch.linspace(-1, 1, 256).view(1, 256, 1).expand(b, 256, 256)
        xs = torch.linspace(-1, 1, 256).view(1, 1, 256).expand(b, 256, 256)
        inside = torch.zeros(b, 256, 256, dtype=torch.bool)
        inside[:, 100:112, 100:112] = True
        hv = torch.stack([torch.where(inside, xs, torch.zeros(())),
                          torch.where(inside, ys, torch.zeros(()))], 1)
        # softmax needs ASYMMETRIC logits: bg 0 / fg -20 outside (P(fg)~2e-9), fg +8 inside
        def np_map():
            m = torch.zeros(b, 2, 256, 256)
            m[:, 1] = -20.0
            m[:, 1, 100:112, 100:112] = 8.0
            return m
        npd = np_map()
        base_np = torch.zeros(b, 2, 256, 256)
        base_np[:, 1] = -20.0  # no base foreground anywhere
        return {"np": base_np, "hv": hv,
                "tp": torch.zeros(b, 6, 256, 256), "tissue": torch.zeros(b, 19),
                "np_dead": npd, "hv_dead": hv}


def test_predict_fold_dead_tail(tiny_fold):
    from nucseg.data.pannuke import PanNukeFold
    f = PanNukeFold(1)
    res = engine.predict_fold(_StubDeadModel(), f, batch_size=2, workers=1, device="cpu",
                              dead_expert=True)
    inst, typ, tissue, dead = res
    assert dead.dtype == np.int32 and dead.shape == (2, 256, 256)
    for i in range(2):  # exactly one candidate per image, inside the confident square
        ids = np.unique(dead[i])[1:]
        assert len(ids) == 1
        ys, xs = np.nonzero(dead[i])
        assert 99 <= ys.min() and ys.max() <= 113
    assert (inst > 0).sum() == 0  # base decode untouched: no foreground anywhere


def test_dead_predictions_write_separate_artifact(tmp_path):
    cli = _load("predict_cellvit")
    (tmp_path / "pred_fold2.npz").write_bytes(b"base-bytes")
    out = cli.write_dead_predictions(tmp_path, "fold2", np.zeros((1, 8, 8), np.int32))
    assert out.name == "pred_fold2_dead.npz"
    assert (tmp_path / "pred_fold2.npz").read_bytes() == b"base-bytes"


def test_predict_cli_dead_flag(tmp_path, monkeypatch, tiny_fold):
    cli = _load("predict_cellvit")
    run = tmp_path / "run"
    run.mkdir()
    calls = {}

    def fake_predict_fold(model, f, **kw):
        calls.update(kw)
        dead = np.zeros((2, 256, 256), np.int32)
        dead[0, 10:14, 10:14] = 1
        return np.zeros((2, 256, 256), np.int32), np.zeros((2, 256, 256), np.uint8), \
            np.zeros((2, 19)), dead

    class _M:
        def cuda(self):
            return self

        def load_state_dict(self, *a, **k):
            pass

    monkeypatch.setattr(cli, "build_model", lambda *a, **k: _M())
    monkeypatch.setattr(cli, "torch", SimpleTorch())
    monkeypatch.setattr(cli, "predict_fold", fake_predict_fold)
    cli.main(["--run", str(run), "--fold", "1", "--dead", "--no-eval", "--no-inst-probs"])
    assert calls["dead_expert"] is True
    payload = np.load(run / "pred_fold1_dead.npz")
    assert payload["dead_inst"].shape == (2, 256, 256)
    assert (run / "pred_fold1.npz").exists()  # base artifacts still written


class SimpleTorch:
    """Stand-in for the script's torch.load against a stub checkpoint."""

    @staticmethod
    def load(*a, **k):
        return {"model": {}}
