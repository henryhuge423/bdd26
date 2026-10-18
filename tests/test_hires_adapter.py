"""E19 high-resolution adapter (spec docs/superpowers/specs/2026-10-18-hires-adapter-design.md).

Covered here: the baseline (hires_adapter=False) is unchanged; the H model is the baseline function
at initialization (the adapter's final 1x1 conv is zero-initialised); shared modules keep identical
names, shapes and initial values; the adapter is live once its output conv is nonzero; the flag is
frozen into the run config, cannot be combined with other scientific interventions, and the dev arm
only trains fold1 and validates fold2.
"""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nucseg.cellvit import engine
from nucseg.cellvit.engine import TrainConfig, build_model
from nucseg.cellvit.model import CellViTUNI, HiResAdapter, HiResCellViTUNI


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ adapter unit behaviour

def test_adapter_output_is_zero_at_init_and_live_when_nonzero():
    torch.manual_seed(0)
    adapter = HiResAdapter()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        head = adapter(x)
    assert head.shape == (2, 64, 32, 32)
    assert float(head.abs().max()) < 1e-12, "zero-init output must be zero"
    with torch.no_grad():
        adapter.out.weight.normal_(0, 0.1)
    with torch.no_grad():
        assert not torch.allclose(adapter(x), torch.zeros_like(head))


def test_hires_is_the_baseline_function_at_init():
    torch.manual_seed(19)
    base = CellViTUNI(uni_ckpt=None)
    torch.manual_seed(19)
    hires = HiResCellViTUNI(uni_ckpt=None)
    # shared modules are identical by construction; load explicitly so the claim does not depend
    # on RNG-consumption order. atol matches the project's other identity-at-init checks.
    hires.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(1, 3, 64, 64)
    base.eval(), hires.eval()
    with torch.no_grad():
        a, b = base(x), hires(x)
    for key in ("np", "hv", "tp", "tissue"):
        assert torch.allclose(a[key], b[key], atol=1e-6), f"{key} differs from the baseline at initialization"


def test_shared_modules_have_identical_names_shapes_and_values():
    torch.manual_seed(19)
    base = CellViTUNI(uni_ckpt=None).state_dict()
    torch.manual_seed(19)
    hires = HiResCellViTUNI(uni_ckpt=None).state_dict()
    only_hires = {k for k in hires if k.startswith("hires.")}
    assert only_hires and all(k not in base for k in only_hires)
    assert {k for k in hires if k not in only_hires} == set(base)
    for k, v in base.items():
        assert torch.equal(v, hires[k]), f"shared parameter {k} changed"


def test_adapter_changes_the_output_once_nonzero():
    torch.manual_seed(19)
    base = CellViTUNI(uni_ckpt=None)
    torch.manual_seed(19)
    hires = HiResCellViTUNI(uni_ckpt=None)
    with torch.no_grad():
        hires.hires.out.weight.normal_(0, 0.1)
    x = torch.randn(1, 3, 64, 64)
    base.eval(), hires.eval()
    with torch.no_grad():
        assert not torch.allclose(base(x)["np"], hires(x)["np"])


def test_freeze_encoder_leaves_the_adapter_trainable():
    model = HiResCellViTUNI(uni_ckpt=None)
    model.freeze_encoder(True)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.hires.parameters())


@pytest.mark.parametrize("hires", [False, True])
def test_forward_shapes_and_finite_gradients(hires):
    torch.manual_seed(5)
    model = (HiResCellViTUNI if hires else CellViTUNI)(uni_ckpt=None)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out["np"].shape == (2, 2, 32, 32) and out["tp"].shape == (2, 6, 32, 32)
    assert all(bool(torch.isfinite(v).all()) for v in (out["np"], out["hv"], out["tp"]))
    (out["np"].square().mean() + out["tp"].square().mean()).backward()
    if hires:
        grads = [p.grad for p in model.hires.parameters()]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


# ------------------------------------------------------------------ frozen guards

def test_build_model_selects_the_hires_class():
    assert isinstance(build_model(TrainConfig(split=1, out_dir="/tmp/x"), pretrained=False), CellViTUNI)
    assert isinstance(build_model(TrainConfig(split=1, out_dir="/tmp/x", hires_adapter=True),
                                 pretrained=False), HiResCellViTUNI)


@pytest.mark.parametrize("other", [
    {"fusion": "gate"}, {"dead_expert": True}, {"widen": 32}, {"upscale": 2},
    {"cp_prob": 0.3}, {"np_wce": 1.0}, {"dead_w": 1.0}, {"small_w": 1.0},
    {"visibility_arm": "A0"},
])
def test_hires_refuses_other_scientific_interventions(other):
    with pytest.raises(ValueError, match="hires"):
        TrainConfig(split=1, out_dir="/tmp/x", hires_adapter=True, **other)


def test_hires_requires_split1(tmp_path):
    cfg = TrainConfig(split=1, out_dir=str(tmp_path), hires_adapter=True)
    with pytest.raises(ValueError, match="fold1 training / fold2 validation"):
        engine.train(cfg, [2], [1])


def test_resume_refuses_changed_hires_flag(tmp_path):
    cfg = TrainConfig(split=1, out_dir=str(tmp_path), hires_adapter=True)
    original = json.dumps(asdict(cfg))
    path = tmp_path / "config.json"
    path.write_text(original)
    changed = TrainConfig(**{**asdict(cfg), "hires_adapter": False})
    with pytest.raises(ValueError, match="hires_adapter"):
        engine.prepare_train_config(changed)
    assert path.read_text() == original


def test_legacy_config_without_hires_key_resumes_as_plain(tmp_path):
    cfg = TrainConfig(split=1, out_dir=str(tmp_path))
    legacy = {k: v for k, v in asdict(cfg).items() if k != "hires_adapter"}
    (tmp_path / "config.json").write_text(json.dumps(legacy))
    engine.prepare_train_config(cfg)  # no mismatch
    assert json.loads((tmp_path / "config.json").read_text())["hires_adapter"] is False


def test_run_hires_adapter_reads_the_config(tmp_path):
    path = tmp_path / "config.json"
    assert engine.run_hires_adapter(tmp_path) is False
    path.write_text(json.dumps({"hires_adapter": True}))
    assert engine.run_hires_adapter(tmp_path) is True
    assert engine.run_hires_adapter(tmp_path, False) is False


def test_cli_hires_requires_split1_and_train_only(tmp_path):
    cli = _load("train_cellvit")
    for argv in (["--split", "1", "--out", str(tmp_path), "--hires-adapter"],            # no --train-only
                 ["--split", "2", "--out", str(tmp_path), "--hires-adapter", "--train-only"],
                 ["--split", "1", "--out", str(tmp_path), "--hires-adapter", "--train-only",
                  "--fusion", "gate"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    args = cli.build_parser().parse_args(["--split", "1", "--out", str(tmp_path),
                                          "--hires-adapter", "--train-only"])
    assert args.hires_adapter is True
