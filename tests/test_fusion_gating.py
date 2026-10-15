"""Skip-path fusion arms (spec docs/superpowers/specs/2026-10-15-fusion-gating-design.md).

Covered here: the baseline function is unchanged for "plain"; both new arms are exactly the
baseline function at initialization (zero-init gate = identity, zero-init adapter = +0); the
gate and its control have identical parameter counts; the fusion flag is frozen into the run
config and cannot be combined with the visibility arms or with test-fold inference.
"""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nucseg.cellvit import engine
from nucseg.cellvit.engine import TrainConfig
from nucseg.cellvit.model import (CellViTUNI, SkipGate, branch_param_count, fusion_param_count,
                                  total_param_count)

# analytic total of the four skip gates: sum(C^2 + C) over C in 64/128/256/512
GATE_PARAMS = 349120


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(19)
    plain = CellViTUNI(uni_ckpt=None, fusion="plain")
    torch.manual_seed(19)
    gate = CellViTUNI(uni_ckpt=None, fusion="gate")
    gate.load_state_dict({k: v for k, v in plain.state_dict().items()}, strict=False)
    return plain, gate


# ------------------------------------------------------------------ SkipGate unit behaviour

def test_zero_init_gate_and_adapter_are_exact_identity():
    """C * softmax(0) == 1 and W(0) == 0, so both arms start at the baseline function."""
    for mode in ("gate", "adapter"):
        module = SkipGate(64, mode)  # powers of two keep 1/C exact
        s = torch.randn(2, 64, 5, 7)
        assert torch.equal(module(s), s), f"{mode} is not an exact identity at initialization"


def test_gate_reweights_channels_and_adapter_only_adds():
    torch.manual_seed(3)
    gate, adapter = SkipGate(64, "gate"), SkipGate(64, "adapter")
    with torch.no_grad():
        gate.proj.weight.normal_(0, 0.5)
        gate.proj.bias.normal_(0, 0.5)
        adapter.proj.weight.normal_(0, 0.5)
        adapter.proj.bias.normal_(0, 0.5)
    s = torch.randn(2, 64, 4, 4)
    with torch.no_grad():
        gated = gate(s)
        added = adapter(s)
        # the gate is multiplicative with per-pixel, per-channel factors whose mean over
        # channels is pinned to 1 (that is what the C * softmax scaling buys)
        factors = gate.channels * torch.softmax(gate.proj(s), dim=1)
        assert torch.allclose(factors.mean(dim=1), torch.ones_like(factors.mean(dim=1)), atol=1e-5)
        assert torch.allclose(gated, s * factors)
        assert not torch.allclose(gated, s)
        assert torch.allclose(added, s + adapter.proj(s))
    # gate0 differs from gate1 by channel count only
    assert SkipGate(128, "gate").channels == 128
    for bad in ("plain", "bogus"):
        with pytest.raises(ValueError):
            SkipGate(64, bad)


# ------------------------------------------------------------------ wiring and parameter budget

def test_plain_is_unchanged_and_gate_is_identity_at_init(models):
    plain, gate = models
    assert not any(k.startswith("gates.") for k in plain.state_dict())
    assert len([k for k in gate.state_dict() if k.startswith("gates.")]) == 8  # 4 gates x (w, b)
    assert plain.fusion == "plain" and gate.fusion == "gate"
    assert fusion_param_count(plain) == 0
    # identical shared weights -> identical outputs, because the gate is the identity at init
    x = torch.randn(1, 3, 64, 64)
    plain.eval(), gate.eval()
    with torch.no_grad():
        for key in ("np", "hv", "tp"):
            assert torch.allclose(plain(x)[key], gate(x)[key], atol=1e-6)
    assert not any("gates" in name for name in branch_param_count(plain))


def test_gate_and_adapter_parameter_counts_match_and_are_small(models):
    _, gate = models
    assert fusion_param_count(gate) == GATE_PARAMS
    for channels in (64, 128, 256, 512):
        assert (sum(p.numel() for p in SkipGate(channels, "gate").parameters())
                == sum(p.numel() for p in SkipGate(channels, "adapter").parameters()))
    assert fusion_param_count(gate) / total_param_count(gate) < 0.001
    assert branch_param_count(gate)["gates"] == GATE_PARAMS


@pytest.mark.parametrize("mode", ["plain", "gate", "adapter"])
def test_forward_shapes_and_finite_gradients(mode):
    torch.manual_seed(5)
    model = CellViTUNI(uni_ckpt=None, fusion=mode)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out["np"].shape == (2, 2, 32, 32) and out["tp"].shape == (2, 6, 32, 32)
    assert all(bool(torch.isfinite(v).all()) for v in (out["np"], out["hv"], out["tp"]))
    (out["np"].square().mean() + out["tp"].square().mean()).backward()
    grads = [p.grad for n, p in model.named_parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    if mode != "plain":
        gate_grads = [p.grad for p in model.gates.parameters()]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in gate_grads)


# ------------------------------------------------------------------ frozen guards

def test_unknown_fusion_mode_is_rejected():
    with pytest.raises(ValueError, match="fusion"):
        TrainConfig(split=1, out_dir="/tmp/x", fusion="bogus")


def test_visibility_and_fusion_are_mutually_exclusive():
    with pytest.raises(ValueError, match="refuses additional scientific interventions"):
        TrainConfig(split=1, out_dir="/tmp/x", visibility_arm="A0", fusion="gate")


def test_resume_refuses_changed_fusion(tmp_path):
    cfg = TrainConfig(split=1, out_dir=str(tmp_path), fusion="gate")
    original = json.dumps(asdict(cfg))
    path = tmp_path / "config.json"
    path.write_text(original)
    for changed_mode in ("plain", "adapter"):
        changed = TrainConfig(**{**asdict(cfg), "fusion": changed_mode})
        with pytest.raises(ValueError, match="fusion"):
            engine.prepare_train_config(changed)
    assert path.read_text() == original


def test_resume_accepts_a_legacy_config_without_the_fusion_key(tmp_path):
    """Pre-2026-10-15 run dirs have no 'fusion' key and must keep resuming as plain."""
    cfg = TrainConfig(split=1, out_dir=str(tmp_path))
    legacy = {k: v for k, v in asdict(cfg).items() if k != "fusion"}
    (tmp_path / "config.json").write_text(json.dumps(legacy))
    engine.prepare_train_config(cfg)  # no mismatch
    assert json.loads((tmp_path / "config.json").read_text())["fusion"] == "plain"


def test_run_fusion_reads_the_config(tmp_path):
    path = tmp_path / "config.json"
    assert engine.run_fusion(tmp_path) == "plain"          # no config
    path.write_text(json.dumps({"fusion": "gate"}))
    assert engine.run_fusion(tmp_path) == "gate"
    assert engine.run_fusion(tmp_path, "adapter") == "adapter"
    assert engine.run_fusion(tmp_path) == "gate"
    path.write_text(json.dumps({"fusion": "bogus"}))
    with pytest.raises(ValueError, match="fusion"):
        engine.run_fusion(tmp_path)
    with pytest.raises(ValueError, match="fusion"):
        engine.run_fusion(tmp_path, "bogus")


def test_dev_folds_and_cli_guards(tmp_path):
    """The dev arm trains fold1 and validates fold2 only, and never reaches test inference."""
    cfg = TrainConfig(split=1, out_dir=str(tmp_path), fusion="gate")
    with pytest.raises(ValueError, match="fold1 training / fold2 validation"):
        engine.train(cfg, [2], [1])
    cli = _load("train_cellvit")
    for argv in (["--split", "1", "--out", str(tmp_path), "--fusion", "gate"],              # no --train-only
                 ["--split", "2", "--out", str(tmp_path), "--fusion", "gate", "--train-only"],
                 ["--split", "1", "--out", str(tmp_path), "--fusion", "gate", "--train-only",
                  "--visibility-arm", "A0"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    args = cli.build_parser().parse_args(["--split", "1", "--out", str(tmp_path),
                                          "--fusion", "gate", "--train-only"])
    assert args.fusion == "gate"
