"""Tests for the R1 probe variant: the pre-declared stopping rule, the R1 constants, and
byte-identity of all frozen computational functions with the P1 probe."""
import importlib.util
import re
from pathlib import Path


def _load(name):
    candidates = [Path(__file__).with_name(name),
                  Path(__file__).resolve().parents[1] / "scripts" / name]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(name)


def api():
    return _load("probe_frozen_point_heatmap_r1.py")


def test_stop_rule_silent_before_min_epoch_and_at_non_multiples():
    m = api()
    for n in (1, 5, 10, 14):
        assert m.r1_stop_decision([1.0] * n, n) == (False, "")
    assert m.r1_stop_decision([0.01] * 16, 16) == (False, "")  # 16 % 5 != 0


def test_stop_rule_loss_target_fires():
    m = api()
    losses = [1.0] * 10 + [0.01] * 5
    stop, reason = m.r1_stop_decision(losses, 15)
    assert stop and reason.startswith("loss_target:")


def test_stop_rule_plateau_fires_when_improvement_small():
    m = api()
    losses = ([2.00, 1.90, 1.80, 1.70, 1.60]
              + [1.55, 1.54, 1.53, 1.52, 1.51]
              + [1.50, 1.499, 1.498, 1.497, 1.496])
    # prev5 mean 1.530, last5 mean 1.498: rel drop 2.1% < 5%, last5 > 0.02
    stop, reason = m.r1_stop_decision(losses, 15)
    assert stop and reason.startswith("plateau:")


def test_stop_rule_continues_when_improving():
    m = api()
    losses = ([4.0, 3.5, 3.0, 2.5, 2.0]
              + [1.8, 1.6, 1.4, 1.2, 1.0]
              + [0.9, 0.8, 0.7, 0.6, 0.5])
    # prev5 mean 1.4 -> last5 mean 0.7: 50% drop -> continue
    assert m.r1_stop_decision(losses, 15) == (False, "")


def test_stop_rule_history_length_guard():
    m = api()
    try:
        m.r1_stop_decision([1.0] * 14, 15)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_r1_constants_and_primary_only_structure():
    m = api()
    assert m.EPOCH_CAP == 100
    assert m.R1_LOSS_TARGET_5EP == 0.02
    assert m.R1_PLATEAU_MIN_REL_DROP == 0.05
    assert m.R1_CHECK_MIN_EPOCH == 15 and m.R1_CHECK_EVERY == 5
    assert m.DEADLINE_S == 12000.0
    src = Path(m.__file__).read_text()
    assert '("uni", None, "uni")' not in src  # secondary arm omitted in R1
    assert "r1_stop_decision" in src


def _grab(text, name):
    m2 = re.search(r"^def %s\(.*?(?=^def |^class |\Z)" % name, text, re.M | re.S)
    assert m2 is not None, name
    return m2.group(0)


def test_frozen_functions_byte_identical_to_p1():
    r1 = _load("probe_frozen_point_heatmap_r1.py")
    p1 = _load("probe_frozen_point_heatmap.py")
    r1_src = Path(r1.__file__).read_text()
    p1_src = Path(p1.__file__).read_text()
    for name in ("gt_objects", "gaussian_target", "focal_center_loss", "decode_peaks",
                 "decode_nms", "hits_for_points", "radius_curve"):
        assert _grab(p1_src, name) == _grab(r1_src, name), name
