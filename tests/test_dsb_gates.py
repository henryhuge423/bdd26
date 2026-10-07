"""DSB gate machinery: pure-logic tests for the oracle-ceiling probe (plan Task 8) and later gates."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ gate D

def test_unmatched_internal_dead_finder():
    m = _load("dsb_oracle_ceiling")
    gt_inst = np.zeros((16, 16), np.int32)
    gt_typ = np.zeros((16, 16), np.uint8)
    gt_inst[5:8, 5:8] = 1; gt_typ[gt_inst == 1] = 4    # internal, missed    -> ADD
    gt_inst[0:3, 14:16] = 2; gt_typ[gt_inst == 2] = 4  # border, missed     -> skip
    gt_inst[10:13, 10:13] = 3; gt_typ[gt_inst == 3] = 4  # internal, matched -> skip
    pred = np.zeros((16, 16), np.int32)
    pred[10:13, 10:13] = 1                             # IoU 1.0 with GT 3
    assert m.unmatched_internal_dead(gt_inst, gt_typ, pred) == [1]
    # a non-Dead missed internal instance is never a candidate
    gt_typ[gt_inst == 1] = 2
    assert m.unmatched_internal_dead(gt_inst, gt_typ, pred) == []


def test_dropout_shrink_exact_iou_and_determinism():
    m = _load("dsb_oracle_ceiling")
    mask = np.zeros((16, 16), bool)
    mask[4:8, 4:9] = True                              # area 20
    s1 = m.shrink_mask(mask, drop_frac=0.3, rng=np.random.default_rng(7))
    s2 = m.shrink_mask(mask, drop_frac=0.3, rng=np.random.default_rng(7))
    assert (s1 == s2).all() and s1.sum() == 14         # 20 * (1 - 0.3)
    inter, union = (s1 & mask).sum(), (s1 | mask).sum()
    assert abs(inter / union - 0.7) < 1e-9             # pixel dropout of f gives IoU exactly 1-f


@pytest.mark.parametrize("d_dead,verdict", [(0.012, "pass"), (0.010, "pass"), (0.0099, "fail")])
def test_gate_d_verdict_threshold(d_dead, verdict):
    m = _load("dsb_oracle_ceiling")
    assert m.gate_d_verdict(d_dead) == verdict


# ------------------------------------------------------------------ gate C

def test_gate_c_verdict_rule():
    m = _load("dsb_gate_c")
    rows = [{"tau_dead": 0.5, "d_dead": 0.0, "d_bpq": 0.0},
            {"tau_dead": 0.4, "d_dead": 0.006, "d_bpq": -0.0005}]
    assert m.gate_c_verdict(rows) == "close_line"   # decision decoupling alone already wins
    rows[1] = {"tau_dead": 0.4, "d_dead": 0.003, "d_bpq": -0.004}
    assert m.gate_c_verdict(rows) == "proceed"


# ------------------------------------------------------------------ gate A

def _nums(dead30, dead8, deadx2, bpq30=0.65, bpq8=0.65):
    return {"arms": {a: {"mean": {"Dead PQ": d, "bPQ": b}} for a, d, b in
                     [("x1_hv30", dead30, bpq30), ("x1_hv8", dead8, bpq8),
                      ("x2_hv30", deadx2, bpq30)]}}


def test_gate_a_rules():
    m = _load("dsb_gate_a")
    assert m.gate_a(_nums(.170, .176, .180))["base"] == "hv8"           # r=.006 >= .5*R=.005
    assert m.gate_a(_nums(.170, .1795, .180))["verdict"] == "downgrade" # r=.0095 >= .9*R, no tax
    assert m.gate_a(_nums(.170, .172, .180))["base"] == "hv30"          # r=.002 < .5*R
    assert m.gate_a(_nums(.170, .175, .169))["verdict"] == "proceed"    # R<=0: support not the driver
    assert m.gate_a(_nums(.170, .1795, .180, bpq8=.648))["verdict"] == "proceed"  # tax blocks downgrade
