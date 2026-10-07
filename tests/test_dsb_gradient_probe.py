"""Pure-logic tests for the gradient-conflict probe (DSB plan Task 10)."""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "dsb_gradient_probe.py"
    spec = importlib.util.spec_from_file_location("dsb_gradient_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_cosine_math():
    m = _load()
    g = torch.tensor([1.0, 2.0])
    assert abs(float(m.grad_cosine(g, g)) - 1.0) < 1e-9
    assert abs(float(m.grad_cosine(g, -g)) + 1.0) < 1e-9
    assert abs(float(m.grad_cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 5.0])))) < 1e-9


def test_conflict_fraction():
    m = _load()
    assert m.conflict_fraction([0.2, -0.1, -0.3, 0.0]) == 0.5   # 2 of 4 batches negative


def test_group_pixel_loss_masking():
    m = _load()
    B, H, W = 1, 4, 4
    np_logits = torch.zeros(B, H, W, 2, requires_grad=True)
    tp_logits = torch.zeros(B, H, W, 6, requires_grad=True)
    hv = torch.zeros(B, H, W, 2, requires_grad=True)
    tp = torch.zeros(B, H, W, dtype=torch.long)
    tp[0, :2] = 4                                   # Dead half, non-Dead half
    tp[0, 2:, 2:] = 2
    np_map = (tp > 0).long()
    hv_map = torch.zeros(B, H, W, 2)
    l_dead = m.group_pixel_loss(np_logits, tp_logits, hv, np_map, tp, hv_map, "dead")
    l_common = m.group_pixel_loss(np_logits, tp_logits, hv, np_map, tp, hv_map, "common")
    assert l_dead.grad_fn is not None and l_common.grad_fn is not None
    # removing every Dead pixel zeroes the dead-group loss but leaves it graph-attached
    tp2 = tp.clone()
    tp2[tp2 == 4] = 2
    assert float(m.group_pixel_loss(np_logits, tp_logits, hv, np_map, tp2, hv_map, "dead")) == 0.0
    # background pixels belong to neither group
    assert float(m.group_pixel_loss(np_logits, tp_logits, hv, torch.zeros_like(np_map),
                                    torch.zeros_like(tp), hv_map, "common")) == 0.0


def test_verdict_rule():
    m = _load()
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.31}}) == "proceed"
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.30}}) == "proceed"
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.29}}) == "fail"


def test_list_cosine_matches_concat():
    m = _load()
    ga = [torch.tensor([1.0, 0.0]), torch.tensor([2.0])]
    gb = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0])]
    assert abs(m.list_cosine(ga, gb) - float(m.grad_cosine(torch.cat(ga), torch.cat(gb)))) < 1e-9
    assert abs(m.list_cosine(ga, ga) - 1.0) < 1e-9
