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
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.31}, "n_active": 64}) == "proceed"
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.30}, "n_active": 64}) == "proceed"
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.29}, "n_active": 64}) == "fail"
    # v1-schema inputs recorded no activity: validity cannot be certified -> void, never fail
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.31}}) == "void"


def test_list_cosine_matches_concat():
    m = _load()
    ga = [torch.tensor([1.0, 0.0]), torch.tensor([2.0])]
    gb = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0])]
    assert abs(m.list_cosine(ga, gb) - float(m.grad_cosine(torch.cat(ga), torch.cat(gb)))) < 1e-9
    assert abs(m.list_cosine(ga, ga) - 1.0) < 1e-9


# --- amendment (2026-10-08): Dead-stratified sampling + active-batch denominator + per-layer clause

def test_dead_stratified_batches():
    import numpy as np
    m = _load()
    dead = list(range(10))
    b1 = m.dead_stratified_batches(dead, n_batches=6, batch=4, seed=1)
    b2 = m.dead_stratified_batches(dead, n_batches=6, batch=4, seed=1)
    assert b1 == b2                                          # deterministic under the fixed seed
    assert len(b1) == 6 and all(len(b) == 4 for b in b1)     # exact batch size (circular stream)
    assert all(set(b) <= set(dead) for b in b1)              # only Dead-positive images
    # the first permutation covers every Dead-positive image within ceil(10/4) batches
    seen = set()
    for b in b1[:3]:
        seen |= set(b)
    assert seen == set(dead)
    assert m.dead_stratified_batches([], n_batches=4, batch=4, seed=0) == []


def test_active_conflict_fraction():
    m = _load()
    cos = [0.2, -0.1, 0.0, -0.3]
    active = [True, True, True, False]
    frac, n_active = m.active_conflict_fraction(cos, active)
    assert n_active == 3                                     # inactive batch leaves the denominator
    assert abs(frac - 1 / 3) < 1e-12                         # the inactive -0.3 does not count
    assert m.active_conflict_fraction([0.1, 0.2], [False, False]) == (0.0, 0)


def test_per_layer_significance():
    import numpy as np
    m = _load()
    rng = np.random.default_rng(0)
    pos = {g: list(rng.normal(0.05, 0.01, 64)) for g in m.GROUPS}
    res = m.per_layer_significance(pos)
    assert not res["any_significant"]
    neg = {g: list(rng.normal(0.05, 0.01, 64)) for g in m.GROUPS}
    neg["decoder"] = list(rng.normal(-0.05, 0.01, 64))
    res2 = m.per_layer_significance(neg)
    assert res2["any_significant"]
    assert res2["modules"]["decoder"]["significant"]
    assert all(r["p_holm"] >= r["p"] - 1e-12 for r in res2["modules"].values())
    # degenerate input (constant cosines, n<2) is never significant
    res3 = m.per_layer_significance({g: [0.0] * 64 if g != "encoder" else [0.1]
                                     for g in m.GROUPS})
    assert not res3["any_significant"]


def test_verdict_rule_both_clauses_and_void():
    m = _load()
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.31}, "n_active": 64}) == "proceed"
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.30}, "n_active": 64}) == "proceed"
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.29}, "n_active": 64,
                             "per_layer": {"any_significant": False}}) == "fail"
    # per-layer clause alone passes the gate (spec §6 B: fraction >= .30 OR per-layer negative)
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.0}, "n_active": 64,
                             "per_layer": {"any_significant": True}}) == "proceed"
    # too few informative batches -> VOID, not fail (the 00:58 run's failure mode)
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.5}, "n_active": 8}) == "void"


def test_verdict_ignores_none_per_layer():
    m = _load()
    # per_layer=None (result-template state) must not crash; the fraction clause decides
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.31}, "n_active": 64,
                             "per_layer": None}) == "proceed"
    assert m.gate_b_verdict({"decoder": {"conflict_fraction": 0.29}, "n_active": 64,
                             "per_layer": None}) == "fail"


def test_dead_positive_indices_column():
    import numpy as np

    class FakeDS:
        # column 3 (DEAD_TYPE-1) is the Dead column: only images 1 and 3 are Dead-positive
        def class_presence(self):
            return np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0],
                             [0, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1]])
    m = _load()
    assert list(m.dead_positive_indices(FakeDS())) == [1, 3]


def test_batch_group_activity():
    import torch
    m = _load()
    tp = torch.zeros(1, 4, 4, dtype=torch.long)
    np_ = torch.zeros(1, 4, 4, dtype=torch.long)
    assert m.batch_group_activity(tp, np_) == (False, False)      # background only
    tp[0, 0, 0] = 4
    assert m.batch_group_activity(tp, np_) == (True, False)       # Dead only, no foreground mask
    np_[0, 1, 1] = 1
    tp[0, 1, 1] = 2
    assert m.batch_group_activity(tp, np_) == (True, True)        # both groups active
    tp[tp == 4] = 0
    assert m.batch_group_activity(tp, np_) == (False, True)       # common only
