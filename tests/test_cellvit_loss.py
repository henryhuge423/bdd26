"""Targeted foreground loss (TrainConfig.np_wce), the small-nucleus map and the masked dead loss."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nucseg.cellvit.data import small_nuclei
from nucseg.cellvit.engine import DEAD, TrainConfig, cellvit_loss


def _batch(rng):
    inst = np.zeros((2, 32, 32), np.int64)
    inst[:, 2:6, 2:6] = 1        # 16 px nucleus
    inst[:, 10:30, 10:30] = 2    # 400 px nucleus
    tp = np.where(inst == 1, DEAD, np.where(inst == 2, 1, 0))
    return {
        "np_map": torch.from_numpy((inst > 0).astype(np.int64)),
        "tp_map": torch.from_numpy(tp),
        "hv_map": torch.zeros(2, 32, 32, 2),
        "small_map": torch.from_numpy(np.stack([small_nuclei(i, 100) for i in inst])),
        "tissue": torch.zeros(2, dtype=torch.long),
    }, {"np": torch.from_numpy(rng.normal(size=(2, 32, 32, 2))).float(),
        "hv": torch.zeros(2, 32, 32, 2), "tp": torch.from_numpy(rng.normal(size=(2, 32, 32, 6))).float(),
        "tissue": torch.zeros(2, 19)}


def test_small_nuclei_map():
    inst = np.zeros((8, 8), np.int64)
    inst[:2, :2] = 3
    inst[4:, 4:] = 5
    assert small_nuclei(inst, 5).sum() == 4 and not small_nuclei(inst, 5)[0, 7]


def test_default_config_is_recipe_and_wce_matches_manual():
    if not torch.cuda.is_available():
        pytest.skip("official msge_loss builds its Sobel kernels on cuda")
    batch, pred = _batch(np.random.default_rng(0))
    batch = {k: v.cuda() for k, v in batch.items()}
    pred = {k: v.cuda() for k, v in pred.items()}
    base, terms = cellvit_loss(pred, batch)
    same, terms2 = cellvit_loss(pred, batch, TrainConfig(split=1, out_dir="x"))
    assert torch.equal(base, same) and "np_wce" not in terms2
    cfg = TrainConfig(split=1, out_dir="x", np_wce=1.0, dead_w=10.0, small_w=2.0)
    loss, t = cellvit_loss(pred, batch, cfg)
    ce = F.cross_entropy(pred["np"].permute(0, 3, 1, 2), batch["np_map"], reduction="none")
    w = 1 + 10.0 * (batch["tp_map"] == DEAD).float() + 2.0 * batch["small_map"].float()
    assert abs(t["np_wce"] - float((ce * w).mean())) < 1e-6
    assert torch.isclose(loss, base + (ce * w).mean())
    assert float(w[0, 3, 3]) == 13.0 and float(w[0, 20, 20]) == 1.0 and float(w[0, 0, 31]) == 1.0


# ---------------------------------------------------------------- dead-expert loss
# The official msge_loss builds its Sobel kernels on CUDA, so the masked-dead tests run the
# loss with a CPU stand-in for it (the dead logic under test never touches msge internals;
# the real operator is covered by the CUDA test above and by GPU smoke runs).

H = W = 8


def _dead_pred():
    r = lambda *s: torch.randn(1, *s, requires_grad=True)
    return {"np": r(H, W, 2), "hv": r(H, W, 2), "tp": r(H, W, 6),
            "tissue": torch.randn(1, 19, requires_grad=True),
            "np_dead": r(H, W, 2), "hv_dead": r(H, W, 2)}


def _dead_batch(positive=True):
    tp = torch.zeros(1, H, W, dtype=torch.long)
    if positive:
        tp[0, 2:5, 2:5] = DEAD  # Dead nucleus
    return {"np_map": (tp > 0).long(), "tp_map": tp, "hv_map": torch.zeros(1, H, W, 2),
            "small_map": torch.zeros(1, H, W, dtype=torch.bool), "tissue": torch.zeros(1, dtype=torch.long),
            "dead_np_map": (tp == DEAD).long(), "dead_hv_map": torch.zeros(1, H, W, 2),
            "dead_pos": torch.tensor(positive)}


@pytest.fixture
def cpu_msge(monkeypatch):
    from nucseg.cellvit import engine
    monkeypatch.setattr(engine, "msge_loss",
                        lambda true, pred, mask: (true - pred).abs().mean() * mask.sum())


def test_dead_terms_zero_but_attached_on_negative_batch(cpu_msge):
    cfg = TrainConfig(split=1, out_dir="x", dead_expert=True)
    loss, terms = cellvit_loss(_dead_pred(), _dead_batch(positive=False), cfg)
    for k in ("dead_np_ft", "dead_np_dice", "dead_hv_mse", "dead_hv_msge"):
        assert terms[k] == 0.0
    loss.backward()  # must not raise "does not require grad"


def test_dead_terms_positive_and_neg_bce(cpu_msge):
    cfg = TrainConfig(split=1, out_dir="x", dead_expert=True, dead_neg_w=0.1)
    _, terms = cellvit_loss(_dead_pred(), _dead_batch(positive=True), cfg)
    assert terms["dead_np_ft"] > 0 and terms["dead_hv_mse"] >= 0
    assert "dead_neg_bce" not in terms  # positive image: no negative-BCE term
    _, terms = cellvit_loss(_dead_pred(), _dead_batch(positive=False), cfg)
    assert terms["dead_neg_bce"] > 0  # negative images push dead-fg logits down
    assert terms["dead_np_ft"] == 0.0


def test_default_cfg_no_dead_terms(cpu_msge):
    loss, terms = cellvit_loss(_dead_pred(), _dead_batch(), TrainConfig(split=1, out_dir="x"))
    assert not any(k.startswith("dead_") for k in terms)
