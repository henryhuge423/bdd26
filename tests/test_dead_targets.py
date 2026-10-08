"""Dead-instance NP/HV targets and the post-augmentation presence flag (DSB spec §4–5)."""

import numpy as np
import pytest
import torch

from nucseg.cellvit.data import PanNukeCellViT
from nucseg.constants import DEAD_TYPE

HISTORICAL_KEYS = {"img", "np_map", "hv_map", "tp_map", "small_map", "tissue", "index"}


@pytest.fixture
def tiny_fold(tmp_path, monkeypatch):
    root = tmp_path / "data"
    fold = root / "fold1"
    fold.mkdir(parents=True)
    inst = np.zeros((2, 256, 256), np.int32)
    inst[0, 20:24, 20:24] = 7                      # Dead nucleus (16 px)
    inst[0, 50:56, 50:56] = 42                      # other class
    inst[1, 10:16, 10:16] = 9                       # image 1: non-Dead only
    typ = np.zeros((2, 256, 256), np.uint8)
    typ[inst == 7] = DEAD_TYPE
    typ[inst == 42] = 1
    typ[inst == 9] = 2
    np.save(fold / "images.npy", np.full((2, 256, 256, 3), 128, np.uint8))
    np.save(fold / "inst.npy", inst)
    np.save(fold / "type.npy", typ)
    np.save(fold / "tissue.npy", np.array(["Colon", "Colon"]))
    monkeypatch.setenv("PANNUKE_ROOT", str(root))
    return inst, typ


def test_dead_targets_present_and_correct(tiny_fold):
    inst, typ = tiny_fold
    ds = PanNukeCellViT([1], train=False, dead_targets=True)
    s = ds[0]
    dead = typ[0] == DEAD_TYPE
    assert set(np.unique(s["dead_np_map"].numpy())) <= {0, 1}
    assert torch.equal(s["dead_np_map"], torch.from_numpy(dead.astype(np.int64)))
    assert tuple(s["dead_hv_map"].shape) == (256, 256, 2)
    assert bool(s["dead_pos"]) is True
    # non-Dead foreground must never leak into the dead channels
    assert s["dead_np_map"][typ[0] == 1].max() == 0


def test_dead_free_image_has_zero_targets_and_negative_flag(tiny_fold):
    ds = PanNukeCellViT([1], train=False, dead_targets=True)
    s = ds[1]
    assert not s["dead_np_map"].any()
    assert not s["dead_hv_map"].any()
    assert bool(s["dead_pos"]) is False


def test_dead_targets_absent_by_default(tiny_fold):
    ds = PanNukeCellViT([1], train=False)
    assert set(ds[0].keys()) == HISTORICAL_KEYS


def test_dead_pos_reflects_augmented_content(tiny_fold):
    # train=True runs the augmentation stack; whatever it produces, dead_pos must be
    # exactly "the augmented sample contains Dead pixels" — never the pre-augmentation answer
    ds = PanNukeCellViT([1], train=True, dead_targets=True)
    for _ in range(8):
        for i in (0, 1):
            s = ds[i]
            assert bool(s["dead_pos"]) == bool((s["tp_map"] == DEAD_TYPE).any())
            assert torch.equal(s["dead_np_map"], (s["tp_map"] == DEAD_TYPE).to(torch.int64))
