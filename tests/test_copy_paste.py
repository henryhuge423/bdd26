"""Copy-paste augmentation invariants (pillar B CP1)."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from nucseg.augment.copy_paste import CopyPasteConfig, NucleusBank, apply_copy_paste
from nucseg.data.pannuke import PanNukeFold


@pytest.fixture(scope="module")
def fold1():
    return PanNukeFold(1)


@pytest.fixture(scope="module")
def bank(fold1):
    return NucleusBank([fold1])


def test_bank_matches_manual_counts(fold1, bank):
    f = fold1
    for j in (0, 7, 100):
        rows = bank.rows[bank.rows.img == j]
        assert len(rows) == len(np.unique(np.asarray(f.inst[j]))) - 1
        inst = np.asarray(f.inst[j])
        typ = np.asarray(f.type[j])
        for r in rows:
            # donor class = majority type inside the instance
            assert r.cls == np.bincount(typ[inst == r.id], minlength=8).argmax()
            assert r.area == int((inst == r.id).sum())
            ys, xs = np.nonzero(inst == r.id)
            assert (r.y0, r.y1, r.x0, r.x1) == (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)


def test_bank_has_dead_and_small_donors(bank):
    dead = bank.by_class[4]
    assert len(dead) > 100
    areas = bank.rows.area[dead]
    assert ((areas >= 50) & (areas <= 400)).mean() > 0.5


def test_bank_excludes_edge_clipped_donors(fold1, bank):
    """Donors touching the patch border would be pasted as straight-edged fragments."""
    H, W = np.asarray(fold1.inst[0]).shape
    r = bank.rows
    assert r.edge.sum() > 0
    flagged = (r.y0[r.edge] < 2) | (r.x0[r.edge] < 2) | (r.y1[r.edge] > H - 2) | (r.x1[r.edge] > W - 2)
    assert flagged.all()
    assert (r.y0[~r.edge] >= 2).all() and (r.x0[~r.edge] >= 2).all()
    assert (r.y1[~r.edge] <= H - 2).all() and (r.x1[~r.edge] <= W - 2).all()


def test_copy_paste_invariants(fold1, bank):
    cfg = CopyPasteConfig(prob=1.0, lam=4, clearance=8)
    rng = np.random.default_rng(0)
    f = fold1
    n_changed = 0
    for j in range(30):
        img = np.asarray(f.images[j]).copy()
        inst = np.asarray(f.inst[j]).astype(np.int32)
        typ = np.asarray(f.type[j]).astype(np.int32)
        img2, inst2, typ2 = apply_copy_paste(img.copy(), inst.copy(), typ.copy(), bank, cfg, rng)
        # labels only ever grow, ids stay unique, type valid, background consistent
        assert (inst2 > 0).sum() >= (inst > 0).sum()
        ids = np.unique(inst2)
        assert len(ids) == ids.max() + 1  # 0..max contiguous
        assert set(np.unique(typ2[inst2 == 0])) <= {0}
        assert set(np.unique(typ2[inst2 > 0])) <= set(range(1, 6))
        # every original nucleus is untouched
        for k in np.unique(inst)[1:]:
            assert (typ2[inst2 == k] == typ[inst == k][0]).all()
        if (inst2 != inst).any():
            n_changed += 1
            # pasted nuclei are isolated: no other nucleus within `clearance`
            new_ids = set(np.unique(inst2)) - set(np.unique(inst)) - {0}
            for k in new_ids:
                grown = ndi.binary_dilation(inst2 == k, iterations=cfg.clearance)
                assert not (grown & (inst2 > 0) & (inst2 != k)).any()
                # pasted pixels carry a real class and non-trivial size
                assert typ2[inst2 == k].min() >= 1
                assert (inst2 == k).sum() >= 20
            # strong image changes stay confined to the pasted regions (+ ~4px feather bleed)
            diff = np.abs(img2.astype(int) - img.astype(int)).sum(-1) > 25
            near = ndi.binary_dilation((inst2 != inst), iterations=4)
            if diff.any():
                assert (diff & near).sum() / diff.sum() > 0.95
    assert n_changed >= 25


def test_copy_paste_respects_prob(fold1, bank):
    cfg = CopyPasteConfig(prob=0.0)
    rng = np.random.default_rng(1)
    img = np.asarray(fold1.images[0])
    inst = np.asarray(fold1.inst[0]).astype(np.int32)
    typ = np.asarray(fold1.type[0]).astype(np.int32)
    out = apply_copy_paste(img, inst, typ, bank, cfg, rng)
    assert out[0] is img and out[1] is inst


def test_donor_class_mix(fold1, bank):
    """Dead should be over-represented among donors but not exclusive (no shortcut)."""
    cfg = CopyPasteConfig(prob=1.0, lam=6, dead_w=0.4)
    rng = np.random.default_rng(3)
    f = fold1
    img = np.asarray(f.images[0])
    inst = np.asarray(f.inst[0]).astype(np.int32)
    typ = np.asarray(f.type[0]).astype(np.int32)
    counts = np.zeros(8, int)
    trials = 0
    for _ in range(40):
        _, inst2, typ2 = apply_copy_paste(img.copy(), inst.copy(), typ.copy(), bank, cfg, rng)
        if (inst2 != inst).any():
            trials += 1
            for k in set(np.unique(inst2)) - set(np.unique(inst)) - {0}:
                counts[typ2[inst2 == k][0]] += 1
    assert trials >= 30
    assert counts[4] > 0 and counts.sum() - counts[4] > 0
    assert 0.15 < counts[4] / counts.sum() < 0.75
