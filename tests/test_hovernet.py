"""HoVer-Net plumbing: HV targets at borders, full-patch output shape, TTA inverse transform."""

import numpy as np
import pytest
import torch

from nucseg.data.pannuke import PanNukeFold
from nucseg.hovernet.data import PanNukeHoVer, hv_targets
from nucseg.hovernet.engine import build_model, dihedral, forward_crop, undo_dihedral


@pytest.fixture(scope="module")
def fold():
    f = PanNukeFold(1)
    if not (f.dir / "inst.npy").exists():
        pytest.skip("PanNuke fold1 not prepared")
    return f


def test_hv_targets_cover_border_nuclei(fold):
    inst = np.asarray(fold.inst[0]).astype(np.int32)
    hv = hv_targets(inst)
    border_ids = set(np.unique(np.r_[inst[0], inst[-1], inst[:, 0], inst[:, -1]])) - {0}
    for i in border_ids:
        m = inst == i
        if m.sum() >= 30:
            assert np.abs(hv[m]).max() > 0.5, f"border nucleus {i} got no hv target"


def test_dataset_and_model_shapes(fold):
    ds = PanNukeHoVer([1], train=True)
    b = ds[0]
    assert b["img"].shape == (352, 352, 3) and b["hv_map"].shape == (256, 256, 2)
    model = build_model(freeze=False).eval()
    with torch.no_grad():
        out = forward_crop(model, b["img"][None])
    assert out["np"].shape == (1, 256, 256, 2) and out["tp"].shape == (1, 256, 256, 6)


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("flip", [False, True])
def test_undo_dihedral_hv(fold, k, flip):
    inst = torch.from_numpy(np.asarray(fold.inst[3]).astype(np.int64))[None]
    ref = torch.from_numpy(hv_targets(inst[0].numpy()))[None]
    t_inst = dihedral(inst, k, flip)[0].numpy()
    t_hv = torch.from_numpy(hv_targets(np.ascontiguousarray(t_inst)))[None]
    back = undo_dihedral({"hv": t_hv}, k, flip)["hv"]
    naive = torch.rot90(t_hv.flip(2) if flip else t_hv, -k, (1, 2))  # pixels moved, vectors not remapped
    fg = inst[0] > 0
    err = (back - ref)[0][fg].abs().mean()
    err_naive = (naive - ref)[0][fg].abs().mean()
    # residual comes only from the official 1-px centre-of-mass rounding (exactly 0 when the
    # transform keeps axis orientation); a sign/axis bug gives errors ~ O(1)
    assert err < 0.15, f"k={k} flip={flip} mean abs err {err:.3f}"
    if (k, flip) != (0, False):
        assert err < 0.25 * err_naive
