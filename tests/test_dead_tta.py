"""TTA dihedral remap must cover every HV channel, including the dead expert's."""

import torch

from nucseg.hovernet.engine import undo_dihedral


def test_undo_dihedral_remaps_all_hv_keys():
    hv = torch.randn(2, 4, 4, 2)
    for k in range(4):
        for flip in (False, True):
            out = undo_dihedral({"hv": hv.clone(), "hv_dead": hv.clone()}, k, flip)
            assert torch.allclose(out["hv_dead"], out["hv"])
    # a plain (non-HV) extra key is only spatially transformed (NHWC dict), never sign-flipped
    img = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4, 1)
    out = undo_dihedral({"hv": hv.clone(), "np": img.clone()}, 1, False)
    assert out["np"].shape == img.shape
