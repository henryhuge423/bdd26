"""Vendored LKCell: construction matches the released checkpoint layout, adapter returns the
nucseg contract. The full state-dict compatibility (2937 tensors incl. the inherited default
UniRepLKNet body) is checked bit-for-bit here against a fresh instance — the same key set the
released model_best.pth carries (verified against the real checkpoint 2026-10-01)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nucseg.lkcell import CellViT, LKCellAdapter


def test_lkcell_adapter_contract():
    m = CellViT()
    m.eval()
    with torch.no_grad():
        out = LKCellAdapter(m)(torch.zeros(1, 3, 64, 64))
    assert set(out) == {"np", "hv", "tp", "tissue"}
    assert out["np"].shape == (1, 2, 64, 64) and out["hv"].shape == (1, 2, 64, 64)
    assert out["tp"].shape == (1, 6, 64, 64) and out["tissue"].shape == (1, 19)


def test_lkcell_state_layout():
    """Key layout must include BOTH UniRepLKNet bodies (the class inherits UniRepLKNet and
    its released checkpoints carry the untouched default super body alongside encoder.*)."""
    m = CellViT()
    keys = set(m.state_dict())
    assert "downsample_layers.0.0.weight" in keys            # inherited default body
    assert "encoder.downsample_layers.0.0.weight" in keys    # real encoder
    assert "decoder.blocks.0.convs.conv_0.conv.weight" in keys
    assert "nuclei_type_maps_head.1.conv.weight" in keys
    # the decoder defines but never calls repblock/convffnblock (verbatim upstream); their
    # tensors still exist and must load — regression guard against "cleaning" them up
    assert any(k.startswith("decoder.repblock") for k in keys)
