import torch

from nucseg.cellvit.model import CellViTUNI, branch_param_count, solve_c1_widen


def _tiny(**kw):
    m = CellViTUNI(uni_ckpt=None, dead_expert=kw.pop("dead_expert", False), **kw)
    # shrink encoder blocks so the test is fast; decoder untouched (what we test).
    # forward unpacks exactly 4 extracted layers, so keep 4 blocks and retarget them.
    m.encoder.blocks = torch.nn.ModuleList(m.encoder.blocks[:4])
    m.encoder.extract_layers = {1, 2, 3, 4}
    return m


def test_default_off_state_dict_identical():
    base, ext = _tiny(), _tiny()
    assert set(base.state_dict()) == set(ext.state_dict())
    assert not any("dead" in k for k in base.state_dict())


def test_dead_expert_forward_shapes():
    with torch.no_grad():
        out = _tiny(dead_expert=True).eval()(torch.randn(2, 3, 256, 256))
    assert out["np_dead"].shape == (2, 2, 256, 256)
    assert out["hv_dead"].shape == (2, 2, 256, 256)
    with torch.no_grad():
        plain = _tiny().eval()(torch.randn(1, 3, 256, 256))
    assert "np_dead" not in plain and "hv_dead" not in plain


def test_c1_widen_param_parity():
    dsb = CellViTUNI(uni_ckpt=None, dead_expert=True)
    c1 = CellViTUNI(uni_ckpt=None, widen=solve_c1_widen())
    p_dsb = sum(branch_param_count(dsb).values())
    p_c1 = sum(branch_param_count(c1).values())
    assert abs(p_dsb - p_c1) / p_dsb < 0.015
