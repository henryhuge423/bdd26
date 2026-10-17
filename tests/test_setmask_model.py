import gc
import hashlib
import importlib
import importlib.util

import pytest
import torch


def api():
    assert importlib.util.find_spec('nucseg.setmask.model') is not None,'Query mask model is absent'
    return importlib.import_module('nucseg.setmask.model')


def test_query_head_preserves_native_features_and_has_finite_gradients():
    m=api();torch.manual_seed(3)
    head=m.QueryMaskHead(channels=4,dim=16,mask_dim=4,queries=3,layers=1,heads=4)
    x=torch.randn(2,4,16,16,requires_grad=True);out=head(x)
    assert out['class_logits'].shape==(2,3,6)
    assert out['mask_embed'].shape==(2,3,4)
    assert out['mask_features'].shape==(2,4,16,16)
    loss=sum(v.square().mean() for v in out.values());loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_query_head_eval_is_repeatable():
    head=api().QueryMaskHead(channels=4,dim=16,mask_dim=4,queries=3,layers=1,heads=4).eval()
    x=torch.randn(1,4,16,16)
    with torch.no_grad():a=head(x);b=head(x)
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_shared_initial_parameters_match_baseline_without_changing_it():
    m=api()
    from nucseg.cellvit.model import CellViTUNI
    prefixes=('encoder.','skip0.','skip1.','skip2.','skip3.','tp_branch.','tissue_head.')
    def hashes(model):
        return {k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()
                for k,v in model.state_dict().items() if k.startswith(prefixes)}
    torch.manual_seed(771);base=CellViTUNI();expected=hashes(base)
    assert hasattr(base,'np_branch') and hasattr(base,'hv_branch') and hasattr(base,'tp_head')
    del base;gc.collect()
    torch.manual_seed(771);query=m.SetMaskUNI()
    assert hashes(query)==expected
    assert not hasattr(query,'np_branch') and not hasattr(query,'hv_branch') and not hasattr(query,'tp_head')
    query.freeze_encoder(True)
    assert not any(p.requires_grad for p in query.encoder.parameters())
    query.freeze_encoder(False)
    assert all(p.requires_grad for p in query.encoder.parameters())


def test_invalid_query_head_dimensions_reject_early():
    with pytest.raises(ValueError):api().QueryMaskHead(channels=4,dim=15,heads=4)
