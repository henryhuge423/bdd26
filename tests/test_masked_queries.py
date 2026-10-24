import importlib.util
from pathlib import Path
from copy import deepcopy
import json
import torch
import pytest


def api():
    path=Path(__file__).with_name('probe_masked_queries.py')
    if not path.is_file():path=Path(__file__).resolve().parents[1]/'scripts/probe_masked_queries.py'
    assert path.is_file(),'E10 masked query mechanism is absent'
    spec=importlib.util.spec_from_file_location('e10probe',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_global_forward_is_bit_exact_and_input_unchanged():
    from nucseg.setmask.model import QueryMaskHead
    torch.manual_seed(13)
    head=QueryMaskHead(channels=4,dim=16,mask_dim=4,queries=4,layers=2,heads=4).eval()
    x=torch.randn(2,4,16,16);saved=x.clone()
    expected=head(x);got=api().query_forward(head,x,False)
    assert all(torch.equal(expected[k],got[k]) for k in expected)
    assert torch.equal(x,saved)


def test_block_mask_uses_correct_sign_and_batch_head_order():
    logits=torch.tensor([[[1.,-1.,0.],[-2.,3.,-4.]],[[1.,1.,1.],[-1.,-1.,-1.]]],requires_grad=True)
    got,stats=api().attention_mask(logits,2)
    assert got.dtype==torch.bool and got.shape==(4,2,3) and not got.requires_grad
    assert got.tolist()==[[[False,True,False],[True,False,True]],[[False,True,False],[True,False,True]],[[False,False,False],[False,False,False]],[[False,False,False],[False,False,False]]]
    assert stats['fallback_rows']==1 and stats['query_rows']==4


def test_masked_attention_excludes_blocked_spatial_content():
    from nucseg.setmask.model import QueryMaskHead
    torch.manual_seed(31)
    head=QueryMaskHead(channels=1,dim=4,mask_dim=1,queries=2,layers=1,heads=1).eval()
    with torch.no_grad():
        head.mask_projection.weight.fill_(1);head.mask_projection.bias.zero_()
        for p in head.mask_head.parameters():p.zero_()
        head.mask_head[-1].bias.fill_(1)
    x=torch.ones(1,1,16,16);x[:,:,:,8:]=-1
    changed=x.clone();changed[:,:,:,8:]=-20
    a=api().query_forward(head,x,True)['class_logits']
    b=api().query_forward(head,changed,True)['class_logits']
    assert torch.allclose(a,b,atol=1e-6,rtol=0)
    assert not torch.allclose(head(x)['class_logits'],head(changed)['class_logits'],atol=1e-4,rtol=0)


def test_all_blocked_mask_falls_back_to_finite_attention_and_gradients():
    from nucseg.setmask.model import QueryMaskHead
    torch.manual_seed(9)
    head=QueryMaskHead(channels=4,dim=16,mask_dim=4,queries=4,layers=2,heads=4).eval()
    x=torch.randn(2,4,16,16);before=x.clone();original=deepcopy(head.state_dict())
    out=api().query_forward(head,x,True)
    loss=sum(v.square().mean() for v in out.values());loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    torch.optim.AdamW(head.parameters(),lr=.001).step()
    assert torch.equal(x,before)
    assert any(not torch.equal(v,original[k]) for k,v in head.state_dict().items())
    mask,stats=api().attention_mask(torch.full((2,4,4),-1.),4)
    assert not mask.any() and stats['fallback_rows']==8


def fixture():
    row={'image':7,'n':4,'pred':5,'tp':2,'dead_n':2,'dead_good':1,'dead_typed_tp':1,'dead_pred':2,'raw_dead_good':1,'interior_dead_n':1,'interior_dead_good':1}
    rates={'recall':.5,'precision':.4,'dead_recall':.5,'dead_typed_f1':.5,'raw_dead_recall':.5,'interior_dead_recall':1.}
    return {**rates,'counts':{k:v for k,v in row.items() if k!='image'},'per_image':[row]}


def test_reuse_validation_rejects_corrupted_aggregate_and_wrong_images():
    m=api();r=fixture();m.validate_measurement(r,[7])
    broken=deepcopy(r);broken['counts']['tp']=3
    with pytest.raises(ValueError):m.validate_measurement(broken,[7])
    broken=deepcopy(r);broken['precision']=.9
    with pytest.raises(ValueError):m.validate_measurement(broken,[7])
    with pytest.raises(ValueError):m.validate_measurement(r,[8])


def test_original_gate_and_a0_guard_remain_separate():
    m=api()
    good={'dead_recall':.4,'recall':.6,'precision':.6,'dead_typed_f1':.4,'fit_recall':.9}
    global_={**good,'dead_recall':.3}
    a0={**good,'precision':.7}
    rows={s:{'A0':a0,'G':global_,'M':good} for s in ('19','1')}
    got=m.allocation(rows)
    assert got['original_gate'] and not got['a0_guard'] and not got['promising']
    for s in rows:rows[s]['A0']=good.copy()
    assert m.allocation(rows)['promising']
