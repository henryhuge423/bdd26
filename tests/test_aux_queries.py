import importlib.util
from pathlib import Path
from copy import deepcopy
import os,math
import torch
import pytest


def api():
    path=Path(__file__).with_name('probe_aux_queries.py')
    if not path.is_file():path=Path(__file__).resolve().parents[1]/'scripts/probe_aux_queries.py'
    assert path.is_file(),'E11 auxiliary supervision is absent'
    spec=importlib.util.spec_from_file_location('e11probe',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def base():
    path=Path(os.environ.get('E10_BASE_PATH',Path(__file__).resolve().parents[1]/'scripts/probe_masked_queries.py'))
    spec=importlib.util.spec_from_file_location('e10_frozen_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def head():
    from nucseg.setmask.model import QueryMaskHead
    torch.manual_seed(13)
    return QueryMaskHead(channels=4,dim=16,mask_dim=4,queries=4,layers=3,heads=4).eval()


def test_auxiliary_predictions_preserve_exact_final_inference_and_shared_features():
    h=head();x=torch.randn(2,4,16,16);saved=x.clone()
    expected=base().query_forward(h,x,True)
    stages=api().staged_forward(base(),h,x)
    assert len(stages)==4
    assert all(torch.equal(expected[k],stages[-1][k]) for k in expected)
    assert all(s['class_logits'].shape==(2,4,6) for s in stages)
    assert all(s['mask_features'].data_ptr()==stages[-1]['mask_features'].data_ptr() for s in stages)
    assert torch.equal(x,saved)


def test_four_stage_loss_keeps_total_scale_and_trains_every_stage():
    # Empty GT means only2*CE(background)+.1*CE(tissue). Uniform6/19-way logits:
    # total=2ln6+.1ln19; each of4 stage/background/query derivatives=-5/24.
    stages=[]
    for _ in range(4):
        stages.append({'class_logits':torch.zeros(1,2,6,requires_grad=True),
                       'mask_embed':torch.zeros(1,2,4,requires_grad=True),
                       'mask_features':torch.zeros(1,4,8,8,requires_grad=True)})
    batch={'inst_map':torch.zeros(1,8,8,dtype=torch.long),'tp_map':torch.zeros(1,8,8,dtype=torch.long),'tissue':torch.zeros(1,dtype=torch.long)}
    loss,record=api().auxiliary_loss(stages,batch,torch.zeros(1,19))
    assert float(loss.detach())==pytest.approx(2*math.log(6)+.1*math.log(19),abs=1e-6)
    assert len(record['stages'])==4
    loss.backward()
    for s in stages:
        assert torch.allclose(s['class_logits'].grad[...,0],torch.full((1,2),-5/24),atol=1e-6)
        assert torch.allclose(s['class_logits'].grad[...,1:],torch.full((1,2,5),1/24),atol=1e-6)


def test_real_auxiliary_training_has_finite_gradients_without_mutating_cached_input():
    h=head();original=deepcopy(h.state_dict());x=torch.randn(2,4,16,16);before=x.clone()
    stages=api().staged_forward(base(),h,x)
    inst=torch.zeros(2,16,16,dtype=torch.long);typ=inst.clone()
    inst[0,4:9,4:9]=7;typ[0,4:9,4:9]=4
    batch={'inst_map':inst,'tp_map':typ,'tissue':torch.zeros(2,dtype=torch.long)}
    loss,_=api().auxiliary_loss(stages,batch,torch.zeros(2,19))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
    torch.optim.AdamW(h.parameters(),lr=.001).step()
    assert torch.equal(before,x)
    assert any(not torch.equal(v,original[k]) for k,v in h.state_dict().items())


def test_incomplete_auxiliary_stage_list_is_rejected():
    with pytest.raises(ValueError):api().auxiliary_loss([],{},torch.zeros(1,19))


def test_allocation_uses_matched_m_control_and_retains_a0_guard():
    good={'dead_recall':.4,'recall':.6,'precision':.6,'dead_typed_f1':.4,'fit_recall':.9}
    control={**good,'dead_recall':.3};a0={**good,'precision':.7}
    rows={s:{'A0':a0,'M':control,'N':good} for s in ('19','1')}
    r=api().allocation(base(),rows)
    assert r['original_gate'] and not r['a0_guard'] and not r['promising']
    for s in rows:rows[s]['A0']=good.copy()
    assert api().allocation(base(),rows)['promising']
