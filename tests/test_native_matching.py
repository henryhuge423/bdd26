import importlib.util,math
from pathlib import Path
import pytest
import torch

def api():
    p=Path(__file__).with_name('probe_native_matching.py')
    if not p.exists():p=Path(__file__).resolve().parents[1]/'scripts'/p.name
    if not p.exists():
        class Missing:pass
        return Missing()
    spec=importlib.util.spec_from_file_location('native',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def case():
    from nucseg.setmask.matching import make_targets
    inst=torch.zeros(1,16,16,dtype=torch.long);inst[0,0,0]=1;typ=(inst>0).long()*4
    t=make_targets(inst,typ)
    f=torch.full((1,2,16,16),-8.);f[0,0,0,0]=4;f[0,0,1:4,1:4]=4;f[0,1,0,0]=2
    return torch.zeros(1,2,6),torch.eye(2)[None]*math.sqrt(2),f,t

def test_native_match_rejects_false_positive_island_hidden_between_sampled_points():
    from nucseg.setmask.matching import match_queries
    args=case();assert match_queries(*args)[0][0].tolist()==[0]
    assert api().native_match_queries(*args)[0][0].tolist()==[1]

def test_exact_cost_matches_direct_full_pixel_loss():
    m=api();c,e,f,t=case();cost=m.native_cost(c[0],e[0],f[0],t[0])
    truth=(t[0]['inst']==1).double();expected=[]
    for q in range(2):
        z=f[0,q].double();p=z.sigmoid()
        bce=torch.nn.functional.binary_cross_entropy_with_logits(z,truth)
        dice=1-(2*(p*truth).sum()+1)/(p.sum()+truth.sum()+1)
        expected.append(-2/6+5*bce+5*dice)
    assert torch.allclose(cost[:,0].double(),torch.stack(expected),atol=1e-6)
    assert not cost.requires_grad

def test_empty_targets_and_capacity_are_guarded():
    from nucseg.setmask.matching import make_targets
    m=api();c,e,f,t=case();empty=make_targets(torch.zeros(1,16,16,dtype=torch.long),torch.zeros(1,16,16,dtype=torch.long))
    q,g=m.native_match_queries(c,e,f,empty)[0];assert q.numel()==g.numel()==0
    with pytest.raises(ValueError):m.native_match_queries(c[:,:0],e[:,:0],f,t)

def test_nonfinite_native_matching_is_rejected():
    m=api();c,e,f,t=case();f[0,0,1,1]=float('nan')
    with pytest.raises(ValueError):m.native_match_queries(c,e,f,t)
