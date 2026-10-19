import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

def module():
    p = Path(__file__).with_name('proposal_p3_utils.py')
    if Path(__file__).parent.name=='tests': p=Path(__file__).parent.parent/'scripts/proposal_p3_utils.py'
    assert p.exists(), 'P3 helpers are not implemented'
    spec = importlib.util.spec_from_file_location('p3', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def test_split_is_stratified_disjoint_and_stable():
    m=module(); tissue=np.array([0]*10+[1]*10+[2]); dead=np.array([False]*5+[True]*5+[False]*10+[True])
    fit,audit=m.split_indices(tissue,dead)
    assert sorted(fit+audit)==list(range(21))
    assert not set(fit)&set(audit)
    assert len(audit)==4 and 20 in fit
    assert (fit,audit)==m.split_indices(tissue,dead)

def test_duplicate_candidates_do_not_inflate_recall_or_precision():
    m=module(); gt=[(4,(10,10),True,20)]
    x=m.score_points([(10,10),(10,11),(10,12)],gt)
    assert x['matched']==1 and x['dead_hit']==1 and x['candidates']==3

def test_matching_maximizes_cardinality_instead_of_greedy():
    m=module(); gt=[(4,(0,0),True,20),(1,(0,4),True,40)]
    x=m.score_points([(0,2),(0,0)],gt)
    assert x['matched']==2 and x['dead_hit']==1

def test_no_candidates_and_no_objects():
    m=module(); x=m.score_points([],[(4,(10,10),True,20)])
    assert x['matched']==0 and x['dead_total']==1
    assert m.score_points([(1,1)],[])['matched']==0

def test_augment_geometry_matches_target_and_is_reproducible():
    m=module(); image=torch.zeros(3,16,16); image[:,2,5]=1
    target=torch.zeros(16,16);target[2,5]=1
    a,b=m.augment_pair(image,target,np.random.default_rng(3),photometric=False)
    assert torch.equal(a[0],b)
    c,d=m.augment_pair(image,target,np.random.default_rng(3),photometric=False)
    assert torch.equal(a,c) and torch.equal(b,d)

def test_partial_accumulation_weights_actual_images():
    m=module()
    assert m.accumulation_denominator(8,10,4)==2
    assert m.accumulation_denominator(0,10,4)==4

def test_decode_is_stable_and_capped():
    m=module(); p=np.zeros((8,8));p[1,1]=p[5,5]=.8
    assert m.decode_peaks(p,cap=1)==[(1,1)]

def test_ground_truth_point_lies_in_concave_object():
    m=module(); inst=np.zeros((8,8),int);inst[2:6,2]=1;inst[5,2:6]=1
    typ=(inst>0).astype(int)*4
    gt=m.gt_objects(inst,typ)
    assert len(gt)==1 and inst[gt[0][1]]==1 and gt[0][2]

def test_gate_requires_support_both_seeds_and_unique_precision():
    m=module(); base=[dict(image=i,tissue=0,matched=8,total=10,dead_hit=4,dead_total=10,candidates=10) for i in range(10)]
    new=[dict(x,matched=9,dead_hit=7) for x in base]
    assert m.admission({'19':{'F1':base,'T1':new},'1':{'F1':base,'T1':new}})['pass']
    assert not m.admission({'19':{'F1':base[:9],'T1':new[:9]},'1':{'F1':base,'T1':new}})['pass']

def trainer():
    p=Path(__file__).with_name('train_proposal_p3.py')
    if Path(__file__).parent.name=='tests': p=Path(__file__).parent.parent/'scripts/train_proposal_p3.py'
    assert p.exists(), 'P3 trainer is not implemented'
    import sys
    sys.path.insert(0,str(p.parent))
    spec=importlib.util.spec_from_file_location('p3trainer',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def test_only_last_twelve_blocks_are_trainable():
    m=trainer()
    enc=torch.nn.Module();enc.blocks=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(24)])
    enc.extra=torch.nn.Linear(2,2)
    m.configure_encoder(enc,True)
    assert not any(p.requires_grad for b in enc.blocks[:12] for p in b.parameters())
    assert all(p.requires_grad for b in enc.blocks[12:] for p in b.parameters())
    assert not any(p.requires_grad for p in enc.extra.parameters())
    assert not enc.training
    m.configure_encoder(enc,False)
    assert not any(p.requires_grad for p in enc.parameters())

def test_focal_accumulation_equals_single_effective_batch_with_unequal_density():
    m=module();logits=torch.randn(3,8,8,requires_grad=True);target=torch.zeros_like(logits)
    target[0,1,1]=1;target[1,1,1]=target[1,2,2]=1
    whole=m.focal_numerator(logits,target)/3
    expected=torch.autograd.grad(whole,logits)[0]
    partial=sum(m.focal_numerator(logits[i:i+1],target[i:i+1])/3 for i in range(3))
    actual=torch.autograd.grad(partial,logits)[0]
    assert torch.allclose(expected,actual,atol=1e-6)

def test_matching_prefers_nearest_before_stable_ties():
    m=module();gt=[(1,(1,1),True,4),(4,(1,3),True,4)]
    assert m.score_points([(1,3)],gt)['dead_hit']==1

def test_checkpoint_refuses_missing_trainable_parameter():
    m=trainer();model=torch.nn.Linear(2,1)
    with pytest.raises(ValueError): m.load_adapted(model,{'weight':model.weight.detach().clone()})
    m.load_adapted(model,model.state_dict())
