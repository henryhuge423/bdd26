"""P7 RGB augmentation must preserve control, labels and evaluation."""
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
torch.set_num_threads(2)

def p7():
    assert importlib.util.find_spec('train_proposal_p7') is not None,'P7 trainer is missing'
    return importlib.import_module('train_proposal_p7')

def test_rgb_gain_analytical_clamp_and_no_mutation():
    m=p7();image=torch.tensor([[[.25,.9]],[[.5,1.]],[[.75,0.]]],dtype=torch.float32)
    before=image.clone();actual=m.apply_rgb_gains(image,np.array([.8,1.2,1.1],np.float32))
    assert torch.allclose(actual,torch.tensor([[[.2,.72]],[[.6,1.]],[[.825,0.]]]),atol=1e-7)
    assert torch.equal(image,before) and actual.dtype==torch.float32

def test_rng_identity_keys_bounds_and_global_state_isolation():
    m=p7();np.random.seed(23);before=np.random.get_state();torch.manual_seed(23);tbefore=torch.get_rng_state().clone()
    a=m.rgb_gains(19,3,17)
    assert a.shape==(3,) and a.dtype==np.float32 and np.all((a>=.8)&(a<=1.2))
    assert np.array_equal(a,m.rgb_gains(19,3,17))
    for key in [(1,3,17),(19,4,17),(19,3,18)]:assert not np.array_equal(a,m.rgb_gains(*key))
    after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert torch.equal(tbefore,torch.get_rng_state())

def make_fold(tmp_path):
    root=tmp_path/'fold1';root.mkdir()
    image=np.random.default_rng(11).integers(10,230,(2,256,256,3),dtype=np.uint8)
    inst=np.zeros((2,256,256),np.uint16);typ=np.zeros_like(inst,dtype=np.uint8)
    inst[0,15:20,22:26]=1;typ[0,15:20,22:26]=4
    for name,value in [('images',image),('inst',inst),('type',typ),('tissue',np.array(['Lung','Skin']))]:np.save(root/(name+'.npy'),value)

def test_batches_preserve_control_geometry_counts_and_eval(tmp_path):
    m=p7();import train_proposal_p3 as old
    make_fold(tmp_path);c=m.Fold(tmp_path,1,'C');j=m.Fold(tmp_path,1,'J');ref=old.Fold(tmp_path,1)
    for ids in ([0,1],[1]):
        a=c.batch(ids,19,3,True);b=j.batch(ids,19,3,True);r=ref.batch(ids,19,3,True)
        assert torch.equal(a[0],r[0]) and torch.equal(a[1],r[1])
        assert torch.equal(a[1],b[1]) and a[2]==b[2]==r[2]==(1 if 0 in ids else 0)
        # Independent calculation from the published SeedSequence contract; no helper for expectation.
        gains=np.stack([np.random.default_rng(np.random.SeedSequence([19,3,int(i),20261021,7])).uniform(.8,1.2,3).astype(np.float32) for i in ids])
        expected=(r[0]*torch.from_numpy(gains)[:,:,None,None]).clamp(0,1)
        assert torch.equal(b[0],expected) and not torch.equal(a[0],b[0])
        ae=c.batch(ids,19,3,False);be=j.batch(ids,19,3,False);re=ref.batch(ids,19,3,False)
        assert torch.equal(ae[0],be[0]) and torch.equal(ae[0],re[0])
        assert torch.equal(ae[1],be[1]) and ae[2]==be[2]

def test_targets_empty_tail_normalization_and_original_scoring():
    m=p7();import train_proposal_p3 as old;import proposal_p3_utils as u
    assert m.Proposal is old.Proposal
    for name in ('split_indices','gaussian_target','augment_pair','focal_numerator','decode_peaks','score_points','aggregate','admission'):assert getattr(m,name) is getattr(u,name)
    targets=torch.from_numpy(np.stack([m.gaussian_target([(4,(15,15),True,20)]),m.gaussian_target([]),m.gaussian_target([(1,(20,20),True,30),(4,(30,30),True,20)])]))
    full=torch.full_like(targets,-1.5,requires_grad=True);expected=torch.autograd.grad(m.focal_numerator(full,targets)/3,full)[0]
    micro=torch.full_like(targets,-1.5,requires_grad=True)
    for a,b in ((0,2),(2,3)):(m.focal_numerator(micro[a:b],targets[a:b])/3).backward()
    assert torch.equal(expected,micro.grad) and not targets[1].any()

def test_config_rejects_other_recipe_arm_or_gain_policy():
    m=p7();config=m.run_config('J',1);m.validate_config(config,config)
    assert config['augmentation']=={'kind':'rgb_gain','low':.8,'high':1.2,'rng_salt':[20261021,7],'order':'after_original_augmentation'}
    assert m.run_config('C',19)['augmentation']=={'kind':'original'}
    for saved in [m.run_config('C',1),{**config,'seed':19},{**config,'augmentation':{'kind':'original'}},{'phase':'A','arm':'J','seed':1}]:
        with pytest.raises(ValueError):m.validate_config(saved,config)
    for arm,seed in [('W',1),('J',2)]:
        with pytest.raises(ValueError):m.run_config(arm,seed)

def test_invalid_fold_arm_and_safe_cli(tmp_path):
    m=p7();base=['--data','d','--weights','w','--out','o','--arm','J']
    assert m.parse_args(base).phase=='A'
    for extra in [['--phase','B'],['--resume','x'],['--seed','2'],['--arm','W']]:
        with pytest.raises(SystemExit):m.parse_args(base+extra)
    for fold,arm in [(2,'C'),(1,'W')]:
        with pytest.raises(ValueError):m.Fold(tmp_path,fold,arm)
