"""P6 target bandwidth must preserve the original control and evaluation."""
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
torch.set_num_threads(2)

def p6():
    assert importlib.util.find_spec('train_proposal_p6') is not None, 'P6 target-bandwidth trainer is missing'
    return importlib.import_module('train_proposal_p6')

def test_control_target_is_bit_exact_original():
    m=p6();from proposal_p3_utils import gaussian_target
    gt=[(4,(0,0),False,8),(1,(128,128),True,40),(3,(255,255),False,10)]
    assert np.array_equal(m.make_target(gt,'C'),gaussian_target(gt))

def test_wide_target_analytical_values_and_support():
    target=p6().make_target([(4,(10,10),True,20)],'W')
    assert target.dtype==np.float32
    assert target[10,10]==1
    assert target[10,11]==pytest.approx(.8824969,abs=1e-7)
    assert target[10,12]==pytest.approx(.60653066,abs=1e-7)
    assert target[10,16]==pytest.approx(.011108996,abs=1e-8)
    assert target[10,17]==0
    assert np.count_nonzero(target==1)==1
    assert np.count_nonzero(target)==169

def test_wide_target_edge_crop_overlap_and_empty():
    m=p6();gt=[(4,(0,0),False,8),(1,(0,2),False,8)]
    a=m.make_target(gt,'W')
    assert np.array_equal(a,m.make_target(list(reversed(gt)),'W'))
    assert a[0,0]==a[0,2]==1
    assert a[0,1]==pytest.approx(.8824969,abs=1e-7)
    assert a[-1,-1]==0 and a.max()==1
    assert not m.make_target([],'W').any()

def test_bandwidth_only_reduces_halo_negative_loss():
    m=p6();gt=[(4,(20,20),True,50)]
    c=torch.from_numpy(m.make_target(gt,'C'))[None]
    w=torch.from_numpy(m.make_target(gt,'W'))[None]
    assert torch.equal(c==1,w==1)
    logits=torch.full_like(c,-2.,requires_grad=True)
    lc=m.focal_numerator(logits,c);lw=m.focal_numerator(logits,w)
    assert lw<lc
    gc=torch.autograd.grad(lc,logits,retain_graph=True)[0]
    gw=torch.autograd.grad(lw,logits)[0]
    assert gc[0,20,20]==gw[0,20,20]<0
    assert 0<gw[0,20,21]<gc[0,20,21]
    assert gw[0,100,100]==gc[0,100,100]

def test_real_batches_preserve_images_counts_augmentation_and_rng(tmp_path):
    m=p6();import train_proposal_p3 as old
    root=tmp_path/'fold1';root.mkdir()
    images=np.random.default_rng(11).integers(0,256,(2,256,256,3),dtype=np.uint8)
    inst=np.zeros((2,256,256),np.uint16);typ=np.zeros_like(inst,dtype=np.uint8)
    inst[0,15:20,22:26]=1;typ[0,15:20,22:26]=4
    for name,value in [('images',images),('inst',inst),('type',typ),('tissue',np.array(['Lung','Skin']))]:np.save(root/(name+'.npy'),value)
    c=m.Fold(tmp_path,1,'C');w=m.Fold(tmp_path,1,'W');reference=old.Fold(tmp_path,1)
    for ids in ([0,1],[1]):
        a=c.batch(ids,19,3,True);b=w.batch(ids,19,3,True);r=reference.batch(ids,19,3,True)
        assert torch.equal(a[0],b[0]) and torch.equal(a[0],r[0])
        assert torch.equal(a[1],r[1])
        assert a[2]==b[2]==r[2]==(1 if 0 in ids else 0)
        assert torch.equal(a[1]==1,b[1]==1)
    with pytest.raises(ValueError):m.Fold(tmp_path,2,'C')
    with pytest.raises(ValueError):m.Fold(tmp_path,1,'bad')

def test_original_model_scoring_cohort_and_focal_are_reused():
    m=p6();import train_proposal_p3 as old;import proposal_p3_utils as utils
    assert m.Proposal is old.Proposal
    for name in ('split_indices','gt_objects','gaussian_target','augment_pair','focal_numerator','decode_peaks','score_points','aggregate','admission'):
        assert getattr(m,name) is getattr(utils,name)

def test_effective_batch_gradient_including_empty_tail_is_unchanged():
    m=p6()
    for arm in ('C','W'):
        targets=torch.from_numpy(np.stack([m.make_target([(4,(15,15),True,20)],arm),m.make_target([],arm),m.make_target([(1,(20,20),True,50),(4,(30,30),True,20)],arm)]))
        full=torch.full_like(targets,-1.5,requires_grad=True)
        loss=m.focal_numerator(full,targets)/3
        expected=torch.autograd.grad(loss,full)[0]
        pieces=torch.full_like(targets,-1.5,requires_grad=True)
        for a,b in ((0,2),(2,3)):(m.focal_numerator(pieces[a:b],targets[a:b])/3).backward()
        assert torch.equal(expected,pieces.grad)

def test_cli_rejects_phaseB_unknown_arm_and_resume():
    m=p6();base=['--data','d','--weights','w','--out','o']
    assert m.parse_args(base+['--arm','W']).phase=='A'
    for extra in (['--phase','B'],['--arm','N'],['--resume']):
        with pytest.raises(SystemExit):m.parse_args(base+extra)

@pytest.mark.parametrize('saved',[{'phase':'A','arm':'C','seed':19,'target_sigma':2},{'phase':'A','arm':'W','seed':19,'target_sigma':2}])
def test_checkpoint_target_or_arm_mismatch_is_rejected(saved):
    m=p6()
    with pytest.raises(ValueError):m.validate_config(saved,m.run_config('C',19))

def test_checkpoint_exact_config_accepted():
    m=p6();c=m.run_config('W',1)
    assert c=={'phase':'A','arm':'W','seed':1,'target_sigma':2,'target_radius':6}
    m.validate_config(c,c)
