"""P4 new behavior tests; P3 scoring remains unchanged."""
import importlib
import math
import numpy as np
import pytest
import torch


def api():
    return importlib.import_module('train_proposal_p4')


@pytest.mark.parametrize('fill', [0, 1])
def test_bce_empty_and_full_images_are_finite(fill):
    m=api(); logits=torch.zeros(2,4,4,requires_grad=True)
    loss=m.balanced_foreground_bce(logits,torch.full_like(logits,fill))
    assert loss.shape==(2,)
    assert torch.allclose(loss,torch.full((2,),.5*math.log(2)))
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


def test_bce_balances_groups_not_pixels():
    m=api(); z=torch.tensor([[[2.,-1.],[-1.,-1.]]],requires_grad=True)
    y=torch.tensor([[[1.,0.],[0.,0.]]])
    expected=.5*(torch.nn.functional.softplus(-z[0,0,0])+torch.nn.functional.softplus(z[0,0,1]))
    assert torch.allclose(m.balanced_foreground_bce(z,y)[0],expected)


def test_partial_microbatch_auxiliary_accumulation_matches_full_batch():
    m=api(); torch.manual_seed(9)
    z=torch.randn(7,4,4,requires_grad=True); y=(torch.randn(7,4,4)>0).float()
    full=m.balanced_foreground_bce(z,y).mean()
    parts=sum(m.balanced_foreground_bce(z[s:s+4],y[s:s+4]).sum()/7 for s in (0,4))
    assert torch.allclose(full,parts)
    assert torch.allclose(torch.autograd.grad(full,z,retain_graph=True)[0],torch.autograd.grad(parts,z)[0])


def test_geometry_transforms_foreground_and_center_together():
    m=api(); fg=torch.zeros(8,8); fg[2:5,1:4]=1; center=torch.zeros(8,8); center[3,2]=1
    image=fg.repeat(3,1,1)
    for seed in range(12):
        im,ct,mask=m.augment_triplet(image,center,fg,np.random.default_rng(seed),photometric=False)
        assert torch.equal(im[0],mask)
        assert (mask[ct==1]==1).all() and ct.sum()==1


def test_head_center_initialization_matches_p3_and_auxiliary_does_not_change_forward():
    m=api(); from train_proposal_p3 import PointHead as Original
    torch.manual_seed(19); old=Original()
    torch.manual_seed(19); new=m.PointHead()
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    with torch.no_grad():
        old.out.weight.normal_()
        new.out.weight.copy_(old.out.weight)
    features=[torch.randn(1,1024,2,2) for _ in range(4)]
    assert torch.equal(old(features),new(features))
    aux=torch.nn.Conv2d(64,1,1)
    assert torch.equal(old(features),new(features))


def test_control_auxiliary_does_not_change_shared_gradients():
    m=api(); shared=torch.nn.Conv2d(3,64,1); aux=torch.nn.Conv2d(64,1,1)
    x=torch.randn(2,3,4,4); target=torch.ones(2,4,4)
    feature=shared(x); center=feature.square().mean()
    base=torch.autograd.grad(center,shared.weight,retain_graph=True)[0]
    # C excludes auxiliary loss; N supplies a real gradient through shared features.
    extra=m.balanced_foreground_bce(aux(feature)[:,0],target).mean()
    joint=torch.autograd.grad(center+extra,shared.weight)[0]
    assert not torch.allclose(base,joint)


def test_main_clip_excludes_auxiliary_parameters():
    m=api(); main=torch.nn.Linear(2,1); aux=torch.nn.Linear(2,1)
    for p in main.parameters(): p.grad=torch.ones_like(p)
    for p in aux.parameters(): p.grad=torch.ones_like(p)*1e5
    expected=torch.nn.utils.clip_grad_norm_(list(main.parameters()),1)
    for p in main.parameters(): p.grad=torch.ones_like(p)
    actual=m.clip_separate(list(main.parameters()),list(aux.parameters()))
    assert torch.allclose(actual,expected)
    assert sum(float(p.grad.square().sum()) for p in main.parameters())==pytest.approx(1,rel=1e-5)


def test_gate_is_original_p3_and_cannot_pass_missing_support():
    m=api(); from proposal_p3_utils import admission
    assert m.admission is admission
    assert not m.admission({})['pass']


def test_fold_batch_preserves_p3_images_centers_and_augmented_foreground():
    m=api(); from train_proposal_p3 import Fold as Original
    from proposal_p3_utils import gt_objects
    fold=m.Fold.__new__(m.Fold)
    fold.images=np.zeros((2,256,256,3),np.uint8)
    fold.inst=np.zeros((2,256,256),np.uint16)
    fold.inst[0,60:70,80:90]=1
    fold.images[0,60:70,80:90]=255
    fold.gt=[gt_objects(i,(i>0).astype(np.uint8)) for i in fold.inst]
    old=Original.__new__(Original);old.__dict__.update(fold.__dict__)
    x,y,count,fg=fold.batch([0,1],19,12,True)
    bx,by,bc=old.batch([0,1],19,12,True)
    assert torch.equal(x,bx) and torch.equal(y,by) and count==bc==1
    assert fg[0].sum()==100 and fg[1].sum()==0 and (fg[0][y[0]==1]==1).all()
