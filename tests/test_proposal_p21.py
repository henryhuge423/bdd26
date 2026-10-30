"""Behavior checks for fixed semantic auxiliary supervision and point visual diagnostics."""
import importlib.util
import sys
from pathlib import Path
import math
import numpy as np
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def module():
    assert importlib.util.find_spec('train_proposal_p21') is not None,'P21 runner is missing'
    import train_proposal_p21
    return train_proposal_p21

def test_typed_centers_keep_class_order_collisions_and_border():
    p=module();gt=[(1,(0,0),False,1),(4,(12,20),True,12),(5,(12,20),True,8),(4,(12,20),True,4)]
    t=p.typed_targets(gt)
    assert t.shape==(6,256,256) and t.dtype==np.float32
    assert t[0,0,0]==t[1,0,0]==1
    assert t[0,12,20]==t[4,12,20]==t[5,12,20]==1
    assert np.count_nonzero(t[2])==np.count_nonzero(t[3])==0
    assert t[4,12,21]==pytest.approx(math.exp(-.5))
    assert t[4,12,24]==0 and t[4].max()==1

@pytest.mark.parametrize('gt',[[],[(2,(30,40),True,9)]])
def test_typed_targets_preserve_original_untyped_map(gt):
    p=module();from proposal_p3_utils import gaussian_target
    t=p.typed_targets(gt);np.testing.assert_array_equal(t[0],gaussian_target(gt))
    if not gt:assert not t.any()

@pytest.mark.parametrize('seed',[19,1])
def test_stacked_augmentation_preserves_original_image_target_rng(seed):
    p=module();from proposal_p3_utils import augment_pair
    gt=[(4,(21,65),True,15),(1,(0,255),False,3)]
    x=torch.linspace(0,1,3*256*256).reshape(3,256,256);t=torch.from_numpy(p.typed_targets(gt))
    a,b=augment_pair(x,t,np.random.default_rng(seed));c,d=augment_pair(x,t[0],np.random.default_rng(seed))
    assert torch.equal(a,c) and torch.equal(b[0],d)
    assert int((b[4]==1).sum())==1

def test_auxiliary_gradient_includes_absent_and_empty_negative_maps():
    p=module();x=torch.tensor([[[0.]]],requires_grad=True);a=torch.zeros(1,5,1,1,requires_grad=True)
    t=torch.zeros(1,6,1,1);t[0,0,0,0]=t[0,4,0,0]=1
    loss=p.loss_for_microbatch(x,t,1,'A',a)
    assert float(loss)==pytest.approx(6*math.log(2)/4)
    gx,ga=torch.autograd.grad(loss,(x,a));v=-(.125+.25*math.log(2))
    assert float(gx)==pytest.approx(v)
    torch.testing.assert_close(ga.flatten(),torch.tensor([-v,-v,-v,v,-v]))
    empty=p.loss_for_microbatch(torch.zeros_like(x),torch.zeros_like(t),0,'A',torch.zeros_like(a))
    assert float(empty)==pytest.approx(6*math.log(2)/4)

def test_control_loss_exact_and_tail_uses_effective_object_count():
    p=module();from proposal_p3_utils import focal_numerator
    x=torch.randn(6,3,3);t=torch.zeros_like(x);t[0,1,1]=1
    assert torch.equal(p.loss_for_microbatch(x,t,7,'C'),focal_numerator(x,t)/7)
    whole=p.loss_for_microbatch(x,t,7,'C')
    split=sum(p.loss_for_microbatch(xx,tt,7,'C') for xx,tt in zip(x.split(4),t.split(4)))
    torch.testing.assert_close(whole,split)

@pytest.mark.parametrize('arm,count,target,aux',[('C',-1,torch.zeros(1,2,2),None),('A',1,torch.zeros(1,2,2),None),('C',1.5,torch.zeros(1,2,2),None),('A',True,torch.zeros(1,6,2,2),torch.zeros(1,5,2,2)),('Q',1,torch.zeros(1,2,2),None)])
def test_loss_rejects_invalid_metadata(arm,count,target,aux):
    p=module()
    with pytest.raises(ValueError):p.loss_for_microbatch(torch.zeros(1,2,2),target,count,arm,aux)

def test_auxiliary_head_preserves_main_initialization_rng_and_inference():
    p=module();from train_proposal_p3 import PointHead
    torch.manual_seed(19);ref=PointHead();rng=torch.get_rng_state()
    torch.manual_seed(19);h=p.SemanticPointHead();assert torch.equal(torch.get_rng_state(),rng)
    for k,v in ref.state_dict().items():assert torch.equal(v,h.state_dict()[k]),k
    feats=[torch.randn(1,1024,16,16) for _ in range(4)]
    with torch.no_grad():
        wanted=ref(feats);main,aux=h.forward_with_aux(feats)
        assert torch.equal(main,wanted) and aux.shape==(1,5,256,256)
        h.auxiliary.weight.normal_();h.auxiliary.bias.normal_()
        def forbidden(*args):raise AssertionError('auxiliary inference executed')
        h.auxiliary.register_forward_hook(forbidden)
        assert torch.equal(h(feats),wanted)

def test_auxiliary_loss_reaches_shared_features_after_zero_head_learns():
    p=module();h=p.SemanticPointHead();features=torch.randn(2,64,3,3,requires_grad=True)
    y=torch.zeros(2,5,3,3);y[0,3,1,1]=1
    from proposal_p3_utils import focal_numerator
    opt=torch.optim.AdamW(h.auxiliary.parameters(),lr=3e-4)
    first=focal_numerator(h.auxiliary(features),y);first.backward();assert not features.grad.any()
    opt.step();opt.zero_grad();features.grad=None
    focal_numerator(h.auxiliary(features),y).backward()
    assert features.grad.norm()>0

def test_smoke_uses_fit_only_distinct_dead_images_and_keeps_134_tail():
    p=module();gt=[[(1,(3,3),True,1)] for _ in range(180)]
    for i in range(140,156):gt[i]=[(4,(3,3),True,1)]
    ids=p.smoke_ids(list(range(180)),gt)
    assert ids==list(range(118))+list(range(140,156))
    assert len(set(ids))==134 and len(ids)%16==6
    with pytest.raises(ValueError):p.smoke_ids(list(range(120)),gt)

def test_config_rejects_cross_arm_and_wrong_batch():
    p=module();c=p.run_config('C',19);a=p.run_config('A',19)
    with pytest.raises(ValueError):p.validate_config(c,a)
    with pytest.raises(ValueError):p.run_config('A',19,16)
    with pytest.raises(ValueError):p.run_config('U',19)

def test_visual_unique_matching_and_gt_only_selection(tmp_path):
    p=module();gt=[(4,(10,10),True,9),(1,(20,20),True,9)]
    pairs=p.match_points([(10,10),(10,11),(250,250)],gt)
    assert pairs==[(0,0)]
    g=[gt,[],[(1,(3,3),True,1)],gt,[],gt,gt,[(1,(4,4),True,1)]]
    assert p.select_examples(list(range(8)),g,8)==[0,3,5,6,1,4,2,7]
    out=tmp_path/'example.png';p.render_example(np.full((256,256,3),240,np.uint8),gt,[(10,10),(250,250)],out,'test')
    from PIL import Image
    image=Image.open(out);assert image.width>=1024 and image.height>=256
    assert out.stat().st_size>1000
