import copy
import numpy as np
import pytest
import torch
import train_proposal_p15 as p
from train_proposal_p3 import Fold as OriginalFold
from proposal_p3_utils import gaussian_target

def objects():
    return [(4,(0,0),True,10),(1,(255,255),False,20),(2,(100,120),True,30),(3,(100,120),True,40)]

@pytest.mark.parametrize('dx,dy',[(16,-16),(-16,16),(0,0),(1,2),(-1,-2)])
def test_integer_geometry_and_denominator(dx,dy):
    image=torch.arange(3*256*256).reshape(3,256,256).float()/(3*256*256)
    x,y,n=p.translate_sample(image,objects(),dx,dy)
    kept=[(g[0],(g[1][0]+dy,g[1][1]+dx),*g[2:]) for g in objects() if 0<=g[1][0]+dy<256 and 0<=g[1][1]+dx<256]
    assert n==len(kept)
    assert torch.equal(y,torch.from_numpy(gaussian_target(kept)))
    assert torch.equal(x[:,120+dy,100+dx],image[:,120,100])
    if dx>0:assert (x[:,:,:dx]==1).all()
    if dx<0:assert (x[:,:,dx:]==1).all()
    if dy>0:assert (x[:,:dy,:]==1).all()
    if dy<0:assert (x[:,dy:,:]==1).all()
    assert torch.equal(image,torch.arange(3*256*256).reshape(3,256,256).float()/(3*256*256))

def test_dropped_center_has_no_halo_and_duplicates_count():
    x,y,n=p.translate_sample(torch.zeros(3,256,256),[(4,(0,0),True,1)],-1,0)
    assert n==0 and not y.any() and not x[:,:,:255].any() and (x[:,:,-1]==1).all()
    _,y,n=p.translate_sample(torch.zeros(3,256,256),objects()[2:],0,0)
    assert n==2 and int((y==1).sum())==1

def test_empty_and_invalid():
    _,y,n=p.translate_sample(torch.zeros(3,256,256),[],16,16)
    assert n==0 and not y.any()
    for dx in [17,-17,True,1.5]:
        with pytest.raises(ValueError):p.translate_sample(torch.zeros(3,256,256),[],dx,0)

def fake_fold(arm):
    f=p.Fold.__new__(p.Fold);f.arm=arm
    f.images=np.random.default_rng(1).integers(0,256,(3,256,256,3),dtype=np.uint8)
    f.gt=[objects(),[],[(4,(15,15),True,10)]]
    return f

@pytest.mark.parametrize('seed',[19,1])
def test_original_control_and_evaluation_parity(seed):
    f=fake_fold('C');ref=OriginalFold.__new__(OriginalFold);ref.images=f.images;ref.gt=f.gt
    for arm,aug in [('C',True),('C',False),('J',False)]:
        f.arm=arm;a=f.batch([0,1,2],seed,7,aug);b=ref.batch([0,1,2],seed,7,aug)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
        assert f.batch_count([0,1,2],seed,7,aug)==b[2]

def test_treatment_batch_count_and_order():
    f=fake_fold('J');seed=19;epoch=3;ids=[0,1,2]
    x,y,n=f.batch(ids,seed,epoch,True);images=[];targets=[];count=0
    from proposal_p3_utils import augment_pair
    for i in ids:
        dx,dy=p.translation_offset(seed,epoch,i)
        im=torch.from_numpy(f.images[i].copy()).permute(2,0,1).float()/255
        im,t,c=p.translate_sample(im,f.gt[i],dx,dy)
        im,t=augment_pair(im,t,np.random.default_rng(np.random.SeedSequence([seed,epoch,i,987])))
        images.append(im);targets.append(t);count+=c
    assert torch.equal(x,torch.stack(images)) and torch.equal(y,torch.stack(targets))
    assert n==count==f.batch_count(ids,seed,epoch,True)
    assert sum(f.batch_count(ids[i:i+1],seed,epoch,True) for i in range(3))==count

def test_local_rng_and_inclusive_offsets():
    np.random.seed(42);torch.manual_seed(42);a=np.random.get_state();b=torch.get_rng_state()
    offsets=[p.translation_offset(19,0,i) for i in range(1000)]
    assert offsets==[p.translation_offset(19,0,i) for i in range(1000)]
    assert set(x for pair in offsets for x in pair)==set(range(-16,17))
    assert np.array_equal(a[1],np.random.get_state()[1]) and torch.equal(b,torch.get_rng_state())
    assert offsets!=[p.translation_offset(19,1,i) for i in range(1000)]

def test_config_and_schedule():
    c=p.run_config('C',19);j=p.run_config('J',19)
    assert c['optimizer']==j['optimizer'] and c['optimizer']['encoder_lr']==1e-5
    assert j['augmentation']['translation_range']==16
    with pytest.raises(ValueError):p.validate_config(c,j)
    with pytest.raises(ValueError):p.run_config('L',19)
    with pytest.raises(ValueError):p.run_config('J',2)

def test_epoch_ledger_replays_counts():
    f=fake_fold('J');d=f.translation_record([0,1,2],19,3)
    assert d['epoch_zero_based']==3 and d['images']==3
    assert d['retained_objects']==f.batch_count([0,1,2],19,3,True)
    assert d['retained_objects']+d['dropped_objects']==5
    assert sum(d['offset_histogram'].values())==3
