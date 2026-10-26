"""Independent numeric and provenance contracts for P16."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_proposal_p16 as p16
import train_proposal_p3 as p3


def test_squared_error_value_and_analytic_gradient():
    z=torch.tensor([[[-2.,0.,2.],[.7,-.6,1.1]]],requires_grad=True)
    y=torch.tensor([[[1.,.6,0.],[.2,1.,0.]]])
    loss=p16.loss_for_microbatch(z,y,3,'R')
    prob=1/(1+np.exp(-z.detach().numpy().astype(np.float64)))
    expected=((prob-y.numpy())**2).sum()/3
    assert float(loss)==pytest.approx(expected,rel=2e-7)
    loss.backward()
    np.testing.assert_allclose(z.grad.numpy(),2*(prob-y.numpy())*prob*(1-prob)/3,rtol=3e-6,atol=1e-8)


def test_halo_is_attracted_where_focal_only_suppresses_negatives():
    y=torch.tensor([[[.6]]])
    r=torch.tensor([[[-2.]]],requires_grad=True);c=r.detach().clone().requires_grad_()
    p16.loss_for_microbatch(r,y,1,'R').backward()
    p16.loss_for_microbatch(c,y,1,'C').backward()
    assert r.grad.item()<0<c.grad.item()


@pytest.mark.parametrize('count',[0,1,7,9])
@pytest.mark.parametrize('arm',['C','R'])
def test_effective_batch_accumulation_and_tail_gradient(count,arm):
    # Five heterogeneous images: an empty image, duplicate centers counted as objects,
    # and a one-image final microbatch. A changing normalization breaks equality.
    z=torch.linspace(-4,3,45).reshape(5,3,3).requires_grad_()
    y=torch.zeros_like(z);y[0,1,1]=1;y[1,1,1]=.6;y[3,0,0]=1
    full=p16.loss_for_microbatch(z,y,count,arm)
    grad=torch.autograd.grad(full,z)[0]
    parts=z.detach().clone().requires_grad_()
    split=sum(p16.loss_for_microbatch(parts[i:i+2],y[i:i+2],count,arm) for i in (0,2,4))
    split.backward()
    torch.testing.assert_close(split,full,rtol=2e-7,atol=1e-6)
    torch.testing.assert_close(parts.grad,grad,rtol=0,atol=0)
    if arm=='R':
        raw=((z.float().sigmoid()-y.float())**2).sum()
        torch.testing.assert_close(full,raw/max(1,count),rtol=0,atol=0)


def test_control_value_gradient_and_rng_exact():
    torch.manual_seed(19);state=torch.get_rng_state().clone()
    z=torch.linspace(-8,8,24).reshape(2,3,4).requires_grad_()
    y=torch.zeros_like(z);y[0,1,2]=1;y[1,1,2]=.6
    a=p16.loss_for_microbatch(z,y,4,'C');b=p3.focal_numerator(z,y)/4
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,z)[0],torch.autograd.grad(b,z)[0])
    assert torch.equal(state,torch.get_rng_state())


def test_regression_fp32_extreme_logits_empty_and_duplicate_denominator():
    z=torch.tensor([[[-1000.,1000.,0.]]],dtype=torch.bfloat16,requires_grad=True)
    y=torch.tensor([[[1.,0.,.5]]])
    loss=p16.loss_for_microbatch(z,y,0,'R')
    assert loss.dtype==torch.float32 and loss.item()==2
    loss.backward();assert torch.isfinite(z.grad).all()
    z=torch.zeros((1,1,1));y=torch.ones_like(z)
    assert p16.loss_for_microbatch(z,y,2,'R').item()==.125
    assert p16.loss_for_microbatch(z,y,0,'R').item()==.25


@pytest.mark.parametrize('count',[-1,True,1.5,np.bool_(False)])
def test_invalid_count_rejected(count):
    with pytest.raises(ValueError):p16.loss_for_microbatch(torch.zeros(1,2,2),torch.zeros(1,2,2),count,'R')


@pytest.mark.parametrize('value',[-.1,1.1,float('nan'),float('inf')])
def test_invalid_target_rejected(value):
    with pytest.raises(ValueError):p16.loss_for_microbatch(torch.zeros(1,2,2),torch.full((1,2,2),value),1,'R')


@pytest.mark.parametrize('arm',['C','R'])
def test_bound_config_and_schedule(arm):
    cfg=p16.run_config(arm,19)
    assert cfg['augmentation']=={'kind':'original'}
    assert cfg['target_sigma']==1 and cfg['target_radius']==3
    assert cfg['trainable_encoder_blocks']==list(range(12,24))
    assert cfg['objective']==({'kind':'original_center_focal','denominator':'effective_batch_objects'} if arm=='C' else {'kind':'gaussian_squared_sigmoid_error','coefficient':1,'reduction':'sum','dtype':'float32','denominator':'effective_batch_objects'})
    assert cfg['optimizer']['encoder_lr']==1e-5
    assert p16.encoder_lr(arm)==1e-5
    opt=torch.optim.AdamW([{'params':[torch.nn.Parameter(torch.ones(1))],'lr':3e-4,'initial_lr':3e-4},{'params':[torch.nn.Parameter(torch.ones(1))],'lr':1e-5,'initial_lr':1e-5}])
    p16.schedule_epoch(opt,129)
    assert [g['lr'] for g in opt.param_groups]==pytest.approx([3e-5,1e-6])
    p16.schedule_epoch(opt,0);assert [g['lr'] for g in opt.param_groups]==[3e-4,1e-5]
    p16.validate_config(cfg,copy.deepcopy(cfg))
    for key,value in [('arm','R' if arm=='C' else 'C'),('seed',1),('objective',{})]:
        wrong=copy.deepcopy(cfg);wrong[key]=value
        with pytest.raises(ValueError):p16.validate_config(wrong,cfg)


def test_invalid_metadata_and_phase():
    for arm in ('J','L','F0'):
        with pytest.raises(ValueError):p16.run_config(arm,19)
        with pytest.raises(ValueError):p16.loss_for_microbatch(torch.zeros(1,1,1),torch.zeros(1,1,1),1,arm)
    for seed in (True,2,19.):
        with pytest.raises(ValueError):p16.run_config('R',seed)
    with pytest.raises(ValueError):p16.Fold('unused',2,'R')
    with pytest.raises(ValueError):p16.Proposal('unused',False,'R')


def test_actual_fold_batch_original_rng_and_targets():
    base=object.__new__(p3.Fold)
    base.images=np.arange(2*256*256*3,dtype=np.uint8).reshape(2,256,256,3)
    base.gt=[[(4,(20,30),True,9),(4,(20,30),True,4)],[]]
    for arm in ('C','R'):
        fold=object.__new__(p16.Fold);fold.__dict__.update(base.__dict__)
        for aug in (False,True):
            np.random.seed(19);torch.manual_seed(19)
            a=base.batch([0,1],19,8,aug);state=torch.get_rng_state().clone();rng=np.random.get_state()
            b=fold.batch([0,1],19,8,aug)
            assert a[2]==b[2]==2
            assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
            assert torch.equal(state,torch.get_rng_state())
            now=np.random.get_state();assert now[0]==rng[0] and np.array_equal(now[1],rng[1]) and now[2:]==rng[2:]
