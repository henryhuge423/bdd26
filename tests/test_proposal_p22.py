"""Behavior checks for fixed full-depth UNI adaptation (P22)."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def module():
    assert importlib.util.find_spec('train_proposal_p22') is not None,'P22 runner is missing'
    import train_proposal_p22
    return train_proposal_p22

class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed=nn.Linear(4,4);self.pos_embed=nn.Parameter(torch.zeros(1,4));self.cls_token=nn.Parameter(torch.zeros(1,4))
        self.norm=nn.LayerNorm(4);self.blocks=nn.ModuleList([nn.Linear(4,4) for _ in range(24)])

def trainable_names(m):
    return sorted(n for n,p in m.named_parameters() if p.requires_grad)

@pytest.mark.parametrize('arm,blocks',[('C',range(12,24)),('M',range(24))])
def test_depth_configuration_trains_exact_blocks_and_freezes_embeddings(arm,blocks):
    p=module();enc=FakeEncoder();enc.train();p.configure_encoder_depth(enc,arm)
    expected=sorted(f'blocks.{b}.{k}' for b in blocks for k in ('weight','bias'))
    assert trainable_names(enc)==expected and not enc.training
    assert not any(q.requires_grad for n,q in enc.named_parameters() if not n.startswith('blocks.'))

def test_control_depth_matches_original_configure_encoder():
    p=module();from train_proposal_p3 import configure_encoder
    a=FakeEncoder();b=FakeEncoder();configure_encoder(a,True);p.configure_encoder_depth(b,'C')
    assert {n:q.requires_grad for n,q in a.named_parameters()}=={n:q.requires_grad for n,q in b.named_parameters()}

@pytest.mark.parametrize('arm',['A','F','',None,'c'])
def test_depth_configuration_rejects_unknown_arms(arm):
    p=module()
    with pytest.raises(ValueError):p.configure_encoder_depth(FakeEncoder(),arm)

@pytest.mark.parametrize('arm',['C','M'])
def test_loss_is_exactly_the_original_focal_over_effective_objects(arm):
    p=module();from proposal_p3_utils import focal_numerator
    x=torch.randn(6,5,5);t=torch.zeros_like(x);t[0,2,2]=t[3,1,4]=1
    assert torch.equal(p.loss_for_microbatch(x,t,9,arm),focal_numerator(x,t)/9)
    assert torch.equal(p.loss_for_microbatch(x,t,0,arm),focal_numerator(x,t))
    whole=p.loss_for_microbatch(x,t,9,arm);split=sum(p.loss_for_microbatch(a,b,9,arm) for a,b in zip(x.split(4),t.split(4)))
    torch.testing.assert_close(whole,split)

@pytest.mark.parametrize('arm,count,target',[('A',1,torch.zeros(1,2,2)),('C',-1,torch.zeros(1,2,2)),('M',1.5,torch.zeros(1,2,2)),('M',True,torch.zeros(1,2,2)),('C',1,torch.zeros(1,3,2)),('C',1,torch.zeros(1,1,2,2))])
def test_loss_rejects_invalid_metadata(arm,count,target):
    p=module()
    with pytest.raises(ValueError):p.loss_for_microbatch(torch.zeros(1,2,2),target,count,arm)

def test_config_differs_only_in_depth_and_rejects_cross_arm():
    p=module();c=p.run_config('C',19);m=p.run_config('M',19)
    assert c['trainable_encoder_blocks']==list(range(12,24)) and m['trainable_encoder_blocks']==list(range(24))
    assert c['architecture']=='original_UNI_last12' and m['architecture']=='UNI_full24'
    assert {k:v for k,v in c.items() if k not in ('arm','trainable_encoder_blocks','architecture')}=={k:v for k,v in m.items() if k not in ('arm','trainable_encoder_blocks','architecture')}
    assert c['encoder_lr']==m['encoder_lr']==1e-5 and c['head_lr']==m['head_lr']==3e-4
    assert c['effective_batch']==16 and c['microbatch']==4 and c['input_size']==256 and c['token_grid']==16
    with pytest.raises(ValueError):p.validate_config(c,m)
    for bad in (('M',19,16),('U',19,4),('M',7,4),('M','19',4)):
        with pytest.raises(ValueError):p.run_config(*bad)

def test_budgets_are_per_arm_and_fixed():
    p=module();assert p.run_budget_hours('C')==12 and p.run_budget_hours('M')==16
    with pytest.raises(ValueError):p.run_budget_hours('A')

def test_smoke_uses_fit_only_distinct_dead_images_and_keeps_134_tail():
    p=module();gt=[[(1,(3,3),True,1)] for _ in range(180)]
    for i in range(140,156):gt[i]=[(4,(3,3),True,1)]
    ids=p.smoke_ids(list(range(180)),gt)
    assert ids==list(range(118))+list(range(140,156)) and len(set(ids))==134 and len(ids)%16==6
    with pytest.raises(ValueError):p.smoke_ids(list(range(120)),gt)
    import train_proposal_p21 as p21
    assert ids==p21.smoke_ids(list(range(180)),gt)

def test_adapted_state_round_trip_requires_exact_full_depth_set():
    p=module();from train_proposal_p3 import load_adapted
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=FakeEncoder();self.head=nn.Linear(4,1)
    for arm in ('C','M'):
        m=Tiny();p.configure_encoder_depth(m.encoder,arm)
        state={n:t for n,t in m.state_dict().items() if n in {k for k,q in m.named_parameters() if q.requires_grad}}
        fresh=Tiny();p.configure_encoder_depth(fresh.encoder,arm);load_adapted(fresh,state)
        other=Tiny();p.configure_encoder_depth(other.encoder,'M' if arm=='C' else 'C')
        with pytest.raises(ValueError):load_adapted(other,state)

def test_visual_helpers_are_the_p21_functions():
    p=module();import train_proposal_p21 as p21
    assert p.match_points is p21.match_points and p.select_examples is p21.select_examples and p.render_example is p21.render_example
