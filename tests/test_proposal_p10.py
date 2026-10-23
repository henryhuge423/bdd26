import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import torch
from torch import nn
import train_proposal_p3 as original


def module():
    import train_proposal_p10
    return train_proposal_p10


class Small(nn.Module):
    def __init__(self):
        super().__init__();self.prefix=nn.Linear(2,2)
        self.blocks=nn.ModuleList([nn.Linear(2,2) for _ in range(24)])
        self.norm=nn.LayerNorm(2)
    def forward(self,x):
        x=self.prefix(x)
        for b in self.blocks:x=x+torch.tanh(b(x))*.1
        return self.norm(x)


@pytest.mark.parametrize('arm,start',[('C',12),('S',20)])
def test_scope_exact_keys_and_frozen_updates(arm,start):
    m=module();torch.manual_seed(1);enc=Small();enc.train();m.configure_scope(enc,arm)
    expected={f'blocks.{i}.{n}' for i in range(start,24) for n in ('weight','bias')}
    assert {n for n,p in enc.named_parameters() if p.requires_grad}==expected
    assert not enc.training and all(not b.training for b in enc.blocks)
    before={n:p.clone().detach() for n,p in enc.named_parameters()}
    opt=torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad],lr=.01,weight_decay=.01)
    enc(torch.tensor([[.3,-.8],[.1,.5]])).square().sum().backward()
    assert all(p.grad is None for n,p in enc.named_parameters() if n not in expected)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in enc.named_parameters() if n in expected)
    opt.step()
    assert all(torch.equal(p,before[n]) for n,p in enc.named_parameters() if n not in expected)
    assert any(not torch.equal(p,before[n]) for n,p in enc.named_parameters() if n in expected)


def test_scope_control_matches_original_and_does_not_consume_rng():
    m=module();torch.manual_seed(19);a=Small();b=Small();b.load_state_dict(a.state_dict())
    rng=torch.get_rng_state();m.configure_scope(a,'C');assert torch.equal(rng,torch.get_rng_state())
    original.configure_encoder(b,True)
    assert [p.requires_grad for p in a.parameters()]==[p.requires_grad for p in b.parameters()]
    assert all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items())


@pytest.mark.parametrize('bad',['D','I','',None])
def test_reject_invalid_scope(bad):
    with pytest.raises(ValueError):module().configure_scope(Small(),bad)


@pytest.mark.parametrize('arm',['C','S'])
@pytest.mark.parametrize('seed',[19,1])
def test_config_and_cross_arm_rejection(arm,seed):
    m=module();c=m.run_config(arm,seed)
    assert c['trainable_encoder_blocks']==list(range(12 if arm=='C' else 20,24))
    m.validate_config(c,m.run_config(arm,seed))
    with pytest.raises(ValueError):m.validate_config(c,m.run_config('S' if arm=='C' else 'C',seed))
    assert m.parse_args(['--data','d','--weights','w','--out','o','--arm',arm,'--seed',str(seed)]).arm==arm


def test_invalid_configs_and_cli():
    m=module()
    for arm,seed in [('D',19),('C',2),('S',False)]:
        with pytest.raises(ValueError):m.run_config(arm,seed)
    with pytest.raises(SystemExit):m.parse_args(['--data','d','--weights','w','--out','o','--arm','S','--phase','B'])


@pytest.mark.parametrize('arm',['C','S'])
@pytest.mark.parametrize('total',[0,7,29])
def test_original_loss_value_gradient_and_micro_tail(arm,total):
    m=module();torch.manual_seed(3);x=torch.randn(5,4,4,requires_grad=True);y=torch.rand_like(x);y[:,1,1]=1
    expected=original.focal_numerator(x,y)/max(1,total)
    actual=sum(m.loss_for_microbatch(x[i:i+2],y[i:i+2],total,arm) for i in range(0,5,2))
    assert torch.allclose(actual,expected)
    assert torch.allclose(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])


@pytest.mark.parametrize('arm',['C','S'])
def test_fold_augmented_batches_equal_original(arm):
    m=module();fold=m.Fold.__new__(m.Fold);fold.images=np.arange(2*256*256*3,dtype=np.uint8).reshape(2,256,256,3)
    fold.gt=[[(4,(7,8),True,32)],[]]
    for aug in (False,True):
        x,y,n=fold.batch([0,1],19,3,aug)
        a,b,c=original.Fold.batch(fold,[0,1],19,3,aug)
        assert torch.equal(x,a) and torch.equal(y,b) and n==c


def test_reload_uses_own_keyset_and_rejects_cross_arm():
    m=module()
    class Mini(nn.Module):
        adapted_state=original.Proposal.adapted_state
        def __init__(self,arm):
            super().__init__();self.encoder=Small();self.head=nn.Linear(2,1);m.configure_scope(self.encoder,arm)
    a,b=Mini('C'),Mini('S')
    original.load_adapted(a,a.adapted_state());original.load_adapted(b,b.adapted_state())
    with pytest.raises(ValueError):original.load_adapted(b,a.adapted_state())
    with pytest.raises(ValueError):original.load_adapted(a,b.adapted_state())
