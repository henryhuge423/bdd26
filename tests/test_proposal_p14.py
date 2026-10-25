import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from torch import nn
import train_proposal_p3 as original

def p14():
    assert importlib.util.find_spec('train_proposal_p14') is not None, 'P14 implementation missing'
    import train_proposal_p14
    return train_proposal_p14

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(2,1);self.encoder=nn.Linear(2,2)
    def forward(self,x):return self.head(self.encoder(x))

@pytest.mark.parametrize('arm,expected',[('C',1e-5),('L',1e-6)])
def test_real_optimizer_rates_membership_and_cosine(arm,expected):
    m=Tiny();opt=p14().make_optimizer(m,arm)
    assert isinstance(opt,torch.optim.AdamW)
    assert [g['lr'] for g in opt.param_groups]==[3e-4,expected]
    assert [g['initial_lr'] for g in opt.param_groups]==[3e-4,expected]
    assert set(map(id,opt.param_groups[0]['params']))==set(map(id,m.head.parameters()))
    assert set(map(id,opt.param_groups[1]['params']))==set(map(id,m.encoder.parameters()))
    assert all(g['betas']==(.9,.999) and g['weight_decay']==.01 for g in opt.param_groups)
    p14().schedule_epoch(opt,129)
    assert [g['lr'] for g in opt.param_groups]==pytest.approx([3e-5,expected*.1])
    p14().schedule_epoch(opt,0)
    assert [g['lr'] for g in opt.param_groups]==[3e-4,expected]

def test_low_rate_changes_actual_encoder_step_and_keeps_head_step():
    torch.manual_seed(19);a=Tiny();b=copy.deepcopy(a);before=copy.deepcopy(a.state_dict())
    x=torch.tensor([[1.,2.],[-1.,3.]])
    for m,arm in [(a,'C'),(b,'L')]:
        opt=p14().make_optimizer(m,arm);m(x).square().mean().backward();opt.step()
    assert torch.equal(a.head.weight,b.head.weight)
    assert not torch.equal(a.encoder.weight,b.encoder.weight)
    c=(a.encoder.weight-before['encoder.weight']).abs().max()
    l=(b.encoder.weight-before['encoder.weight']).abs().max()
    assert c>0 and l>0 and .08<float(l/c)<.12

@pytest.mark.parametrize('arm',['C','L'])
def test_full_proposal_initialization_original_without_rng_or_dropout(monkeypatch,arm):
    from nucseg.cellvit import model as module
    class FakeEncoder(nn.Module):
        def __init__(self,**kw):super().__init__();self.blocks=nn.ModuleList([nn.Linear(1,1) for _ in range(24)])
        def load_state_dict(self,state,strict=True):return None
    monkeypatch.setattr(module,'UNIEncoder',FakeEncoder);monkeypatch.setattr(torch,'load',lambda *a,**kw:{})
    torch.manual_seed(19);a=original.Proposal('unused',True);rng=torch.get_rng_state().clone()
    torch.manual_seed(19);b=p14().Proposal('unused',True,arm)
    assert torch.equal(rng,torch.get_rng_state())
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert not any(isinstance(m,nn.Dropout2d) for m in b.modules())
    assert all(p.requires_grad==(i>=12) for i,block in enumerate(b.encoder.blocks) for p in block.parameters())
    b.train();assert not b.encoder.training and b.head.training
    opt=p14().make_optimizer(b,arm)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in b.parameters() if p.requires_grad}

@pytest.mark.parametrize('arm',['C','L'])
@pytest.mark.parametrize('count',[0,7])
def test_original_focal_loss_and_gradient_on_mixed_tail_and_empty(arm,count):
    x=torch.linspace(-20,20,120).reshape(6,4,5).requires_grad_();y=torch.zeros_like(x);y[0,1,2]=1;y[-1,0,0]=.5
    a=p14().loss_for_microbatch(x,y,count,arm);b=original.focal_numerator(x,y)/max(1,count)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0])
    assert torch.allclose(a,sum(p14().loss_for_microbatch(x[i:i+2],y[i:i+2],count,arm) for i in range(0,6,2)))

@pytest.mark.parametrize('arm,seed',[('S',19),('R',19),('C',True),('L',2)])
def test_invalid_configs(arm,seed):
    with pytest.raises(ValueError):p14().run_config(arm,seed)

def test_config_binds_optimizer_recipe_and_reload_rejects_mismatches():
    p=p14();c=p.run_config('C',19);l=p.run_config('L',19)
    assert c['optimizer']['encoder_lr']==1e-5 and l['optimizer']['encoder_lr']==1e-6
    assert c['optimizer']['head_lr']==l['optimizer']['head_lr']==3e-4
    assert c['trainable_encoder_blocks']==l['trainable_encoder_blocks']==list(range(12,24))
    assert 'sampling' not in c and 'decoder_dropout' not in c
    p.validate_config(c,copy.deepcopy(c))
    for q in [l,p.run_config('C',1),{**c,'optimizer':{**c['optimizer'],'encoder_lr':2e-5}}]:
        with pytest.raises(ValueError):p.validate_config(c,q)
    for argv in [['--phase','B'],['--arm','S'],['--seed','2']]:
        with pytest.raises(SystemExit):p.parse_args(['--data','x','--weights','x','--out','x','--arm','C']+argv)

@pytest.mark.parametrize('epoch',[-1,130,True,1.5])
def test_invalid_schedule_epoch(epoch):
    with pytest.raises(ValueError):p14().schedule_epoch(p14().make_optimizer(Tiny(),'C'),epoch)

def test_control_verifier_optimizer_matches_original_actual_state():
    import verify_p14_control as verifier
    torch.manual_seed(7);a=Tiny();b=copy.deepcopy(a)
    opt_a=verifier.reference_optimizer(a);opt_b=p14().make_optimizer(b,'C')
    x=torch.tensor([[1.,2.],[-1.,3.]])
    for _ in range(2):
        for model,opt in [(a,opt_a),(b,opt_b)]:
            opt.zero_grad(set_to_none=True);model(x).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
    assert verifier.optimizer_hash(opt_a.state_dict())==verifier.optimizer_hash(opt_b.state_dict())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
