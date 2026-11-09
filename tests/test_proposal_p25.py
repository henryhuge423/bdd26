import copy
import importlib.util
import math
import pytest
import torch
from torch import nn

def p25():
    assert importlib.util.find_spec('train_proposal_p25') is not None,'P25 implementation missing'
    import train_proposal_p25
    return train_proposal_p25

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([nn.Linear(2,2) for _ in range(24)]);self.norm=nn.LayerNorm(2);self.embed=nn.Linear(2,2)
class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(2,1);self.encoder=Encoder()

@pytest.mark.parametrize('arm',['C','D'])
def test_depth_freezes_embeddings_norm_and_covers_exact_optimizer_members(arm):
    p=p25();m=Tiny();p.configure_encoder_depth(m.encoder,arm);opt=p.make_optimizer(m,arm)
    assert not m.encoder.training
    assert all(not q.requires_grad for q in m.encoder.norm.parameters())
    assert all(not q.requires_grad for q in m.encoder.embed.parameters())
    assert all(q.requires_grad==(arm=='D' or b>=12) for b,block in enumerate(m.encoder.blocks) for q in block.parameters())
    ids=[id(q) for g in opt.param_groups for q in g['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(q) for q in m.parameters() if q.requires_grad}
    assert len(opt.param_groups)==(2 if arm=='C' else 25)

def test_all24_rates_independent_formula_membership_and_actual_updates():
    p=p25();torch.manual_seed(19);m=Tiny();p.configure_encoder_depth(m.encoder,'D');before=copy.deepcopy(m.state_dict());opt=p.make_optimizer(m,'D')
    expected=[3e-4]+[3e-5*math.pow(.95,23-b) for b in range(24)]
    assert [g['lr'] for g in opt.param_groups]==pytest.approx(expected,rel=1e-14)
    for b,g in enumerate(opt.param_groups[1:]):assert set(map(id,g['params']))==set(map(id,m.encoder.blocks[b].parameters()))
    for group in opt.param_groups:
        assert group['betas']==(.9,.999) and group['weight_decay']==.01
        for q in group['params']:q.grad=torch.ones_like(q)
    opt.step()
    for b,block in enumerate(m.encoder.blocks):
        lr=expected[b+1];old=before[f'encoder.blocks.{b}.weight'];actual=block.weight.detach()
        assert torch.allclose(actual,old*(1-lr*.01)-lr/(1+1e-8),atol=4e-8,rtol=0)
        assert not torch.equal(actual,old*(1-lr*.01))
    for n in ('encoder.norm.weight','encoder.norm.bias','encoder.embed.weight','encoder.embed.bias'):assert torch.equal(before[n],m.state_dict()[n])
    p.schedule_epoch(opt,129);assert [g['lr'] for g in opt.param_groups]==pytest.approx([v*.1 for v in expected],rel=1e-14)
    p.schedule_epoch(opt,0);assert [g['lr'] for g in opt.param_groups]==pytest.approx(expected,rel=1e-14)

def test_c_real_adamw_two_steps_state_and_parameters_equal_original():
    from verify_p24_control import reference_optimizer,optimizer_hash
    p=p25();torch.manual_seed(1);a=Tiny();p.configure_encoder_depth(a.encoder,'C');b=copy.deepcopy(a)
    oa=reference_optimizer(a);ob=p.make_optimizer(b,'C')
    for step in (1,2):
        for m,opt in ((a,oa),(b,ob)):
            opt.zero_grad(set_to_none=True)
            loss=sum(q.square().sum()*(i+1) for i,q in enumerate(m.parameters()) if q.requires_grad);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
    assert optimizer_hash(oa.state_dict())==optimizer_hash(ob.state_dict())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())

@pytest.mark.parametrize('arm',['C','D'])
@pytest.mark.parametrize('count',[0,7])
def test_original_focal_gradients_and_split_tail(arm,count):
    import train_proposal_p3 as original
    p=p25();x=torch.linspace(-20,20,120).reshape(6,4,5).requires_grad_();y=torch.zeros_like(x);y[0,1,2]=1;y[-1,0,0]=.5
    a=p.loss_for_microbatch(x,y,count,arm);b=original.focal_numerator(x,y)/max(1,count)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x,retain_graph=True)[0])
    assert torch.allclose(a,sum(p.loss_for_microbatch(x[i:i+2],y[i:i+2],count,arm) for i in range(0,6,2)))

@pytest.mark.parametrize('arm',['C','D'])
def test_full_model_initialization_rng_and_train_mode_match_original(monkeypatch,arm):
    import train_proposal_p3 as original
    from nucseg.cellvit import model as module
    class Fake(Encoder):
        def __init__(self,**kw):super().__init__()
        def load_state_dict(self,state,strict=True):return None
    monkeypatch.setattr(module,'UNIEncoder',Fake);monkeypatch.setattr(torch,'load',lambda *a,**kw:{})
    torch.manual_seed(19);a=original.Proposal('unused',True);rng=torch.get_rng_state().clone()
    torch.manual_seed(19);b=p25().Proposal('unused',True,arm)
    assert torch.equal(rng,torch.get_rng_state()) and a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    b.train();assert not b.encoder.training and b.head.training

def test_config_binds_every_rate_and_refuses_cross_recipe_reload():
    p=p25();c=p.run_config('C',19);d=p.run_config('D',19)
    assert c['trainable_encoder_blocks']==list(range(12,24)) and d['trainable_encoder_blocks']==list(range(24))
    assert d['block_initial_lrs']==pytest.approx([3e-5*.95**(23-b) for b in range(24)])
    assert d['layer_decay']==.95 and d['head_lr']==3e-4 and c['encoder_lr']==1e-5
    p.validate_config(d,copy.deepcopy(d))
    wrong=copy.deepcopy(d);wrong['block_initial_lrs'][0]*=2
    for other in [c,wrong,{**d,'layer_decay':.9},p.run_config('D',1)]:
        with pytest.raises(ValueError):p.validate_config(d,other)
    assert p.run_budget_hours('C')==p.run_budget_hours('D')==12

@pytest.mark.parametrize('arm,seed',[('L',19),('M',19),('D',True),('D',2)])
def test_invalid_config(arm,seed):
    with pytest.raises(ValueError):p25().run_config(arm,seed)

@pytest.mark.parametrize('epoch',[-1,130,True,1.5])
def test_invalid_schedule(epoch):
    p=p25();m=Tiny();p.configure_encoder_depth(m.encoder,'C')
    with pytest.raises(ValueError):p.schedule_epoch(p.make_optimizer(m,'C'),epoch)

def test_learning_evidence_requires_early_middle_late_and_beyond_decay():
    assert importlib.util.find_spec('verify_p25_control') is not None,'P25 witness missing'
    from verify_p25_control import validate_learning
    good={k:.1 for k in ('block0','block11','block12','block23','output')}
    assert validate_learning(good,good,good)
    for k in good:
        for value in (0.,float('nan'),float('inf')):
            with pytest.raises(ValueError):validate_learning(good,good,{**good,k:value})
