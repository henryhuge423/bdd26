import copy
import pytest
import torch
from torch import nn
import train_proposal_p3 as original
import train_proposal_p12 as p12


def heads(arm):
    torch.manual_seed(13)
    a=original.PointHead()
    torch.manual_seed(13)
    b=p12.DropoutPointHead(arm)
    # Constant initialized output obscures feature dropout; use nonzero weights.
    with torch.no_grad():a.out.weight.fill_(.01);b.out.weight.copy_(a.out.weight)
    return a,b


@pytest.mark.parametrize('arm,training',[('C',True),('C',False),('R',False)])
def test_original_output_gradient_and_rng_parity(arm,training):
    a,b=heads(arm);a.train(training);b.train(training)
    f=[torch.randn(1,1024,2,2,requires_grad=True) for _ in range(4)]
    g=[x.detach().clone().requires_grad_() for x in f]
    before=torch.get_rng_state().clone()
    x=a(f);x.square().mean().backward();after=torch.get_rng_state().clone()
    torch.set_rng_state(before);y=b(g);y.square().mean().backward()
    assert torch.equal(after,torch.get_rng_state())
    assert torch.equal(x,y)
    assert a.state_dict().keys()==b.state_dict().keys()
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p.grad,q.grad)
    for p,q in zip(f,g):assert torch.equal(p.grad,q.grad)


def test_channel_mask_scaling_independence_and_expectation():
    head=p12.DropoutPointHead('R').train()
    x=torch.ones(128,256,3,3,requires_grad=True)
    torch.manual_seed(77);y=head.dropout(x)
    assert set(y.detach().unique().tolist())=={0.,1.25}
    assert torch.equal(y,y[:,:,:1,:1].expand_as(y))
    assert not torch.equal(y[0],y[1])
    assert abs(float((y==0).float().mean())-.2)<.015
    assert abs(float(y.mean())-1)<.02
    y.sum().backward();assert torch.equal(x.grad,y.detach())
    torch.manual_seed(77);assert torch.equal(y,head.dropout(x))
    assert not torch.equal(y,head.dropout(x))


def test_dropout_location_and_stochastic_training_then_eval():
    a,b=heads('R');b.train();captured=[]
    h=b.dropout.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
    f=[torch.rand(2,1024,2,2) for _ in range(4)]
    x=b(f);y=b(f);h.remove()
    assert len(captured)==2 and captured[0].shape==(2,256,2,2)
    # Exact dropout input is the independently evaluated original stem activation.
    expected=torch.relu(a.stem(torch.cat([p(t) for p,t in zip(a.proj,f)],1)))
    assert torch.equal(captured[0],expected) and torch.equal(captured[0],captured[1])
    assert not torch.equal(x,y)
    b.eval();a.eval();assert torch.equal(a(f),b(f))


def test_full_proposal_registration_and_initial_state(monkeypatch):
    from nucseg.cellvit import model as module
    class FakeEncoder(nn.Module):
        def __init__(self,**kwargs):super().__init__();self.blocks=nn.ModuleList([nn.Linear(1,1) for _ in range(24)])
        def load_state_dict(self,state,strict=True):return None
    monkeypatch.setattr(module,'UNIEncoder',FakeEncoder)
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:{})
    models=[]
    for arm in ('C','R'):
        torch.manual_seed(9);models.append(p12.Proposal('unused',True,arm))
    a,b=models
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    b.train();assert b.head.dropout.training and not b.encoder.training
    b.eval();assert not b.head.dropout.training
    assert b.head.dropout.p==.2 and a.head.dropout.p==0
    assert all(p.requires_grad==(i>=12) for i,block in enumerate(b.encoder.blocks) for p in block.parameters())


@pytest.mark.parametrize('arm',['C','R'])
def test_loss_is_original_for_empty_mixed_tail_batches(arm):
    x=torch.linspace(-80,80,60).reshape(3,4,5).requires_grad_();y=torch.zeros_like(x);y[0,1,2]=1;y[2,0,0]=.5
    a=p12.loss_for_microbatch(x,y,7,arm);g=torch.autograd.grad(a,x)[0]
    z=x.detach().clone().requires_grad_();b=original.focal_numerator(z,y)/7
    assert torch.equal(a,b) and torch.equal(g,torch.autograd.grad(b,z)[0])
    assert torch.allclose(a,sum(p12.loss_for_microbatch(x[i:i+1],y[i:i+1],7,arm) for i in range(3)))
    assert torch.isfinite(p12.loss_for_microbatch(x,torch.zeros_like(y),0,arm))


@pytest.mark.parametrize('arm,seed',[('G',19),('S',1),('C',True),('R',2)])
def test_invalid_configuration(arm,seed):
    with pytest.raises(ValueError):p12.run_config(arm,seed)


def test_config_and_cli_reject_wrong_experiment():
    c=p12.run_config('C',19);r=p12.run_config('R',19)
    assert c['decoder_dropout']==dict(p=0.,location='post_stem_relu',kind='channel2d')
    assert r['decoder_dropout']==dict(p=.2,location='post_stem_relu',kind='channel2d')
    with pytest.raises(ValueError):p12.validate_config(c,r)
    for args in (['--phase','B'],['--arm','G'],['--seed','2']):
        with pytest.raises(SystemExit):p12.parse_args(['--data','x','--weights','x','--out','x','--arm','C']+args)



def test_stochastic_witness_survives_bf16_output_quantization():
    from verify_p12_control import assert_training_stochastic
    class HeadModel(nn.Module):
        def __init__(self):super().__init__();self.head=p12.DropoutPointHead('R')
        def forward(self,x):return self.head(x)
    torch.manual_seed(19);model=HeadModel()
    features=[torch.randn(2,1024,16,16) for _ in range(4)]
    assert assert_training_stochastic(model,features,device_type='cpu')>0
