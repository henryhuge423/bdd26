"""EMA behavior: analytical values, state isolation, rejected corrupt checkpoints."""
import copy
import importlib.util
from pathlib import Path
import sys
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def module():
    assert importlib.util.find_spec('train_proposal_p23') is not None, 'P23 runner missing'
    import train_proposal_p23
    return train_proposal_p23

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor([2.,4.]))
        self.fixed=nn.Parameter(torch.tensor([9.]),requires_grad=False)

def test_analytic_updates_and_final_tail_event():
    p=module();m=Tiny();e=p.ParameterEMA(m)
    with torch.no_grad():m.weight.copy_(torch.tensor([4.,8.]))
    e.update(m)
    torch.testing.assert_close(e.shadow['weight'],torch.tensor([2.002,4.004]),rtol=0,atol=5e-7)
    with torch.no_grad():m.weight.copy_(torch.tensor([6.,12.]))
    e.update(m)
    torch.testing.assert_close(e.shadow['weight'],torch.tensor([2.005998,4.011996]),rtol=0,atol=5e-7)
    assert e.updates==2
    before=e.shadow['weight'].clone();e.update(m,successful_step=False)
    assert e.updates==2 and torch.equal(before,e.shadow['weight'])

def test_init_update_do_not_alias_online_or_consume_rng():
    p=module();m=Tiny();rng=torch.get_rng_state().clone();e=p.ParameterEMA(m)
    assert e.shadow['weight'].data_ptr()!=m.weight.data_ptr()
    assert set(e.shadow)=={'weight'} and e.shadow['weight'].dtype==torch.float32
    before={n:t.clone() for n,t in m.state_dict().items()};e.update(m)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(t,m.state_dict()[n]) for n,t in before.items())
    saved=e.state_dict();saved['shadow']['weight'].zero_()
    assert not torch.equal(saved['shadow']['weight'],e.shadow['weight'])

def test_apply_preserves_frozen_and_saved_online(tmp_path):
    p=module();m=Tiny();e=p.ParameterEMA(m)
    with torch.no_grad():m.weight.add_(10)
    e.update(m);path=tmp_path/'online.pt';torch.save(m.state_dict(),path)
    old=path.read_bytes();e.apply(m)
    assert torch.equal(m.weight,e.shadow['weight']) and m.fixed.item()==9
    assert path.read_bytes()==old
    assert torch.equal(torch.load(path,weights_only=True)['weight'],torch.tensor([12.,14.]))

def test_shadow_roundtrip_continuation_equals_uninterrupted():
    p=module();m=Tiny();e=p.ParameterEMA(m)
    with torch.no_grad():m.weight.add_(5)
    e.update(m);saved=e.state_dict();r=p.ParameterEMA(m);r.load_state_dict(saved,expected_updates=1)
    with torch.no_grad():m.weight.sub_(3)
    e.update(m);r.update(m)
    assert torch.equal(e.shadow['weight'],r.shadow['weight']) and e.updates==r.updates==2

@pytest.mark.parametrize('kind',['keys','shape','dtype','nan','inf','decay','updates','negative','bool','extra'])
def test_reject_corrupt_shadow_atomically(kind):
    p=module();m=Tiny();e=p.ParameterEMA(m);s=e.state_dict()
    if kind=='keys':s['shadow']={}
    elif kind=='shape':s['shadow']['weight']=torch.zeros(3)
    elif kind=='dtype':s['shadow']['weight']=s['shadow']['weight'].double()
    elif kind in ('nan','inf'):s['shadow']['weight'][0]=float(kind)
    elif kind=='decay':s['decay']=.99
    elif kind=='updates':s['updates']=1
    elif kind=='negative':s['updates']=-1
    elif kind=='bool':s['updates']=False
    else:s['unexpected']=1
    with pytest.raises(ValueError):e.load_state_dict(s,expected_updates=0)
    assert e.updates==0 and torch.equal(e.shadow['weight'],torch.tensor([2.,4.]))

def test_buffer_and_changed_membership_rejected():
    p=module();m=Tiny();m.register_buffer('stats',torch.ones(1))
    with pytest.raises(ValueError):p.ParameterEMA(m)
    m=Tiny();e=p.ParameterEMA(m);m.fixed.requires_grad_(True)
    with pytest.raises(ValueError):e.update(m)

def test_optimizer_step_updates_ema_once_and_abort_does_not_advance():
    p=module();m=Tiny();e=p.ParameterEMA(m);o=torch.optim.SGD([m.weight],lr=.1)
    m.weight.grad=torch.tensor([.1,.2]);p.optimizer_step(m,o,e)
    assert e.updates==1
    torch.testing.assert_close(e.shadow['weight'],torch.tensor([1.99999,3.99998]),rtol=0,atol=5e-7)
    m.weight.grad=torch.tensor([float('nan'),0.]);before=m.weight.detach().clone()
    with pytest.raises(RuntimeError):p.optimizer_step(m,o,e)
    assert e.updates==1 and torch.equal(before,m.weight)

def test_config_preserves_baseline_and_requires_two_seeds():
    p=module()
    for seed in (19,1):
        c=p.run_config(seed)
        assert c['epochs']==130 and c['effective_batch']==16 and c['microbatch']==4
        assert c['trainable_encoder_blocks']==list(range(12,24))
        assert c['ema_decay']==.999 and c['outputs']==['C','E'] and c['seed']==seed
        assert c['encoder_lr']==1e-5 and c['head_lr']==3e-4
    for seed in (2,True,19.):
        with pytest.raises(ValueError):p.run_config(seed)
    with pytest.raises(SystemExit):p.parse_args(['--data','d','--weights','w','--out','o','--resume'])

def test_output_refuses_existing_directory(tmp_path):
    p=module();o=tmp_path/'existing';o.mkdir();(o/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):p.create_output(o)
    assert (o/'sentinel').read_text()=='keep'
