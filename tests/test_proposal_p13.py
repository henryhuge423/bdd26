import copy
import random
import numpy as np
import pytest
import torch
from torch import nn
import train_proposal_p3 as original
import train_proposal_p13 as p13


def order(arm='S',seed=19,epoch=0,fit=None,pool=None,**kw):
    return p13.epoch_order(list(range(37)) if fit is None else fit,
        [2,7,29] if pool is None else pool,seed,epoch,arm,population_size=100,**kw)


@pytest.mark.parametrize('seed',[19,1])
def test_control_original_order_and_true_tail(seed):
    ids,proof=order('C',seed)
    expected=np.random.default_rng(np.random.SeedSequence([seed,0,123])).permutation(list(range(37))).tolist()
    assert ids==expected and proof['replacement_count']==0
    assert [len(ids[i:i+16]) for i in range(0,len(ids),16)]==[16,16,5]
    assert proof['exposures']==37 and proof['unique_images']==37


@pytest.mark.parametrize('seed,epoch',[(19,0),(1,0),(19,6)])
def test_sampled_order_independent_position_and_draw_provenance(seed,epoch):
    ids,proof=order(seed=seed,epoch=epoch)
    expected=np.random.default_rng(np.random.SeedSequence([seed,epoch,123])).permutation(37)
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,20261024,13]))
    positions=rng.choice(37,size=9,replace=False).tolist()
    draws=rng.choice([2,7,29],size=9,replace=True).tolist()
    for i,v in zip(positions,draws):expected[i]=v
    assert ids==expected.tolist()
    assert proof['replacement_positions']==positions and proof['replacement_draws']==draws
    assert proof['replacement_count']==9 and proof['exposures']==37
    assert proof['dead_exposures']==sum(i in (2,7,29) for i in ids)
    assert proof['unique_positive_images']==len(set(ids)&{2,7,29})
    assert proof['order_sha256']==p13.ids_sha(ids)


def test_determinism_change_seed_epoch_and_global_rng_isolation():
    random.seed(72);np.random.seed(72);torch.manual_seed(72)
    before=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    for arm in ('C','S'):
        assert order(arm)==order(arm)
        assert order(arm)[0]!=order(arm,epoch=1)[0]
        assert order(arm)[0]!=order(arm,seed=1)[0]
    assert random.getstate()==before[0]
    after=np.random.get_state()
    assert after[0]==before[1][0] and np.array_equal(after[1],before[1][1]) and after[2:]==before[1][2:]
    assert torch.equal(torch.get_rng_state(),before[2])


@pytest.mark.parametrize('changes',[
    {'fit':[]},{'fit':[1,1]},{'fit':[-1,2]},{'fit':[1,100]}, {'fit':[True,2]},
    {'fit':[1.,2.]},{'pool':[]},{'pool':[2,2]},{'pool':[99]},
    {'seed':2},{'seed':True},{'epoch':-1},{'epoch':True},{'epoch':1.5},{'arm':'R'}])
def test_invalid_sampler_inputs(changes):
    with pytest.raises(ValueError):order(**changes)


@pytest.mark.parametrize('size',[0,-1,True,4.5])
def test_invalid_population_size(size):
    with pytest.raises(ValueError):p13.epoch_order([0],[0],19,0,'C',population_size=size)


def test_base_and_pool_must_belong_to_declared_eligible_cohort():
    for base,pool in (([0,2],[0]),([0],[2])):
        with pytest.raises(ValueError):
            p13.epoch_order(base,pool,19,0,'S',population_size=3,eligible_fit=[0,1])


def test_smoke_full_fit_pool_and_border_positive_eligibility_without_audit():
    # Image3 contains only border Dead; image4 is not fit and must not be inspected.
    gt=[[],[(2,(1,1),True,4)],[],[(4,(0,0),False,1)]]
    pool=p13.positive_pool([0,1,2,3],gt)
    assert pool==[3]
    ids,proof=p13.epoch_order([0,1,2],pool,19,0,'S',population_size=4,eligible_fit=[0,1,2,3])
    assert len(ids)==3 and proof['replacement_count']==0
    ids,proof=p13.epoch_order([0,1,2,3],pool,19,0,'S',population_size=4)
    assert proof['replacement_draws']==[3] and proof['dead_exposures']>0


def test_smoke_allows_zero_positive_control_base_and_requires_sampled_exposure():
    base=list(range(256));pool=[399];eligible=list(range(400))
    _,control=p13.epoch_order(base,pool,19,0,'C',population_size=400,eligible_fit=eligible)
    _,sampled=p13.epoch_order(base,pool,19,0,'S',population_size=400,eligible_fit=eligible)
    assert control['dead_exposures']==0 and sampled['dead_exposures']==64
    p13.validate_smoke_sampling(control,'C');p13.validate_smoke_sampling(sampled,'S')
    with pytest.raises(ValueError):p13.validate_smoke_sampling(control,'S')


def test_duplicate_occurrences_use_original_identical_augmented_view():
    fold=object.__new__(p13.Fold)
    fold.images=np.arange(256*256*3,dtype=np.uint8).reshape(1,256,256,3)
    fold.gt=[[(4,(12,15),True,3)]]
    x,y,n=fold.batch([0,0],19,8,True)
    assert torch.equal(x[0],x[1]) and torch.equal(y[0],y[1]) and n==2
    a,b,_=original.Fold.batch(fold,[0],19,8,True)
    assert torch.equal(x[:1],a) and torch.equal(y[:1],b)


@pytest.mark.parametrize('arm',['C','S'])
def test_original_initial_model_architecture_and_rng(monkeypatch,arm):
    from nucseg.cellvit import model as module
    class FakeEncoder(nn.Module):
        def __init__(self,**kwargs):super().__init__();self.blocks=nn.ModuleList([nn.Linear(1,1) for _ in range(24)])
        def load_state_dict(self,state,strict=True):return None
    monkeypatch.setattr(module,'UNIEncoder',FakeEncoder)
    monkeypatch.setattr(torch,'load',lambda *a,**kw:{})
    torch.manual_seed(19);a=original.Proposal('unused',True);rng=torch.get_rng_state().clone()
    torch.manual_seed(19);b=p13.Proposal('unused',True,arm)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state())
    assert type(b.head) is original.PointHead and not any(isinstance(m,nn.Dropout2d) for m in b.modules())
    b.train();assert not b.encoder.training
    assert all(p.requires_grad==(i>=12) for i,block in enumerate(b.encoder.blocks) for p in block.parameters())


@pytest.mark.parametrize('arm',['C','S'])
def test_original_loss_effective_denominator_including_tail(arm):
    x=torch.linspace(-50,50,60).reshape(3,4,5).requires_grad_();y=torch.zeros_like(x);y[0,1,1]=1
    loss=p13.loss_for_microbatch(x,y,7,arm)
    expected=original.focal_numerator(x,y)/7
    assert torch.equal(loss,expected)
    assert torch.equal(torch.autograd.grad(loss,x)[0],torch.autograd.grad(expected,x)[0])
    assert torch.isfinite(p13.loss_for_microbatch(x,y,0,arm))


def test_config_reload_rejects_arm_seed_recipe_and_original_fit_evaluation(monkeypatch):
    c=p13.run_config('C',19);s=p13.run_config('S',19)
    assert 'decoder_dropout' not in c
    for other in (s,p13.run_config('C',1),dict(c,sampling={})):
        with pytest.raises(ValueError):p13.validate_config(c,other)
    assert c['sampling']['replacement_fraction']==.25
    seen=[]
    monkeypatch.setattr(p13,'evaluate',lambda model,fold,ids:seen.append(list(ids)) or [])
    fit=[0,1,2,3];sampled=[0,0,3,3]
    p13.evaluate_fit(None,None,fit)
    assert seen==[fit] and seen[0]!=sampled
    with pytest.raises(ValueError):p13.evaluate_fit(None,None,sampled)


@pytest.mark.parametrize('arm,seed',[('R',19),('C',True),('S',2)])
def test_invalid_config(arm,seed):
    with pytest.raises(ValueError):p13.run_config(arm,seed)


def test_checkpoint_roundtrip_config_and_adapted_tensors(tmp_path):
    # Small real tensors exercise saved metadata and the original strict adapted loader.
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__();self.head=nn.Linear(2,1);self.encoder=nn.Linear(2,2)
            self.encoder.weight.requires_grad_(False);self.encoder.bias.requires_grad_(False)
    a=TinyModel();b=TinyModel();state={n:p.detach().clone() for n,p in a.named_parameters() if p.requires_grad}
    path=tmp_path/'last.pth';torch.save(dict(model=state,config=p13.run_config('S',1)),path)
    saved=torch.load(path,weights_only=False)
    p13.validate_config(saved['config'],p13.run_config('S',1));p13.load_adapted(b,saved['model'])
    assert all(torch.equal(p,saved['model'][n]) for n,p in b.named_parameters() if p.requires_grad)
    with pytest.raises(ValueError):p13.validate_config(saved['config'],p13.run_config('C',1))
    with pytest.raises(ValueError):p13.load_adapted(b,dict(saved['model'],extra=torch.ones(1)))


def test_frozen_hashes_include_frozen_parameters_and_all_buffers():
    model=nn.Module();model.encoder=nn.Linear(2,2)
    model.encoder.weight.requires_grad_(False);model.encoder.register_buffer('running',torch.ones(1))
    before=p13.frozen_hashes(model)
    assert set(before)=={'weight','running'}
    with torch.no_grad():model.encoder.bias.add_(1)
    assert p13.frozen_hashes(model)==before
    model.encoder.running.add_(1)
    assert p13.frozen_hashes(model)!=before


@pytest.mark.parametrize('args',[['--phase','B'],['--arm','R'],['--seed','2']])
def test_cli_rejects_wrong_experiment(args):
    with pytest.raises(SystemExit):
        p13.parse_args(['--data','x','--weights','x','--out','x','--arm','C']+args)
