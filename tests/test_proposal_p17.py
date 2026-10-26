import importlib.util, pathlib,sys,math
import numpy as np
import pytest
import torch
S=pathlib.Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(S))
spec=importlib.util.spec_from_file_location('p17',S/'train_proposal_p17.py')
p17=importlib.util.module_from_spec(spec);spec.loader.exec_module(p17)

def test_local_model_learns_each_stage_and_roundtrips():
    torch.set_num_threads(2);torch.manual_seed(19)
    m=p17.Proposal(None,True,'K');x=torch.rand(1,3,256,256)
    opt=p17.make_optimizer(m,'K');before={n:p.detach().clone() for n,p in m.named_parameters()}
    y=torch.zeros(1,256,256);y[0,100,100]=1
    assert torch.allclose(m(x),torch.full((1,256,256),math.log(.01/.99)),atol=1e-5)
    for step in range(2):
        opt.zero_grad();out=m(x);assert out.shape==(1,256,256) and out.dtype==torch.float32
        loss=p17.loss_for_microbatch(out,y,1,'K');loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    for prefix in ('encoder.blocks.0','encoder.blocks.1','encoder.blocks.2','head'):
        assert any(not torch.equal(before[n],p) for n,p in m.named_parameters() if n.startswith(prefix)),prefix
    fresh=p17.Proposal(None,True,'K');p17.load_adapted(fresh,m.adapted_state())
    assert set(m.adapted_state())==set(m.state_dict())
    with torch.no_grad():assert torch.equal(m.eval()(x),fresh.eval()(x))

def test_k_encoder_receives_original_rgb_normalization():
    m=p17.Proposal(None,True,'K');seen=[]
    h=m.encoder.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
    m(torch.zeros(1,3,256,256));h.remove()
    expected=torch.tensor([-.485/.229,-.456/.224,-.406/.225])
    torch.testing.assert_close(seen[0][0,:,0,0],expected)

@pytest.mark.parametrize('arm',['C','K'])
def test_focal_effective_batch_value_gradient_and_tail(arm):
    x=torch.tensor([[[0.,1.]],[[2.,-1.]],[[.5,.2]]],requires_grad=True);y=torch.zeros_like(x);y[0,0,0]=1
    ref=p17.focal_numerator(x,y)/5;actual=sum(p17.loss_for_microbatch(a,b,5,arm) for a,b in zip(x.split(1),y.split(1)))
    torch.testing.assert_close(actual,ref)
    torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(ref,x)[0])
    for count in (0,1,2):torch.testing.assert_close(p17.loss_for_microbatch(x[-1:],y[-1:],count,arm),p17.focal_numerator(x[-1:],y[-1:])/max(1,count))

def test_k_optimizer_and_schedule_cover_all_tensors_once():
    m=p17.Proposal(None,True,'K');opt=p17.make_optimizer(m,'K')
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    assert set(ids)=={id(p) for p in m.parameters()}
    assert all(g['lr']==3e-4 for g in opt.param_groups)
    p17.schedule_epoch(opt,129);assert all(abs(g['lr']-3e-5)<1e-18 for g in opt.param_groups)

@pytest.mark.parametrize('seed',[19,1])
def test_config_binds_architecture_and_disallows_old_objective(seed):
    c=p17.run_config('C',seed);k=p17.run_config('K',seed)
    with pytest.raises(ValueError):p17.validate_config(k,c)
    with pytest.raises(ValueError):p17.validate_config(k,p17.run_config('K',1 if seed==19 else 19))
    assert k['objective']['kind']=='original_center_focal'

@pytest.mark.parametrize('arm',['R','J','',None])
def test_old_arms_cannot_run(arm):
    with pytest.raises(ValueError):p17.run_config(arm,19)
    with pytest.raises(ValueError):p17.Proposal(None,True,arm)

def test_fold_cannot_read_new_folds():
    with pytest.raises(ValueError):p17.Fold('unavailable',2,'K')

@pytest.mark.parametrize('seed',[True,2,19.0])
def test_reject_invalid_seed(seed):
    with pytest.raises(ValueError):p17.run_config('K',seed)
