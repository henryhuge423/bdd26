"""P8 objective arithmetic, tail accumulation and control invariants."""
import importlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
torch.set_num_threads(2)


def p8():
    assert importlib.util.find_spec('train_proposal_p8') is not None,'P8 trainer is missing'
    return importlib.import_module('train_proposal_p8')


def test_image_normalization_analytical_counts_and_empty_image():
    m=p8()
    logits=torch.zeros((3,2,2),requires_grad=True)
    target=torch.zeros_like(logits)
    target[0,0,0]=target[1,0,0]=1
    # Three true objects in image1 can share the same rounded center: use counts, not target peaks.
    loss=m.loss_for_microbatch(logits,target,[1,3,0],4,3,'I')
    assert float(loss)==pytest.approx(math.log(2)*7/9,rel=1e-6)
    loss.backward()
    magnitude=(.25*math.log(2)+.125)
    signs=torch.where(target==1,-1.,1.)
    expected=signs*magnitude*torch.tensor([1.,1/3,1.])[:,None,None]/3
    assert torch.allclose(logits.grad,expected,rtol=1e-6,atol=1e-8)


def test_control_loss_and_gradient_are_bitwise_original():
    m=p8();import proposal_p3_utils as u
    target=torch.zeros((4,7,7));target[:,3,3]=1;target[2]=0
    values=torch.linspace(-2,2,target.numel()).reshape_as(target)
    a=values.clone().requires_grad_();b=values.clone().requires_grad_()
    actual=m.loss_for_microbatch(a,target,[1,7,0,2],31,16,'C')
    expected=u.focal_numerator(b,target)/31
    assert torch.equal(actual,expected)
    actual.backward();expected.backward();assert torch.equal(a.grad,b.grad)


@pytest.mark.parametrize('counts',[[1,3,0],[0,0,0],[6,2,1,0,9]])
def test_image_full_effective_vs_micro_gradients_and_tail(counts):
    m=p8();import torch.nn.functional as F
    values=torch.linspace(-2,2,len(counts)*20).reshape(len(counts),4,5)
    target=torch.zeros_like(values)
    for i,n in enumerate(counts):
        if n:target[i,1,1]=1;target[i,1,2]=.6
    full=values.clone().requires_grad_();micro=values.clone().requires_grad_()
    # Independent focal expression and weights: no production normalization helper.
    prob=full.sigmoid()
    terms=-(F.logsigmoid(full)*(1-prob)**2*(target==1)+F.logsigmoid(-full)*prob**2*(1-target)**4*(target<1))
    weights=full.new_tensor([1/max(1,n)/len(counts) for n in counts])
    expected=(terms.sum((1,2))*weights).sum();expected.backward()
    measured=[]
    for start in range(0,len(counts),2):
        end=min(start+2,len(counts))
        value=m.loss_for_microbatch(micro[start:end],target[start:end],counts[start:end],sum(counts),len(counts),'I')
        measured.append(float(value.detach()));value.backward()
    assert sum(measured)==pytest.approx(float(expected.detach()),rel=1e-6)
    assert torch.allclose(micro.grad,full.grad,rtol=1e-6,atol=1e-7)
    if not any(counts):assert micro.grad.abs().sum()>0


def test_all_empty_control_stays_original():
    m=p8();import proposal_p3_utils as u
    x=torch.full((2,4,4),-.8,requires_grad=True);y=torch.zeros_like(x)
    assert torch.equal(m.loss_for_microbatch(x,y,[0,0],0,2,'C'),u.focal_numerator(x,y))


@pytest.mark.parametrize('counts,total,batch,arm',[
    ([1],1,1,'J'),([-1],1,1,'I'),([1.5],2,1,'I'),([True],1,1,'I'),
    ([],0,1,'I'),([1],0,1,'C'),([1],1,0,'I'),([1],1.5,1,'C'),([1],1,True,'I')])
def test_invalid_objective_metadata_rejected(counts,total,batch,arm):
    m=p8()
    with pytest.raises(ValueError):m.loss_for_microbatch(torch.zeros((1,2,2)),torch.zeros((1,2,2)),counts,total,batch,arm)


def test_shape_mismatch_rejected():
    m=p8()
    with pytest.raises(ValueError):m.loss_for_microbatch(torch.zeros((2,2,2)),torch.zeros((1,2,2)),[1,1],2,2,'I')


def test_batches_and_all_original_science_functions(tmp_path):
    m=p8();import train_proposal_p3 as old;import proposal_p3_utils as u
    root=tmp_path/'fold1';root.mkdir()
    image=np.random.default_rng(11).integers(10,230,(2,256,256,3),dtype=np.uint8)
    inst=np.zeros((2,256,256),np.uint16);typ=np.zeros_like(inst,dtype=np.uint8)
    inst[0,15:20,22:26]=1;typ[0,15:20,22:26]=4
    for name,value in [('images',image),('inst',inst),('type',typ),('tissue',np.array(['Lung','Skin']))]:np.save(root/(name+'.npy'),value)
    c=m.Fold(tmp_path,1,'C');i=m.Fold(tmp_path,1,'I');ref=old.Fold(tmp_path,1)
    for aug in (False,True):
        a=c.batch([0,1],19,3,aug);b=i.batch([0,1],19,3,aug);r=ref.batch([0,1],19,3,aug)
        assert torch.equal(a[0],b[0]) and torch.equal(a[0],r[0])
        assert torch.equal(a[1],b[1]) and torch.equal(a[1],r[1]) and a[2]==b[2]==r[2]==1
    assert m.Proposal is old.Proposal and m.evaluate is old.evaluate
    for name in ('split_indices','gaussian_target','augment_pair','focal_numerator','decode_peaks','score_points','aggregate','admission'):assert getattr(m,name) is getattr(u,name)


def test_config_contract_and_safe_cli(tmp_path):
    m=p8();config=m.run_config('I',1);m.validate_config(config,config)
    assert config['normalization']==dict(kind='image_mean',count_source='instance_count',empty='full_negative',effective_batch='actual')
    assert m.run_config('C',19)['normalization']['kind']=='effective_batch_object'
    for saved in [m.run_config('C',1),{**config,'seed':19},{**config,'normalization':{'kind':'image_mean'}}]:
        with pytest.raises(ValueError):m.validate_config(saved,config)
    for arm,seed in [('J',1),('I',2)]:
        with pytest.raises(ValueError):m.run_config(arm,seed)
    base=['--data','d','--weights','w','--out','o','--arm','I']
    assert m.parse_args(base).phase=='A'
    for extra in [['--phase','B'],['--resume'],['--seed','2'],['--arm','J']]:
        with pytest.raises(SystemExit):m.parse_args(base+extra)
    for fold,arm in [(2,'C'),(1,'J')]:
        with pytest.raises(ValueError):m.Fold(tmp_path,fold,arm)
