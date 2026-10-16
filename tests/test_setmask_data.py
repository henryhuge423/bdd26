import copy
import importlib
import importlib.util
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


def api():
    assert importlib.util.find_spec('nucseg.setmask') is not None, 'E6 paired data package is absent'
    return importlib.import_module('nucseg.setmask.data')


@pytest.fixture
def tiny_fold(tmp_path):
    d=tmp_path/'fold1';d.mkdir()
    rng=np.random.default_rng(7)
    images=rng.integers(0,256,(4,256,256,3),dtype=np.uint8)
    inst=np.zeros((4,256,256),np.uint16)
    inst[0,50:70,80:100]=17;inst[1,100:116,80:96]=2;inst[2,3,7]=91
    typ=np.where(inst>0,4,0).astype(np.uint8)
    for name,array in dict(images=images,inst=inst,type=typ,tissue=np.array(['Breast']*4)).items():
        np.save(d/(name+'.npy'),array)
    return tmp_path


def test_sample_keys_are_independent_of_model_rng():
    m=api();weights=torch.tensor([1.,2.,3.])
    keys=m.sample_keys(weights,19,4)
    torch.rand(100);np.random.rand(100);random.random()
    assert m.sample_keys(weights,19,4)==keys
    assert m.sample_keys(weights,19,5)!=keys
    assert [k[2] for k in keys]==[0,1,2]
    assert all(0<=k[0]<3 and k[1]==4 for k in keys)


def test_workers_and_global_rng_do_not_change_augmented_targets(tiny_fold):
    m=api();ds=m.PairedDataset(1,True,19,root=tiny_fold)
    keys=m.sample_keys(torch.ones(4),19,3)
    def collect(workers):
        loader=DataLoader(ds,batch_size=2,sampler=keys,num_workers=workers,
                          generator=torch.Generator().manual_seed(44))
        return [(b['digest'],b['inst_map'].clone(),b['img'].clone()) for b in loader]
    first=collect(0)
    random.seed(990);np.random.seed(991);torch.manual_seed(992)
    second=collect(2)
    assert [x[0] for x in first]==[x[0] for x in second]
    for a,b in zip(first,second):
        assert torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])


def test_epoch_restart_reconstructs_same_stream(tiny_fold):
    m=api();keys=m.sample_keys(torch.ones(4),1,9)
    a=m.PairedDataset(1,True,1,root=tiny_fold)
    b=m.PairedDataset(1,True,1,root=tiny_fold)
    assert [a[k]['digest'] for k in keys]==[b[k]['digest'] for k in keys]


def test_dataset_refuses_fold3_before_opening_files(tmp_path):
    with pytest.raises(ValueError):api().PairedDataset(3,False,19,root=tmp_path)


def test_empty_image_remains_valid_negative(tiny_fold):
    ds=api().PairedDataset(1,False,19,root=tiny_fold)
    b=ds[3]
    assert b['inst_map'].count_nonzero()==0
    assert b['np_map'].count_nonzero()==0
    assert torch.isfinite(b['img']).all()


def test_data_access_does_not_advance_caller_rng(tiny_fold):
    ds=api().PairedDataset(1,True,19,root=tiny_fold)
    random.seed(33);np.random.seed(34)
    r=random.getstate();n=np.random.get_state()
    ds[(0,1,0)]
    expected_r=random.Random();expected_r.setstate(r)
    expected_n=np.random.RandomState();expected_n.set_state(n)
    assert random.random()==expected_r.random()
    assert np.random.rand()==expected_n.rand()


def test_rng_checkpoint_restores_dropout_optimizer_and_scheduler(tmp_path):
    api();s=importlib.import_module('nucseg.setmask.state')
    torch.manual_seed(4);random.seed(4);np.random.seed(4)
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.4),torch.nn.Linear(4,1))
    opt=torch.optim.AdamW(model.parameters(),lr=.02)
    sched=torch.optim.lr_scheduler.ExponentialLR(opt,.85)
    x=torch.arange(12,dtype=torch.float32).reshape(4,3)/10
    def step(m,o,sc):
        o.zero_grad();loss=m(x).square().mean();loss.backward();o.step();sc.step()
    step(model,opt,sched)
    saved=dict(model=copy.deepcopy(model.state_dict()),opt=copy.deepcopy(opt.state_dict()),
               sched=copy.deepcopy(sched.state_dict()),rng=s.capture_rng())
    s.atomic_save(tmp_path/'last.pth',**saved)
    step(model,opt,sched)
    want=copy.deepcopy(model.state_dict())
    resumed=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.4),torch.nn.Linear(4,1))
    ro=torch.optim.AdamW(resumed.parameters(),lr=.02)
    rs=torch.optim.lr_scheduler.ExponentialLR(ro,.85)
    state=torch.load(tmp_path/'last.pth',weights_only=False)
    resumed.load_state_dict(state['model']);ro.load_state_dict(state['opt']);rs.load_state_dict(state['sched'])
    s.restore_rng(state['rng']);step(resumed,ro,rs)
    assert all(torch.equal(want[k],resumed.state_dict()[k]) for k in want)
    assert ro.param_groups[0]['lr']==opt.param_groups[0]['lr']
    for k,v in opt.state_dict()['state'].items():
        for name,value in v.items():
            assert torch.equal(value,ro.state_dict()['state'][k][name])
