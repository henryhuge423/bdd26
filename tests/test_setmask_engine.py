import copy
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec('nucseg.setmask.engine') is not None,'Guarded E6 engine absent'
    return importlib.import_module('nucseg.setmask.engine')


def test_prediction_rejects_test_fold_before_run_or_model_access(tmp_path):
    with pytest.raises(ValueError):api().predict_dev(tmp_path/'absent',fold=3,device='cpu')


@pytest.mark.parametrize('arm,seed',[('gate',19),('B',2),('Q',20)])
def test_frozen_arm_and_seeds_reject_changes(tmp_path,arm,seed):
    with pytest.raises(ValueError):api().SetMaskConfig(arm,seed,str(tmp_path))


def test_existing_run_requires_explicit_resume_and_same_config(tmp_path):
    m=api();cfg=m.SetMaskConfig('B',19,str(tmp_path/'run'))
    provenance={'source':'one','inputs':{'data':'abc'}}
    m.prepare_run(cfg,provenance)
    original=(tmp_path/'run/config.json').read_bytes()
    with pytest.raises(ValueError):m.prepare_run(cfg,provenance)
    with pytest.raises(ValueError):m.prepare_run(m.SetMaskConfig('Q',19,cfg.out_dir),provenance,resume=True)
    assert (tmp_path/'run/config.json').read_bytes()==original


def test_partial_checkpoint_or_changed_provenance_is_refused(tmp_path):
    m=api();cfg=m.SetMaskConfig('Q',1,str(tmp_path/'run'));p={'source':'one','inputs':{}}
    m.prepare_run(cfg,p)
    with pytest.raises(ValueError):m.prepare_run(cfg,p,resume=True)
    torch.save({'model':{}},tmp_path/'run/last.pth')
    with pytest.raises(ValueError):m.prepare_run(cfg,p,resume=True)
    with pytest.raises(ValueError):m.prepare_run(cfg,{**p,'source':'two'},resume=True)


def test_unowned_directory_is_not_overwritten(tmp_path):
    m=api();out=tmp_path/'run';out.mkdir();keep=out/'keep.txt';keep.write_text('mine')
    with pytest.raises(ValueError):m.prepare_run(m.SetMaskConfig('B',19,str(out)),{})
    assert keep.read_text()=='mine' and not (out/'config.json').exists()


class ToyDataset(torch.utils.data.Dataset):
    def __len__(self):return 4
    def __getitem__(self,key):
        index,epoch,draw=key
        return {'img':torch.tensor([index/3.,epoch/10.,draw/3.]),'digest':f'{index}:{epoch}:{draw}'}


def test_real_epoch_runner_resume_reproduces_weights_and_sample_digests(tmp_path):
    m=api();s=importlib.import_module('nucseg.setmask.state')
    def create():
        net=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,1))
        opt=torch.optim.AdamW(net.parameters(),lr=.01)
        return net,opt,torch.optim.lr_scheduler.ExponentialLR(opt,.85)
    def loss(pred,batch):
        value=pred.square().mean();return value,{'toy':float(value.detach())}
    def run(net,opt,sc,epoch):
        loader=m.epoch_loader(ToyDataset(),torch.ones(4),19,epoch,batch_size=2,workers=0)
        rec=m.run_epoch(net,loader,opt,loss,'cpu','B');sc.step();return rec
    torch.manual_seed(98);net,opt,sc=create();run(net,opt,sc,0)
    state={'model':copy.deepcopy(net.state_dict()),'opt':copy.deepcopy(opt.state_dict()),
           'sched':copy.deepcopy(sc.state_dict()),'rng':s.capture_rng()}
    want_digest=run(net,opt,sc,1)['data_digest'];want=copy.deepcopy(net.state_dict())
    resumed,ro,rs=create();resumed.load_state_dict(state['model']);ro.load_state_dict(state['opt']);rs.load_state_dict(state['sched'])
    s.restore_rng(state['rng']);got=run(resumed,ro,rs,1)
    assert got['data_digest']==want_digest
    assert all(torch.equal(want[k],resumed.state_dict()[k]) for k in want)
    assert ro.param_groups[0]['lr']==opt.param_groups[0]['lr']
