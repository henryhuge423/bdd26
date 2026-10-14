"""Frozen crop/HV controls: identity, final working area, RNG and dev-fold guards."""
import random
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from nucseg.cellvit import data, engine
from nucseg.hovernet.data import hv_targets


def geometry():
    inst = np.zeros((32,32), np.int32)
    inst[10:16, 7:12] = 7  # complete, cut at x=10 to two columns
    inst[20:23, 12:15] = 42  # complete small object, fully retained
    inst[0:3, 7:12] = 99  # originally incomplete, also crop-cut
    return inst


def crop(inst):
    tr = data.VisibilityCrop((16,32), 32,32,p=1)
    mask = np.stack([inst, (inst>0).astype(np.int32)], -1)
    out = tr.apply_to_mask(mask, crop_height=32,crop_width=22,h_start=0,w_start=10/11,interpolation=0)
    return tr, out[...,0]


def test_crop_identity_before_resize_and_original_border_exclusion():
    inst=geometry()
    tr, final=crop(inst)
    assert tr.truncated_ids == {7,99}
    assert data.original_border_ids(inst) == {99}
    # Working area is evaluated after final geometry, never in the native crop.
    final[final==7]=0
    final[10:13,0:2]=7
    base=hv_targets(final, min_size=30)
    actual, counts=data.visibility_hv_targets(final,'A2',tr.truncated_ids-data.original_border_ids(inst))
    mask=final==7
    np.testing.assert_array_equal(actual[~mask],base[~mask])
    assert np.any(actual[mask])
    assert counts['eligible']==1
    assert not np.any(actual[final==42])
    assert not np.any(actual[final==99])
    np.testing.assert_array_equal(data.visibility_hv_targets(final,'A3',set())[0],hv_targets(final,min_size=1))


def test_full_crop_edge_contact_without_pixel_loss_is_complete():
    inst=geometry()
    tr=data.VisibilityCrop((32,32),32,32,p=1)
    tr.apply_to_mask(np.stack([inst,inst],-1),crop_height=32,crop_width=32,h_start=0,w_start=0)
    assert tr.truncated_ids==set()


@pytest.mark.parametrize('arm',['A0','A1'])
def test_baseline_targets_exact(arm):
    inst=geometry()
    np.testing.assert_array_equal(data.visibility_hv_targets(inst,arm,{42})[0],hv_targets(inst,min_size=30))


def test_final_area_and_degenerate_targets():
    inst=np.zeros((32,32),np.int32)
    inst[10:16,10:15]=7  # exactly30, must keep default
    inst[20,20]=42
    inst[25,20:24]=99
    actual, counts=data.visibility_hv_targets(inst,'A2',{7,42,99})
    assert counts['eligible']==2 and counts['single_pixel']==1 and counts['single_axis']==2
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[inst==7],hv_targets(inst,min_size=30)[inst==7])
    np.testing.assert_array_equal(actual[inst==42],hv_targets(inst,min_size=1)[inst==42])


def test_same_random_sequence_for_augmented_controls():
    inst=geometry()
    img=np.repeat(inst[...,None].astype(np.uint8),3,-1)
    results=[]
    for arm in ['A1','A2','A3']:
        random.seed(83); np.random.seed(83)
        aug=data.cellvit_train_augs(32,visibility_arm=arm)
        aug.transforms[-2].p=1
        aug.transforms[-1].p=1
        output=aug(image=img,mask=np.stack([inst,inst],-1))
        results.append((output,random.getstate(),np.random.get_state()))
    for out, py, ns in results[1:]:
        np.testing.assert_array_equal(out['image'],results[0][0]['image'])
        np.testing.assert_array_equal(out['mask'],results[0][0]['mask'])
        assert py==results[0][1]
        np.testing.assert_array_equal(ns[1],results[0][2][1])
        assert ns[2:]==results[0][2][2:]


def test_a0_uses_unmodified_crop_recipe():
    aug=data.cellvit_train_augs(visibility_arm='A0')
    assert type(aug.transforms[-2]).__name__=='RandomSizedCrop'
    assert aug.transforms[-2].p==.1
    for arm in ['A1','A2','A3']:
        assert data.cellvit_train_augs(visibility_arm=arm).transforms[-2].p==.5


@pytest.mark.parametrize('arm',['A0','A1','A2','A3'])
def test_config_arm_frozen_and_guarded(tmp_path,arm):
    cfg=engine.TrainConfig(split=1,out_dir=str(tmp_path),visibility_arm=arm)
    assert cfg.visibility_arm==arm
    with pytest.raises(ValueError,match='split1'):
        engine.TrainConfig(split=2,out_dir=str(tmp_path),visibility_arm=arm)
    with pytest.raises(ValueError,match='fold1.*fold2'):
        engine.train(cfg,[1],[3],device='cpu')
    assert not (tmp_path/'config.json').exists()


@pytest.mark.parametrize('kw',[{'upscale':2},{'hv_min_size':1},{'dead_expert':True},{'cp_prob':.1},{'np_wce':1},{'synth_frac':.1},{'widen':2}])
def test_arm_refuses_other_interventions(tmp_path,kw):
    with pytest.raises(ValueError,match='visibility'):
        engine.TrainConfig(split=1,out_dir=str(tmp_path),visibility_arm='A2',**kw)


def cli():
    spec=importlib.util.spec_from_file_location('visibility_cli',Path(__file__).resolve().parents[1]/'scripts/train_cellvit.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('arm',['A0','A1','A2','A3'])
def test_cli_requires_dev_train_only(tmp_path,arm,monkeypatch):
    m=cli()
    def forbidden(*a,**kw):pytest.fail('unsafe CLI reached training')
    monkeypatch.setattr(m,'train',forbidden)
    with pytest.raises(SystemExit):m.main(['--split','1','--out',str(tmp_path),'--visibility-arm',arm])
    seen=[]
    monkeypatch.setattr(m,'train',lambda cfg,tr,va:seen.append((cfg,tr,va)))
    m.main(['--split','1','--out',str(tmp_path),'--visibility-arm',arm,'--train-only'])
    assert seen[0][0].visibility_arm==arm and seen[0][1:]==([1],[2])


@pytest.fixture
def sample_fold(monkeypatch):
    class Fold:
        def __init__(self,k):
            assert k in (1,2), "test fold accessed"
            self.inst=geometry()[None]
            self.type=np.where(self.inst>0,4,0).astype(np.uint8)
            self.images=np.repeat(self.inst[...,None].astype(np.uint8),3,-1)
            self.tissue=np.array(["Colon"])
        def __len__(self): return 1
    monkeypatch.setattr(data,'PanNukeFold',Fold)


def test_dataset_selective_targets_and_skipped_crop_reset(sample_fold):
    import albumentations as A
    ds=data.PanNukeCellViT([1],True,visibility_arm='A2')
    cropper=data.VisibilityCrop((32,32),32,32,p=1)
    cropper.get_params=lambda:dict(crop_height=32,crop_width=22,h_start=0,w_start=10/11)
    ds.augs=A.Compose([cropper])
    first=ds[0]
    assert first['visibility_counts'][0]==1
    assert first['hv_map'].abs().sum()>0
    cropper.p=0
    second=ds[0]
    assert second['visibility_counts'][0]==0
    np.testing.assert_array_equal(second['hv_map'],hv_targets(geometry(),min_size=30))


def test_a0_dataset_is_bit_identical_to_legacy(sample_fold):
    old=data.PanNukeCellViT([1],True)
    new=data.PanNukeCellViT([1],True,visibility_arm='A0')
    old.augs=data.cellvit_train_augs(32)
    new.augs=data.cellvit_train_augs(32,visibility_arm='A0')
    for seed in range(5):
        random.seed(seed);np.random.seed(seed);a=old[0]
        random.seed(seed);np.random.seed(seed);b=new[0]
        for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_rotated_original_border_exclusion_and_identity(sample_fold):
    import albumentations as A
    ds=data.PanNukeCellViT([1],True,visibility_arm='A2')
    rotation=A.RandomRotate90(p=1)
    rotation.get_params=lambda:dict(factor=1)
    cropper=data.VisibilityCrop((32,32),32,32,p=1)
    cropper.get_params=lambda:dict(crop_height=22,crop_width=32,h_start=0,w_start=0)
    ds.augs=A.Compose([rotation,cropper])
    ds[0]
    # original99 rotates onto left border; preexisting incompleteness remains excluded
    assert 99 in data.original_border_ids(np.rot90(geometry()))
    assert ds[0]['visibility_counts'][0]==1


@pytest.mark.parametrize('arm',['A0','A1','A2','A3'])
def test_validation_uses_original_geometry(sample_fold,arm):
    ds=data.PanNukeCellViT([2],False,visibility_arm=arm)
    sample=ds[0]
    assert ds.augs is None
    np.testing.assert_array_equal(sample['hv_map'],hv_targets(geometry(),min_size=30))
    assert sample['visibility_counts'][0]==0


def test_dataset_refuses_fold3_before_io(monkeypatch):
    def forbidden(*a):pytest.fail('fold3 opened')
    monkeypatch.setattr(data,'PanNukeFold',forbidden)
    with pytest.raises(ValueError,match='fold1.*fold2'):
        data.PanNukeCellViT([3],False,visibility_arm='A2')


def test_resume_refuses_visibility_arm_change(tmp_path):
    a=engine.TrainConfig(split=1,out_dir=str(tmp_path),visibility_arm='A1')
    engine.prepare_train_config(a)
    with pytest.raises(ValueError,match='visibility_arm'):
        engine.prepare_train_config(engine.TrainConfig(split=1,out_dir=str(tmp_path),visibility_arm='A2'))


def test_train_seeds_python_augmentation_rng(tmp_path,monkeypatch):
    import torch
    seen=[]
    def stop(*a,**kw):
        seen.append(random.random())
        raise RuntimeError('dataset stop')
    monkeypatch.setattr(engine,'PanNukeCellViT',stop)
    cfg=engine.TrainConfig(split=1,out_dir=str(tmp_path),visibility_arm='A2',seed=19)
    random.seed(999)
    with pytest.raises(RuntimeError,match='dataset stop'):
        engine.train(cfg,[1],[2],device='cpu',model=torch.nn.Linear(1,1))
    assert seen==[random.Random(19).random()]


@pytest.mark.parametrize('arm',['A0','A1','A2','A3'])
def test_engine_validation_dataset_unified_hv30(sample_fold,tmp_path,monkeypatch,arm):
    import torch
    real_dataset=engine.PanNukeCellViT
    datasets=[]
    def capture(*args,**kwargs):
        ds=real_dataset(*args,**kwargs)
        datasets.append(ds)
        return ds
    monkeypatch.setattr(engine,'PanNukeCellViT',capture)
    cfg=engine.TrainConfig(split=1,out_dir=str(tmp_path),visibility_arm=arm,epochs=0,workers=1)
    engine.train(cfg,[1],[2],device='cpu',model=torch.nn.Linear(1,1))
    assert len(datasets)==2
    train_ds,val_ds=datasets
    assert train_ds.visibility_arm==arm and not val_ds.train
    np.testing.assert_array_equal(val_ds[0]['hv_map'],hv_targets(geometry(),min_size=30))
