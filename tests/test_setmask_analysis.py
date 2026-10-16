import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def api():
    path=Path(__file__).resolve().parents[1]/'scripts/analyze_setmask_dev.py'
    assert path.is_file(),'Frozen E6 analysis is absent'
    spec=importlib.util.spec_from_file_location('e6_analysis_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixtures():
    base={'mPQ':.5,'bPQ':.66,'strict_mPQ':.44,'strict_dead_PQ':.13,'dead_PQ':.17,'merge_rate':.04}
    good={**base,'mPQ':.501,'bPQ':.661,'strict_mPQ':.441,'strict_dead_PQ':.14,'dead_PQ':.18}
    endpoints={f'{arm}_seed{seed}':dict(base if arm=='B' else good) for arm in ('B','Q') for seed in (19,1)}
    comparison={'per_seed_matched_improvement':{'19':.03,'1':.04},'matched_improvement':.035,'lo':.01,'hi':.06}
    return endpoints,comparison,{'19':1.5,'1':1.6}


def test_all_frozen_gates_pass_on_independent_positive_fixture():
    ep,c,cost=fixtures();assert api().admission(c,ep,cost)['admission']


def test_dead_pq_gain_cannot_replace_failed_geometric_primary():
    ep,c,cost=fixtures();c['per_seed_matched_improvement']['19']=-.01
    assert not api().admission(c,ep,cost)['admission']


def test_any_strict_mpq_decrease_fails_even_if_other_guards_pass():
    ep,c,cost=fixtures();ep['Q_seed1']['strict_mPQ']=.44-1e-7
    assert not api().admission(c,ep,cost)['admission']


def test_cost_limit_applies_to_each_seed_and_missing_cost_cannot_pass():
    ep,c,cost=fixtures();cost['1']=2.001
    assert not api().admission(c,ep,cost)['admission']
    assert not api().admission(c,ep,{})['admission']


def test_mid_iou_is_neither_matched_nor_missed():
    result=api().object_rates(np.array([.05,.3,.8]))
    assert result['matched'].tolist()==[False,False,True]
    assert result['miss'].tolist()==[True,False,False]


def test_paired_comparison_calculates_actual_miss_independently():
    rows={}
    for seed in (19,1):
        for arm,values in [('B',[.05,.3,.8]),('Q',[.2,.3,.9])]:
            rows[(arm,seed)]=[dict(image=i,gt_id=17,cls=4,interior=True,tissue='Lung',max_iou=v)
                              for i,v in enumerate(values)]
    result=api().paired_comparison(rows,n_boot=50)
    assert result['matched_improvement']==0
    assert result['miss_improvement']==pytest.approx(1/3)
    assert result['n_objects']==3 and result['n_images']==3


def test_missing_paired_identity_is_rejected():
    row=dict(image=0,gt_id=17,cls=4,interior=True,tissue='Lung',max_iou=.8)
    rows={(arm,seed):[dict(row)] for arm in ('B','Q') for seed in (19,1)}
    rows[('Q',1)]=[]
    with pytest.raises(ValueError):api().paired_comparison(rows,n_boot=5)


@pytest.mark.parametrize('field,value',[('split',3),('seed',7),('round','E5')])
def test_round_rejects_changed_scientific_configuration_before_fold_access(tmp_path,field,value):
    from dataclasses import asdict
    import json
    from nucseg.setmask.engine import SetMaskConfig
    run=tmp_path/'B_seed19';run.mkdir()
    payload=asdict(SetMaskConfig('B',19,str(run)));payload[field]=value
    (run/'config.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):api().verify_metadata(tmp_path)


def test_provenance_binds_reused_analysis_helpers(tmp_path,monkeypatch):
    import nucseg.setmask.engine as e
    root=tmp_path
    (root/'src/nucseg').mkdir(parents=True);(root/'scripts').mkdir()
    for name in ('analyze_fusion_arms.py','probe_visibility.py','analyze_setmask_dev.py'):
        (root/'scripts'/name).write_text('# fixed helper')
    for name in ('images','inst','type','tissue'):(root/(name+'.npy')).write_bytes(b'fixture')
    (root/'uni').write_bytes(b'UNI')
    class Fold:
        def __init__(self,fold):self.dir=root
    monkeypatch.setattr(e,'ROOT',root);monkeypatch.setattr(e,'PanNukeFold',Fold)
    sources=e.make_provenance(e.SetMaskConfig('B',19,'unused',str(root/'uni')))['sources']
    assert 'scripts/analyze_fusion_arms.py' in sources and 'scripts/probe_visibility.py' in sources
