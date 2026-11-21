import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_calibrated_consensus_p38 as p


def phases():
    return [[] for _ in range(16)]


def test_original_alignment_excluded_and_phase_duplicates_do_not_change_mean():
    pp=phases(); pp[p.ORIGINAL]=[(50,50,.99)]
    a,b=p.SHIFTED[:2]
    pp[a]=[(100,100,.8),(101,100,.6)];pp[b]=[(102,100,.7)]
    g=p.low_score_groups(pp)
    assert len(g)==1 and g[0]['support']==2
    assert g[0]['position']==[101.,100.]
    assert g[0]['features'][2]==2/15
    assert g[0]['features'][3]==1.


def test_floor_seed_radius_and_deterministic_ties():
    pp=phases();a,b=p.SHIFTED[:2]
    pp[a]=[(100,100,.2),(100,105,.2),(20,20,.099)]
    pp[b]=[(100,104,.2)]
    g=p.low_score_groups(pp)
    assert len(g)==2
    assert g[0]['position']==[100.,102.] and g[1]['position']==[100.,105.]
    for row in pp:row.reverse()
    assert p.low_score_groups(pp)==g


def test_border_support_drops_alignments_invalid_at_seed():
    pp=phases()
    # Dynamically find an alignment valid at a point within seed radius but
    # invalid at the seed; the stored point itself remains a legal observation.
    found=False
    for k in p.SHIFTED:
        for x in range(9):
            if not p.VALID[k,100,x] and p.VALID[k,100,x+4]:
                legal=next(j for j in p.SHIFTED if p.VALID[j,100,x])
                pp[legal]=[(100,x,.9)];pp[k]=[(100,x+4,.2)]
                g=p.low_score_groups(pp)[0]
                assert g['support']==1 and g['support']<=g['usable']
                assert g['position']==[100.,float(x)]
                found=True;break
        if found:break
    assert found


def test_image_split_is_order_invariant_disjoint_and_complete():
    fit,select=p.calibration_split(range(240))
    assert len(fit)==len(select)==120 and not set(fit)&set(select)
    assert set(fit)|set(select)==set(range(240))
    assert p.calibration_split(reversed(range(240)))==(fit,select)
    with pytest.raises(ValueError):p.calibration_split([0]*240)


def test_logistic_fit_learns_quality_and_handles_constant_features():
    rng=np.random.default_rng(19)
    x=rng.normal(size=(200,6));x[:,2]=.8
    y=(x[:,0]+.2*x[:,1]>0).astype(int)
    m=p.logistic_fit(x,y)
    q=p.qualities([dict(features=row) for row in x],m)
    assert m['converged'] and np.isfinite(q).all()
    assert np.mean(q[y==1])>np.mean(q[y==0])+.5
    assert m['scale'][2]==1.
    assert p.qualities([],m).shape==(0,)
    with pytest.raises(ValueError):p.logistic_fit(x,np.zeros(200))


def test_empty_images_contribute_extra_points_to_cutoff_selection():
    g=[dict(position=[100.,100.],support=15,usable=15)]
    gt=[(0,(100,100),True,100)]
    # At .5, the empty image supplies a false positive and violates precision.
    # At .6 the true point remains and the empty image is correctly filtered.
    rule,_=p.choose_rule([(g,gt,np.array([.9])),(g,[],np.array([.55]))],'calibrated')
    assert rule['matched']==1 and rule['points']==1 and rule['cutoff']>.55
    assert rule['cutoff']==.9


def test_no_admissible_nonempty_rule_is_explicit_rejection():
    g=[dict(position=[100.,100.],support=15,usable=15)]
    rule,grid=p.choose_rule([(g,[],np.array([.99]))],'calibrated')
    assert rule['reject_all'] and len(grid)==19
    assert p.group_points(g,'calibrated',rule,np.array([.99]))==[]


def test_support_fraction_correctly_handles_border_denominator():
    g=[dict(position=[100.,100.],support=2,usable=5)]
    assert p.group_points(g,'support_only',dict(votes=6))==[[100.,100.]]
    assert p.group_points(g,'support_only',dict(votes=7))==[]


def test_all_selections_are_persisted_before_development_images_open(tmp_path,monkeypatch):
    import json
    source=tmp_path/'records';source.mkdir();out=tmp_path/'result'
    image=tmp_path/'images.npy';image.write_bytes(b'integration fixture')
    for e,a,s in p.RUNS:
        key=f'{e}_{a}_s{s}';folder=source/key;folder.mkdir()
        (folder/'complete.json').write_text(json.dumps(dict(model=key,exit_code=0,phases=16)))
        (folder/'provenance.json').write_text(json.dumps(dict(input_sha256={'fold1/images.npy':p.digest(image)})))
    def fitting(folder,target,input_log):
        assert not (out/'selection_complete.json').exists()
        result=dict(calibration_images=list(range(120)),selection_images=list(range(120,240)),rules={})
        p.write_json(target/'selection.json',result)
        return result
    def open_images(*args,**kwargs):
        marker=json.loads((out/'selection_complete.json').read_text())
        assert marker['development_records_opened']==0
        assert len(marker['selection_sha256'])==8
        for key,h in marker['selection_sha256'].items():assert p.digest(out/key/'selection.json')==h
        raise RuntimeError('verified development-access barrier')
    monkeypatch.setattr(p,'fit_selection',fitting)
    monkeypatch.setattr(p.np,'load',open_images)
    with pytest.raises(RuntimeError,match='verified development-access barrier'):
        p.main(['--root',str(source),'--out',str(out),'--images',str(image)])
