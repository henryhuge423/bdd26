import importlib.util
from pathlib import Path
import numpy as np
import pytest

P = Path(__file__).parents[1] / 'scripts/analyze_visibility_supervision.py'

def module():
    assert P.exists(), 'mechanism runner is not implemented'
    spec = importlib.util.spec_from_file_location('mechanism', P)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_primary_pairs_exact_identity_and_averages_seeds():
    m = module()
    rows = []
    for seed in (19, 1):
        for image in (0, 1):
            for arm, miss in [('A1', True), ('A2', False)]:
                rows.append(dict(seed=seed, arm=arm, image=image, tissue='Breast', gt_id=7,
                                 cls=4, group='truncated', miss=miss))
    out = m.paired_primary(rows, n_boot=20)
    assert out['improvement'] == 1.0
    assert out['lo'] == out['hi'] == 1.0
    assert out['pass']
    broken = [dict(r) for r in rows]
    broken[-1]['gt_id'] = 8
    with pytest.raises(ValueError, match='identit'):
        m.paired_primary(broken, n_boot=20)

def test_guard_rejects_other_round_and_test_fold_before_read():
    m = module()
    with pytest.raises(ValueError):
        m.validate_request(Path('runs/cellvit_uni/split1'), 2)
    with pytest.raises(ValueError):
        m.validate_request(m.ROUND, 3)
    m.validate_request(m.ROUND, 2)

def test_zero_effect_not_admitted_and_seed_reversal_fails():
    m = module()
    rows = [dict(seed=s, arm=a, image=i, tissue='Breast', gt_id=i+1, cls=4,
                 group='truncated', miss=(a=='A1' if s==19 else a=='A2'))
            for s in (19,1) for a in ('A1','A2') for i in (0,1)]
    out = m.paired_primary(rows, n_boot=20)
    assert out['improvement'] == 0
    assert not out['pass']

def test_primary_is_pooled_nuclear_rate_not_mean_of_image_rates():
    m = module()
    rows=[]
    for seed in (19,1):
        for image,n in [(0,1),(1,9)]:
            for gt_id in range(1,n+1):
                for arm in ('A1','A2'):
                    rows.append(dict(seed=seed,arm=arm,image=image,tissue='Breast',
                        gt_id=gt_id,cls=4,group='truncated',
                        miss=(arm=='A1' if image==0 else arm=='A2')))
    out=m.paired_primary(rows,n_boot=100)
    assert out['improvement']==pytest.approx(-.8)
    assert out['n_objects_per_seed']==10
    assert out['lo']==-1 and out['hi']==1
    assert not out['pass']
