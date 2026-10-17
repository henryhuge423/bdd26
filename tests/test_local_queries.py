import importlib.util
from pathlib import Path
from copy import deepcopy
import torch
import pytest


def api():
    path = Path(__file__).with_name('probe_local_queries.py')
    if not path.is_file():
        path = Path(__file__).resolve().parents[1]/'scripts'/'probe_local_queries.py'
    assert path.is_file(), 'E9 local-query probe is absent'
    spec = importlib.util.spec_from_file_location('e9probe', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_global_branch_reproduces_frozen_head_and_preserves_input():
    from nucseg.setmask.model import QueryMaskHead
    head = QueryMaskHead(channels=4, dim=16, mask_dim=4, queries=512, layers=1, heads=4).eval()
    x = torch.randn(1, 4, 16, 16)
    before = x.clone()
    expected = head(x)
    actual = api().query_forward(head, x, False)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert torch.equal(x, before)


def test_local_initial_content_uses_row_major_spatial_cells():
    from nucseg.setmask.model import QueryMaskHead
    head = QueryMaskHead(channels=1, dim=4, mask_dim=1, queries=512, layers=1, heads=1).eval()
    with torch.no_grad():
        head.memory_projection.weight.fill_(1)
        head.memory_projection.bias.zero_()
        head.query_content.weight.zero_()
    x = torch.arange(512.).reshape(1,1,16,32)
    got = api().initial_queries(head, x, True)
    assert got.shape == (1,512,4)
    assert got[0,0].tolist() == [0.,0.,0.,0.]
    assert got[0,31].tolist() == [31.,31.,31.,31.]
    assert got[0,32].tolist() == [32.,32.,32.,32.]
    assert got[0,511].tolist() == [511.,511.,511.,511.]


def test_local_training_changes_head_without_mutating_frozen_features():
    from nucseg.setmask.model import QueryMaskHead
    head = QueryMaskHead(channels=4, dim=16, mask_dim=4, queries=512, layers=1, heads=4).eval()
    fitted = deepcopy(head)
    x = torch.randn(1,4,16,16)
    saved = x.clone()
    out = api().query_forward(fitted,x,True)
    sum(v.square().mean() for v in out.values()).backward()
    assert fitted.memory_projection.weight.grad.abs().sum() > 0
    opt = torch.optim.AdamW(fitted.parameters(),lr=.001)
    opt.step()
    assert torch.equal(x,saved)
    assert any(not torch.equal(v,head.state_dict()[k]) for k,v in fitted.state_dict().items())


def test_cohort_has_dead_and_negative_coverage_without_overlap():
    groups = api().cohorts([0,1,2,3], [4,5,6,7,8,9])
    assert len(groups['fit']) == 4 and len(groups['audit']) == 4
    assert not set(groups['fit']) & set(groups['audit'])
    for ids in groups.values():
        assert len(set(ids)&{0,1,2,3}) == 2
        assert len(set(ids)&{4,5,6,7,8,9}) == 2


def test_gate_rejects_recall_gain_at_false_positive_cost_and_one_seed_failure():
    m=api()
    base={'dead_recall':.2,'recall':.5,'precision':.6,'dead_typed_f1':.2,'fit_recall':.9}
    changed={**base,'dead_recall':.3,'recall':.55,'dead_typed_f1':.25}
    rows={s:{'G':base,'L':changed} for s in ('19','1')}
    assert m.allocation(rows)['promising']
    rows['1']['L']={**changed,'precision':.59}
    assert not m.allocation(rows)['promising']
    rows['1']['L']={**changed,'dead_recall':.24}
    assert not m.allocation(rows)['promising']


def test_empty_negative_image_has_empty_boolean_interior_selection():
    import numpy as np
    selected=api().interior_dead_mask(np.zeros((8,8),np.int64),torch.tensor([],dtype=torch.long),np.array([],dtype=np.int64))
    assert selected.dtype==np.bool_ and selected.shape==(0,)


def test_interior_dead_selection_excludes_border_and_other_classes():
    import numpy as np
    inst=np.zeros((8,8),np.int64)
    inst[0,0]=1;inst[3,3]=2;inst[5,5]=3
    selected=api().interior_dead_mask(inst,torch.tensor([1,2,3]),np.array([4,4,1]))
    assert selected.tolist()==[False,True,False]
