import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('d37',Path(__file__).parents[1]/'scripts/analyze_alignment_misses_d37.py')
d37=importlib.util.module_from_spec(spec); spec.loader.exec_module(d37)


def test_partition_priority_and_complete_boolean_domain():
    import itertools
    for flags in itertools.product((False,True),repeat=4):
        expected=next((i for i,v in enumerate(flags) if v),4)
        assert d37.partition(*flags)==d37.CATEGORIES[expected]


def test_near_point_can_be_assigned_elsewhere():
    gt=[(4,(10,10),True,50),(4,(10,12),True,50)]
    points=[(10,11)]
    assigned={int(g) for g,_ in d37.match_points(points,gt)}
    assert len(assigned)==1
    missing=({0,1}-assigned).pop()
    assert abs(gt[missing][1][1]-points[0][1])<=3
    assert d37.partition(False,False,False,True)=='near_point_assigned_elsewhere'
