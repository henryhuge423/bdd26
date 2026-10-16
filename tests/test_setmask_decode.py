import importlib
import importlib.util

import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec('nucseg.setmask.decode') is not None,'Streamed decode is absent'
    return importlib.import_module('nucseg.setmask.decode')


def fixture():
    return {'class_logits':torch.tensor([[[-9.,9.,-9.,-9.,-9.,-9.],[-9.,-9.,9.,-9.,-9.,-9.]]]),
            'mask_embed':torch.ones(1,2,1),'mask_features':torch.ones(1,1,3,3)}


def test_exact_overlap_tie_uses_lower_query_id_across_chunks():
    p=fixture();a=api().decode_set(p,chunk_size=1);b=api().decode_set(p,chunk_size=16)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert (a[0]==1).all() and (a[1]==1).all()
    assert a[0].dtype==np.int32 and a[1].dtype==np.uint8


def test_no_object_and_low_confidence_queries_are_rejected():
    p=fixture();p['class_logits'][0,0]=torch.tensor([9.,-9.,-9.,-9.,-9.,-9.])
    p['class_logits'][0,1].zero_()
    inst,typ=api().decode_set(p)
    assert not inst.any() and not typ.any()


def test_single_pixel_survives_without_area_filter():
    p=fixture();p['mask_features'].fill_(-10);p['mask_features'][0,0,1,2]=10
    inst,typ=api().decode_set(p)
    assert np.count_nonzero(inst)==1 and inst[0,1,2]==1 and typ[0,1,2]==1


def test_disconnected_query_support_is_not_silently_split():
    p=fixture();p['mask_features'].fill_(-10)
    p['mask_features'][0,0,0,0]=10;p['mask_features'][0,0,2,2]=10
    inst,_=api().decode_set(p)
    assert inst[0,0,0]==inst[0,2,2]==1


def test_batch_predictions_do_not_interfere():
    p=fixture();p={k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in p.items()}
    p['mask_features'][1].neg_()
    inst,typ=api().decode_set(p)
    assert (inst[0]==1).all() and not inst[1].any() and not typ[1].any()


def test_nonfinite_logits_fail_instead_of_empty_prediction():
    p=fixture();p['class_logits'][0,0,0]=float('nan')
    with pytest.raises(ValueError):api().decode_set(p)
