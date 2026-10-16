import importlib
import importlib.util

import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec('nucseg.setmask.matching') is not None,'Set matching/loss is absent'
    return importlib.import_module('nucseg.setmask.matching')


def example():
    inst=torch.zeros(1,8,8,dtype=torch.long)
    inst[:,:,:4]=17;inst[:,:,4:]=2
    typ=torch.where(inst==17,1,4)
    left=torch.where(torch.arange(8)[None,:]<4,10.,-10.).expand(8,8)
    features=torch.stack((left,-left))[None].clone().requires_grad_()
    pred={'class_logits':torch.tensor([[[-8.,8.,-8.,-8.,-8.,-8.],
                                        [-8.,-8.,-8.,-8.,8.,-8.]]],requires_grad=True),
          'mask_embed':torch.eye(2)[None].requires_grad_(),'mask_features':features,
          'tissue':torch.zeros(1,19,requires_grad=True)}
    batch={'inst_map':inst,'tp_map':typ,'tissue':torch.tensor([0])}
    return pred,batch


def test_single_pixel_retains_type_id_and_sampling_point():
    m=api();inst=torch.zeros(1,32,32,dtype=torch.long);inst[0,7,11]=17
    t=m.make_targets(inst,(inst>0).long()*4)[0]
    assert t['ids'].tolist()==[17] and t['classes'].tolist()==[4]
    assert t['points'].tolist()==[[7,11]]
    points=m.matching_points(t,32,32)
    assert ((points==torch.tensor([7,11])).all(1)).any()


def test_disconnected_centroid_tie_selects_actual_row_major_pixel():
    m=api();inst=torch.zeros(1,8,8,dtype=torch.long)
    inst[0,0,0]=2;inst[0,0,2]=2
    t=m.make_targets(inst,(inst>0).long()*4)[0]
    assert t['ids'].tolist()==[2] and t['points'].tolist()==[[0,0]]


def test_capacity_512_succeeds_513_fails_without_truncation():
    m=api();inst=torch.zeros(1,32,32,dtype=torch.long)
    inst.reshape(-1)[:512]=torch.arange(1,513)
    assert len(m.make_targets(inst,(inst>0).long()*4)[0]['ids'])==512
    inst.reshape(-1)[512]=513
    with pytest.raises(ValueError):m.make_targets(inst,(inst>0).long()*4)


@pytest.mark.parametrize('bad_type',[0,6,-1])
def test_invalid_instance_class_is_rejected(bad_type):
    m=api();inst=torch.zeros(1,8,8,dtype=torch.long);inst[0,2,3]=4
    typ=torch.zeros_like(inst);typ[inst>0]=bad_type
    with pytest.raises(ValueError):m.make_targets(inst,typ)


def test_perfect_masks_match_noncontiguous_ids_correctly():
    m=api();pred,batch=example();targets=m.make_targets(batch['inst_map'],batch['tp_map'])
    qi,ti=m.match_queries(pred['class_logits'],pred['mask_embed'],pred['mask_features'],targets)[0]
    assert qi.tolist()==[0,1] and ti.tolist()==[1,0]


def test_matching_respects_target_permutation():
    m=api();pred,batch=example();targets=m.make_targets(batch['inst_map'],batch['tp_map'])
    target=targets[0]
    reordered={**target,**{k:target[k].flip(0) for k in ('ids','classes','points')}}
    qi,ti=m.match_queries(pred['class_logits'],pred['mask_embed'],pred['mask_features'],[reordered])[0]
    assert qi.tolist()==[0,1] and reordered['ids'][ti].tolist()==[17,2]


def test_full_resolution_loss_is_finite_and_backpropagates():
    m=api();pred,batch=example();loss,terms=m.setmask_loss(pred,batch)
    assert torch.isfinite(loss) and set(terms)=={'query_ce','mask_bce','mask_dice','tissue_ce'}
    loss.backward()
    for key in ('class_logits','mask_embed','mask_features','tissue'):
        assert pred[key].grad is not None and torch.isfinite(pred[key].grad).all()
    assert pred['mask_features'].grad.abs().sum()>0


def test_loss_invariant_to_query_permutation_and_better_for_correct_masks():
    m=api();pred,batch=example();want=m.setmask_loss(pred,batch)[0]
    perm={**pred,'class_logits':pred['class_logits'].flip(1),'mask_embed':pred['mask_embed'].flip(1)}
    assert torch.allclose(m.setmask_loss(perm,batch)[0],want,atol=1e-6)
    wrong={**pred,'mask_features':torch.zeros_like(pred['mask_features'])}
    assert m.setmask_loss(wrong,batch)[0]>want


def test_empty_targets_have_zero_mask_loss_and_valid_backward():
    m=api();pred,batch=example();batch['inst_map'].zero_();batch['tp_map'].zero_()
    loss,terms=m.setmask_loss(pred,batch)
    assert terms['mask_bce']==0 and terms['mask_dice']==0
    loss.backward()
    for v in pred.values():assert v.grad is not None and torch.isfinite(v.grad).all()


def test_mask_loss_uses_all_pixels_not_only_matching_samples():
    m=api();pred,batch=example()
    base=m.setmask_loss(pred,batch)[0]
    # A pixel outside the sampled centres changes full native mask loss.
    changed={**pred,'mask_features':pred['mask_features'].detach().clone()}
    changed['mask_features'][0,:,1,1].neg_()
    assert m.setmask_loss(changed,batch)[0]>base
