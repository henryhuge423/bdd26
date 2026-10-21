"""Protect P5's matched-capacity spatial comparison and reload contract."""
import importlib
import importlib.util
import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
torch.set_num_threads(2)


def p5():
    assert importlib.util.find_spec('train_proposal_p5') is not None, 'P5 trainer is not implemented'
    return importlib.import_module('train_proposal_p5')


def test_matched_initialization_and_capacity():
    m=p5()
    torch.manual_seed(19);low=m.PointHead(native=False)
    torch.manual_seed(19);native=m.PointHead(native=True)
    assert low.state_dict().keys()==native.state_dict().keys()
    assert all(torch.equal(v,native.state_dict()[k]) for k,v in low.state_dict().items())
    assert sum(p.numel() for p in low.parameters())==sum(p.numel() for p in native.parameters())
    assert low.refine.weight.shape==(64,64,3,3)


@pytest.mark.parametrize('native,expected',[(False,128),(True,256)])
def test_refinement_grid_and_native_output(native,expected):
    head=p5().PointHead(native=native).eval()
    sizes=[]
    handle=head.refine.register_forward_pre_hook(lambda module,args:sizes.append(tuple(args[0].shape[-2:])))
    with torch.no_grad():result=head([torch.zeros(1,1024,16,16) for _ in range(4)])
    handle.remove()
    assert sizes==[(expected,expected)]
    assert result.shape==(1,256,256)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize('native',[False,True])
def test_refinement_receives_finite_gradient_after_output_updates(native):
    head=p5().PointHead(native=native)
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4)
    features=[torch.randn(1,1024,16,16) for _ in range(4)]
    before=head.refine.weight.detach().clone()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        logits=head(features)
        loss=(logits-1).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        opt.step()
    assert head.refine.weight.grad.abs().sum()>0
    assert not torch.equal(head.refine.weight,before)


def test_missing_refinement_checkpoint_is_rejected():
    m=p5();head=m.PointHead(native=True)
    state=dict(head.state_dict());state.pop('refine.weight')
    with pytest.raises(ValueError,match='trainable key mismatch'):m.load_adapted(head,state)


def test_refinement_checkpoint_restores_predictions():
    m=p5();head=m.PointHead(native=True).eval()
    with torch.no_grad():head.out.weight.fill_(.01)
    fresh=m.PointHead(native=True).eval();m.load_adapted(fresh,head.state_dict())
    features=[torch.randn(1,1024,16,16) for _ in range(4)]
    with torch.no_grad():assert torch.equal(head(features),fresh(features))


def test_original_scoring_and_cohorts_are_reused():
    m=p5();original=importlib.import_module('proposal_p3_utils')
    for name in ('split_indices','gt_objects','gaussian_target','augment_pair','focal_numerator','decode_peaks','score_points','aggregate','admission'):
        assert getattr(m,name) is getattr(original,name)


@pytest.mark.parametrize('bad',[0.,float('nan'),float('inf')])
def test_smoke_rejects_nonlearning_or_nonfinite_gradient(bad):
    m=p5()
    assert hasattr(m,'require_positive_gradients'), 'Actual gradient gate is missing'
    with pytest.raises(ValueError):m.require_positive_gradients(dict(encoder=bad,refinement=1.,output=1.))


def test_same_cohort_counts_with_changed_membership_are_rejected():
    m=p5()
    assert hasattr(m,'validate_cohort'), 'Exact cohort identity gate is missing'
    with pytest.raises(ValueError):m.validate_cohort(list(range(2134)),list(range(2134,2656)))


def test_changed_submission_script_is_rejected(tmp_path):
    m=p5()
    assert hasattr(m,'verify_hashes'), 'Script identity gate is missing'
    p=tmp_path/'train_array.sbatch';p.write_bytes(b'approved script')
    expected={p.name:hashlib.sha256(b'approved script').hexdigest()}
    m.verify_hashes(tmp_path,expected)
    p.write_bytes(b'changed resources')
    with pytest.raises(ValueError):m.verify_hashes(tmp_path,expected)


@pytest.mark.parametrize('native,logits',[(False,'torch.float32'),(True,'torch.bfloat16')])
def test_cuda_autocast_trace_accepts_documented_interpolation_dtype(native,logits):
    m=p5();assert hasattr(m,'validate_precision_trace'), 'CUDA AMP policy trace gate is missing'
    m.validate_precision_trace(native,dict(enabled=True,amp_dtype='torch.bfloat16',layers=dict(encoder='torch.bfloat16',refinement='torch.bfloat16',output='torch.bfloat16'),logits_dtype=logits))


@pytest.mark.parametrize('enabled,encoder',[(False,'torch.bfloat16'),(True,'torch.float32')])
def test_cuda_autocast_trace_rejects_disabled_or_fp32_linear_path(enabled,encoder):
    m=p5();assert hasattr(m,'validate_precision_trace'), 'CUDA AMP policy trace gate is missing'
    with pytest.raises(ValueError):m.validate_precision_trace(False,dict(enabled=enabled,amp_dtype='torch.bfloat16',layers=dict(encoder=encoder,refinement='torch.bfloat16',output='torch.bfloat16'),logits_dtype='torch.float32'))
