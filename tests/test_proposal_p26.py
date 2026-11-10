import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn


def p26():
    assert importlib.util.find_spec('train_proposal_p26') is not None, 'P26 implementation missing'
    import train_proposal_p26
    return train_proposal_p26


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(24)])
        self.norm = nn.LayerNorm(2)
        self.embed = nn.Linear(2, 2)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(2, 1)

    def adapted_state(self):
        return {n: q.detach().clone() for n, q in self.named_parameters() if q.requires_grad}


def model():
    import train_proposal_p25 as p25
    torch.manual_seed(19)
    m = Tiny()
    p25.configure_encoder_depth(m.encoder, 'C')
    return m


def test_initial_anchor_zero_then_exact_sum_and_analytic_gradients():
    p = p26(); m = model(); anchor = p.snapshot_anchor(m)
    assert float(p.anchor_penalty(m, anchor)) == 0
    assert set(anchor) == {n for n, q in m.encoder.named_parameters() if q.requires_grad}
    assert all(not q.requires_grad and q.grad_fn is None for q in anchor.values())
    with torch.no_grad():
        for q in m.encoder.parameters():
            if q.requires_grad: q.add_(0.2)
        m.head.weight.add_(3)
    penalty = p.anchor_penalty(m, anchor)
    count = sum(q.numel() for q in m.encoder.parameters() if q.requires_grad)
    assert float(penalty) == pytest.approx(.01 / 2 * count * .2**2, rel=1e-5)
    penalty.backward()
    for name, q in m.encoder.named_parameters():
        if q.requires_grad:
            assert torch.allclose(q.grad, .01 * (q.detach() - anchor[name]))
        else:
            assert q.grad is None
    assert all(q.grad is None for q in m.head.parameters())


def test_anchors_are_fixed_non_optimizer_copies_and_follow_no_rng():
    p = p26(); m = model(); before = torch.get_rng_state().clone()
    anchor = p.snapshot_anchor(m)
    assert torch.equal(before, torch.get_rng_state())
    opt = p.make_optimizer(m, 'R')
    optimizer_ids = {id(q) for g in opt.param_groups for q in g['params']}
    assert optimizer_ids == {id(q) for q in m.parameters() if q.requires_grad}
    assert not optimizer_ids.intersection(map(id, anchor.values()))
    saved = {k: v.clone() for k, v in anchor.items()}
    with torch.no_grad(): m.encoder.blocks[12].weight.add_(1)
    assert all(torch.equal(saved[k], v) for k, v in anchor.items())


@pytest.mark.parametrize('size', [16, 6])
def test_one_anchor_backward_per_effective_batch_including_tail(size):
    p = p26(); m = model(); anchor = p.snapshot_anchor(m)
    with torch.no_grad(): m.encoder.blocks[12].weight.add_(.25)
    q = m.encoder.blocks[12].weight
    def losses():
        for _ in range(math.ceil(size / 4)):
            yield q.sum() / math.ceil(size / 4)
    focal, reg = p.backward_effective_batch(m, anchor, losses())
    assert focal == pytest.approx(float(q.detach().sum()))
    assert reg == pytest.approx(float(p.anchor_penalty(m, anchor)))
    assert torch.allclose(q.grad, torch.ones_like(q) + .01 * (q.detach() - anchor['blocks.12.weight']))

def test_original_c_adamw_two_updates_match_independent_reference():
    p = p26(); a = model(); b = copy.deepcopy(a)
    oa = torch.optim.AdamW([dict(params=list(a.head.parameters()),lr=3e-4,initial_lr=3e-4),
        dict(params=[q for q in a.encoder.parameters() if q.requires_grad],lr=1e-5,initial_lr=1e-5)],betas=(.9,.999),weight_decay=.01)
    ob = p.make_optimizer(b, 'C')
    for _ in range(2):
        for m, opt in ((a, oa), (b, ob)):
            opt.zero_grad(set_to_none=True)
            sum(q.square().sum() for q in m.parameters() if q.requires_grad).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1, error_if_nonfinite=True)
            opt.step()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert oa.state_dict()['param_groups'] == ob.state_dict()['param_groups']
    for key, state in oa.state_dict()['state'].items():
        for field, value in state.items():
            assert torch.equal(value, ob.state_dict()['state'][key][field])


def test_config_binds_anchor_and_rejects_cross_arm_seed_lambda_source():
    p = p26(); c = p.run_config('C', 19); r = p.run_config('R', 19)
    assert r['anchor_lambda'] == .01 and r['anchor_scope'] == 'trainable_encoder'
    assert r['anchor_weights_sha256'] == p.WEIGHT_SHA
    assert c['anchor_lambda'] == 0
    p.validate_config(r, copy.deepcopy(r))
    for changed in (c, p.run_config('R', 1), {**r, 'anchor_lambda': .02}, {**r, 'anchor_weights_sha256': 'wrong'}):
        with pytest.raises(ValueError): p.validate_config(r, changed)


def test_invalid_cli_and_output_refuse_resume_or_overwrite(tmp_path):
    p = p26(); args = ['--data','d','--weights','w','--out',str(tmp_path / 'out'),'--arm','R']
    assert p.parse_args(args).arm == 'R'
    for suffix in (['--resume','x'], ['--arm','D'], ['--microbatch','8']):
        with pytest.raises(SystemExit): p.parse_args(args + suffix)
    p.prepare_output(tmp_path / 'out')
    marker = tmp_path / 'out' / 'keep'; marker.write_text('unchanged')
    with pytest.raises(FileExistsError): p.prepare_output(tmp_path / 'out')
    assert marker.read_text() == 'unchanged'


def test_loss_log_uses_same_image_weighting_for_components_and_total():
    p = p26(); totals = {'focal': 0., 'reg': 0., 'total': 0., 'images': 0}
    p.accumulate_epoch_loss(totals, 2., .5, 16)
    p.accumulate_epoch_loss(totals, 4., 1., 6)
    assert totals['images'] == 22
    assert totals['focal'] == 56 and totals['reg'] == 14
    assert totals['total'] == pytest.approx(totals['focal'] + totals['reg'])


def test_final_payload_has_no_anchor_tensor():
    p = p26(); m = model(); anchor = p.snapshot_anchor(m)
    payload = p.final_payload(m, p.run_config('R', 19))
    assert set(payload) == {'model', 'weights_sha256', 'config'}
    assert payload['weights_sha256'] == p.WEIGHT_SHA
    assert 'anchor' not in payload and not any(id(v) in {id(q) for q in anchor.values()} for v in payload['model'].values())


def test_gpu_verifier_executes_production_effective_batch(monkeypatch):
    p = p26()
    import verify_p26_control as verify
    m = model(); anchor = p.snapshot_anchor(m)
    with torch.no_grad(): m.encoder.blocks[12].weight.add_(.25)
    q = m.encoder.blocks[12].weight
    calls = []
    real = p.backward_effective_batch
    def tracked(model, anchor, losses):
        calls.append((model, anchor))
        return real(model, anchor, losses)
    monkeypatch.setattr(p, 'backward_effective_batch', tracked)
    focal, reg = verify.backward_candidate(m, anchor, (q.sum() / 2 for _ in range(2)))
    assert calls == [(m, anchor)]
    assert focal == pytest.approx(float(q.detach().sum())) and reg > 0
    assert torch.allclose(q.grad, torch.ones_like(q) + .01 * (q.detach() - anchor['blocks.12.weight']))


