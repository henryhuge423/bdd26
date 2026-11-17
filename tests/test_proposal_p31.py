import pytest
import torch
from torch import nn
import train_proposal_p3 as original
import train_proposal_p26 as shared
import train_proposal_p31 as p31


@pytest.mark.parametrize('seed', [19, 1])
def test_shared_initial_convolutions_and_rng_preserved(seed):
    torch.manual_seed(seed)
    control = original.PointHead()
    initial = {n: p.detach().clone() for n, p in control.named_parameters()}
    rng = torch.get_rng_state().clone()
    intervention = p31.GroupNormalizedHead(control)
    assert torch.equal(rng, torch.get_rng_state())
    params = dict(intervention.named_parameters())
    assert all(torch.equal(params[n], value) for n, value in initial.items())
    extras = {n: p for n, p in params.items() if n not in initial}
    assert sum(p.numel() for p in extras.values()) == 2816
    assert len(intervention.norms) == 8
    assert all(torch.equal(norm.weight, torch.ones_like(norm.weight)) and
               torch.equal(norm.bias, torch.zeros_like(norm.bias)) for norm in intervention.norms)


@pytest.mark.parametrize('channels', [192, 256, 128, 64])
def test_group_normalization_matches_independent_per_image_formula(channels):
    torch.manual_seed(channels)
    x = torch.randn(2, channels, 3, 5, dtype=torch.float64)
    x[0] = x[0] * 17 + 30
    norm = nn.GroupNorm(32, channels, eps=1e-5, affine=True).double()
    grouped = x.reshape(2, 32, -1)
    mean = grouped.mean(-1, keepdim=True)
    variance = ((grouped - mean) ** 2).mean(-1, keepdim=True)
    expected = ((grouped - mean) / torch.sqrt(variance + 1e-5)).reshape_as(x)
    torch.testing.assert_close(norm(x), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(norm(x)[:1], norm(x[:1]), rtol=0, atol=0)
    norm.eval()
    torch.testing.assert_close(norm(x), expected, rtol=1e-12, atol=1e-12)
    assert not dict(norm.named_buffers())


def test_production_head_constant_initial_prediction_then_real_norm_learning():
    torch.manual_seed(1)
    head = p31.GroupNormalizedHead(original.PointHead())
    feats = [torch.randn(2, 1024, 2, 2) for _ in range(4)]
    initial = head(feats)
    assert initial.shape == (2, 256, 256)
    torch.testing.assert_close(initial.sigmoid(), torch.full_like(initial, .01), rtol=1e-6, atol=1e-8)
    target = torch.zeros_like(initial)
    target[:, 70, 90] = 1
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss = original.focal_numerator(head(feats), target) / 2
        loss.backward()
        if step == 1:
            before = [norm.weight.detach().clone() for norm in head.norms]
            assert all(torch.isfinite(norm.weight.grad).all() and norm.weight.grad.norm() > 0
                       for norm in head.norms)
        nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
        opt.step()
    decay = 1 - 3e-4 * .01
    assert all((norm.weight.detach() - prior * decay).abs().max() > 0
               for norm, prior in zip(head.norms, before))
    restored = p31.GroupNormalizedHead(original.PointHead())
    restored.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(restored(feats), head(feats), rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        original.PointHead().load_state_dict(head.state_dict(), strict=True)


@pytest.mark.parametrize('arm', ['C', 'G'])
def test_cli_preserves_original_objective_and_frozen_cohort(monkeypatch, arm):
    calls = []
    monkeypatch.setattr(shared, 'train', lambda *a, **kw: calls.append((a, kw)))
    p31.main(['--arm', arm, '--seed', '19', '--data', 'data', '--weights', 'weights', '--out', 'out'])
    (args, config, proposal, objective), options = calls[0]
    assert proposal is p31.Proposal and objective == 'C' and args.arm == arm
    assert options == {'experiment': 'P31'}
    assert config == p31.run_config(arm, 19)
    baseline = shared.run_config('C', 19)
    changed = {'experiment', 'arm', 'architecture', 'decoder_normalization',
               'normalization_eps', 'normalization_channels'}
    assert {k: v for k, v in config.items() if k not in changed} == {
        k: v for k, v in baseline.items() if k not in changed}
    assert config['trainable_encoder_blocks'] == list(range(12, 24))
    with pytest.raises(ValueError, match='configuration mismatch'):
        shared.validate_config(p31.run_config('C', 19), p31.run_config('G', 19))


def test_invalid_arms_and_feature_count():
    with pytest.raises(ValueError):
        p31.run_config('R', 1)
    with pytest.raises(ValueError):
        p31.Proposal('missing', False, 'G')
    with pytest.raises(ValueError):
        p31.GroupNormalizedHead(original.PointHead())([torch.zeros(1, 1024, 2, 2)])
