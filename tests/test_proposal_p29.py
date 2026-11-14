from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train_proposal_p26 as shared
import train_proposal_p29 as p29


def make_encoder():
    encoder = nn.Module()
    encoder.patch_embed = nn.Linear(4, 4)
    encoder.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(24))
    encoder.norm = nn.LayerNorm(4)
    return encoder


@pytest.mark.parametrize('arm,expected', [('C', range(12, 24)), ('B', tuple(range(6)) + tuple(range(18, 24)))])
def test_equal_budget_and_gradient_through_frozen_suffix(arm, expected):
    torch.manual_seed(17)
    model = nn.Module()
    model.encoder = make_encoder()
    model.head = nn.Linear(4, 1)
    p29.configure_encoder_depth(model.encoder, arm)
    assert not model.encoder.training
    assert not any(p.requires_grad for p in model.encoder.patch_embed.parameters())
    assert not any(p.requires_grad for p in model.encoder.norm.parameters())
    assert [i for i, b in enumerate(model.encoder.blocks) if b.weight.requires_grad] == list(expected)
    assert sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) == 240
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = shared.make_optimizer(model, 'C')
    assert [g['lr'] for g in opt.param_groups] == [3e-4, 1e-5]
    assert {id(p) for g in opt.param_groups for p in g['params']} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    x = model.encoder.patch_embed(torch.randn(3, 4))
    for block in model.encoder.blocks:
        x = x + block(x)
    model.head(model.encoder.norm(x)).square().sum().backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p.grad).all() and p.grad.abs().max() > 0, n
        else:
            assert p.grad is None, n
    opt.step()
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]) is (not p.requires_grad), n


def test_equal_config_and_strict_arm_reload():
    c, e = (p29.run_config(a, 1) for a in p29.ARMS)
    differences = {k for k in c if c[k] != e[k]}
    assert differences == {'arm', 'architecture', 'trainable_encoder_blocks'}
    with pytest.raises(ValueError, match='configuration mismatch'):
        shared.validate_config(c, e)
    model = SimpleNamespace(encoder=make_encoder())
    p29.configure_encoder_depth(model.encoder, 'B')
    state = {n: v for n, v in model.encoder.state_dict().items()
             if n.startswith(tuple(f'blocks.{i}.' for i in range(12, 24)))}
    with pytest.raises(ValueError, match='trainable key mismatch'):
        shared.load_adapted(model.encoder, state)


@pytest.mark.parametrize('arm', ['C', 'B'])
def test_cli_dispatches_frozen_config_to_shared_runner(monkeypatch, arm):
    calls = []
    monkeypatch.setattr(shared, 'train', lambda *a, **kw: calls.append((a, kw)))
    argv = ['--arm', arm, '--seed', '1', '--data', 'data', '--weights', 'weights', '--out', 'out']
    p29.main(argv)
    (args, config, proposal, objective), options = calls[0]
    assert args.arm == arm and args.seed == 1
    assert config == p29.run_config(arm, 1)
    assert proposal is p29.Proposal and objective == 'C' and options == {'experiment': 'P29'}
    with pytest.raises(SystemExit):
        shared.parse_args(['--arm', 'B', *argv[2:]])


@pytest.mark.parametrize('arm,depth', [('B',23), ('unknown',24)])
def test_invalid_depth_or_arm_fails_before_optimizer(arm, depth):
    encoder = make_encoder()
    encoder.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(depth))
    with pytest.raises(ValueError, match='24-block'):
        p29.configure_encoder_depth(encoder, arm)
