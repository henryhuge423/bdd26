import importlib.util
import math
from pathlib import Path

import pytest
import torch

P = Path(__file__).resolve().parents[1] / 'scripts' / 'probe_query_alignment.py'
spec = importlib.util.spec_from_file_location('probe_query_alignment', P)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def loss_function():
    assert callable(getattr(probe, 'alignment_loss', None)), 'alignment loss is not implemented'
    return probe.alignment_loss


def test_balanced_loss_uses_equal_group_means_not_query_counts():
    loss = loss_function()
    logits = torch.tensor([[2., 0.]] * 5 + [[0., 0.]])
    labels = torch.tensor([0] * 5 + [1])
    expected = math.log1p(math.exp(-2)) + math.log(2)
    assert float(loss(logits, labels, True)) == pytest.approx(expected, abs=1e-6)
    original = 2 * (.5 * math.log1p(math.exp(-2)) + math.log(2)) / 1.5
    assert float(loss(logits, labels, False)) == pytest.approx(original, abs=1e-6)


def test_balanced_loss_is_invariant_to_repeating_background_examples():
    loss = loss_function()
    pair = torch.tensor([[2., 0.], [0., 0.]])
    many = torch.tensor([[2., 0.]] * 20 + [[0., 0.]])
    assert loss(pair, torch.tensor([0, 1]), True) == pytest.approx(
        float(loss(many, torch.tensor([0] * 20 + [1]), True)))
    assert abs(float(loss(pair, torch.tensor([0, 1]), False)) -
               float(loss(many, torch.tensor([0] * 20 + [1]), False))) > .1


@pytest.mark.parametrize('labels', [torch.zeros(4, dtype=torch.long),
                                  torch.full((4,), 2, dtype=torch.long)])
def test_loss_rejects_absent_foreground_or_out_of_range_labels(labels):
    loss = loss_function()
    with pytest.raises(ValueError):
        loss(torch.zeros(4, 2), labels, True)


def test_fitting_updates_only_copy_of_classifier_and_detaches_features():
    assert callable(getattr(probe, 'fit_classifier', None)), 'classifier fit is not implemented'
    head = torch.nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    features = torch.tensor([[1., 0.]] * 5 + [[0., 1.]], requires_grad=True)
    labels = torch.tensor([0] * 5 + [1])
    before = {k: v.clone() for k, v in head.state_dict().items()}
    fitted, rows = probe.fit_classifier(head, features, labels, True, steps=100, lr=.05)
    assert all(torch.equal(v, before[k]) for k, v in head.state_dict().items())
    assert features.grad is None
    assert torch.equal(rows[0]['initial_logits'], torch.zeros(6, 2))
    assert torch.equal(fitted(features.detach()).argmax(-1), labels)
    assert rows[-1]['loss'] < rows[0]['loss'] * .1


def test_allocation_requires_audit_gain_in_both_seeds_and_fit_accuracy():
    assert callable(getattr(probe, 'allocation', None)), 'allocation is not implemented'
    results = {str(s): {'A0': {'audit': {'best_final_recall50': .20, 'foreground_accuracy': .50}},
                       'A2': {'fit': {'foreground_accuracy': .95},
                              'audit': {'best_final_recall50': .27, 'foreground_accuracy': .51}}}
               for s in (19, 1)}
    assert probe.allocation(results)['promising'] is True
    results['1']['A2']['audit']['best_final_recall50'] = .24
    assert probe.allocation(results)['promising'] is False
    results['1']['A2']['audit']['best_final_recall50'] = .27
    results['19']['A2']['fit']['foreground_accuracy'] = .89
    assert probe.allocation(results)['promising'] is False


def test_fit_preserves_batched_query_shape_until_after_linear_projection():
    assert callable(getattr(probe, 'fit_classifier', None)), 'classifier fit is not implemented'
    head = torch.nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    features = torch.tensor([[[1., 0.]] * 3, [[1., 0.], [1., 0.], [0., 1.]]],
                            requires_grad=True)
    labels = torch.tensor([[0, 0, 0], [0, 0, 1]])
    fitted, rows = probe.fit_classifier(head, features, labels, True, steps=100, lr=.05)
    assert rows[0]['initial_logits'].shape == (2, 3, 2)
    assert torch.equal(fitted(features.detach()).argmax(-1), labels)
    assert features.grad is None
