"""Verify original control parity and production GroupNorm learning on V100."""
import gc
import math
from pathlib import Path
import sys
import numpy as np
import torch
import train_proposal_p3 as original
import train_proposal_p26 as shared
import train_proposal_p31 as p31
from verify_p26_control import hashes, reset, rng_hashes, run


def witness_normalization(model, fold, ids, seed):
    opt = shared.make_optimizer(model, 'C')
    frozen = shared.frozen_hashes(model)
    assert {id(p) for g in opt.param_groups for p in g['params']} == {
        id(p) for p in model.parameters() if p.requires_grad}
    watched = {f'norm{i}': norm.weight for i, norm in enumerate(model.head.norms)}
    watched.update({f'block{i}': model.encoder.blocks[i].attn.qkv.weight for i in (12, 23)})
    measurements = []

    def independent_formula(norm, inputs, output):
        x = inputs[0].float()
        grouped = x.reshape(len(x), 32, -1)
        mean = grouped.mean(-1, keepdim=True)
        variance = (grouped - mean).square().mean(-1, keepdim=True)
        expected = ((grouped - mean) / (variance + norm.eps).sqrt()).reshape_as(x)
        expected = expected * norm.weight.float()[None, :, None, None] + norm.bias.float()[None, :, None, None]
        error = float((output.float() - expected).abs().max())
        assert torch.isfinite(output).all() and math.isfinite(error) and error < .04
        measurements.append(dict(max_formula_error=error, min_input_variance=float(variance.min()),
                                 output_dtype=str(output.dtype)))

    handles = [norm.register_forward_hook(independent_formula) for norm in model.head.norms]
    model.train()
    assert not model.encoder.training
    try:
        for start in (0, 16):
            selected = ids[start:start+16]
            count = sum(len(fold.gt[i]) for i in selected)
            opt.zero_grad(set_to_none=True)
            def losses():
                for offset in range(0, 16, 4):
                    x, y, _ = fold.batch(selected[offset:offset+4], seed, 0, True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(x.cuda())
                        assert torch.is_autocast_enabled('cuda')
                    assert logits.dtype == torch.float32
                    yield shared.loss_for_microbatch(logits, y.cuda(), count, 'C')
            focal, reg = shared.backward_effective_batch(model, None, losses())
            assert math.isfinite(focal) and focal > 0 and reg == 0
            if start == 16:
                gradients = {k: float(p.grad.norm()) for k, p in watched.items()}
                before = {k: p.detach().clone() for k, p in watched.items()}
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          1, error_if_nonfinite=True)
            opt.step()
    finally:
        for handle in handles:
            handle.remove()
    updates = {}
    for k, p in watched.items():
        group = opt.param_groups[0 if k.startswith('norm') else 1]
        decay = 1 - group['lr'] * group['weight_decay']
        updates[k] = float((p.detach() - before[k] * decay).abs().max())
    assert all(math.isfinite(v) and v > 0 for v in [*gradients.values(), *updates.values()])
    assert len(measurements) == 64
    assert shared.frozen_hashes(model) == frozen
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    return dict(gradients=gradients, beyond_decay=updates, normalization_calls=len(measurements),
                max_formula_error=max(r['max_formula_error'] for r in measurements),
                min_input_variance=min(r['min_input_variance'] for r in measurements),
                frozen_tensors_unchanged=True, exact_optimizer_membership=True)


def main():
    weights, data, out = sys.argv[1:]
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and original.sha(weights) == original.WEIGHT_SHA
    for name, digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):
            assert original.sha(Path(data)/name) == digest
    assert original.sha(Path(data)/'fold1/tissue.npy') == shared.TISSUE_SHA
    fold = original.Fold(data, 1)
    fit, audit = shared.split_indices(fold.tissue, [any(g[0]==4 and g[2] for g in row) for row in fold.gt])
    shared.validate_cohort(fit, audit)
    ids_fit = shared.smoke_ids(fit, fold.gt)
    assert len(ids_fit) == 134 and not set(ids_fit).intersection(audit)
    seeds = {}
    for seed in (19, 1):
        ids = np.random.default_rng(np.random.SeedSequence([seed, 0, 123])).permutation(ids_fit).tolist()
        initial, counts, traces, predictions = {}, {}, {}, {}
        for arm in ('P3', 'C', 'G'):
            reset(seed)
            model = original.Proposal(weights, True) if arm == 'P3' else p31.Proposal(weights, True, arm)
            initial[arm] = hashes((n,v) for n,v in model.state_dict().items() if '.norms.' not in n), rng_hashes()
            counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.cuda().eval()
            x, _, _ = fold.batch(ids[:2], seed, 0, False)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                predictions[arm] = model(x.cuda()).float().cpu()
            if arm == 'G':
                normalization = witness_normalization(model, fold, ids, seed)
            else:
                traces[arm] = run(model, fold, ids, seed, None if arm == 'P3' else 'C')
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assert initial['P3'] == initial['C'] == initial['G']
        assert counts['P3'] == counts['C'] and counts['G'] == counts['C'] + 2816
        assert torch.equal(predictions['P3'], predictions['C']) and torch.equal(predictions['C'], predictions['G'])
        assert traces['P3'] == traces['C']
        seeds[str(seed)] = dict(shared_initial_state_equal=True, initial_rng_equal=True,
                                initial_prediction_equal=True, control_two_step_parity=True,
                                trainable_parameters=counts, normalization=normalization)
    shared.dump(out, dict(status='PASS', seeds=seeds, smoke_fit=ids_fit, fold1_only=True))


if __name__ == '__main__':
    main()
