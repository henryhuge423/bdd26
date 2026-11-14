"""Verify P29 on the same V100 using the original fit-only examples."""

import gc
import math
from pathlib import Path
import sys

import numpy as np
import torch

import train_proposal_p3 as original
import train_proposal_p26 as shared
import train_proposal_p29 as p29
from verify_p26_control import hashes, reset, rng_hashes, run


def witness_boundary(model, fold, ids, seed):
    opt = shared.make_optimizer(model, 'C')
    assert [g['lr'] for g in opt.param_groups] == [3e-4, 1e-5]
    frozen = shared.frozen_hashes(model)
    model.train()
    assert not model.encoder.training
    watched = {str(i): model.encoder.blocks[i].attn.qkv.weight for i in (0, 23)}

    for start in (0, 16):
        selected = ids[start:start + 16]
        assert len(selected) == 16
        count = sum(len(fold.gt[i]) for i in selected)
        opt.zero_grad(set_to_none=True)

        def losses():
            for offset in range(0, 16, 4):
                x, y, _ = fold.batch(selected[offset:offset + 4], seed, 0, True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x.cuda())
                    assert torch.is_autocast_enabled('cuda')
                assert logits.dtype == torch.float32
                yield shared.loss_for_microbatch(logits, y.cuda(), count, 'C')

        focal, regularizer = shared.backward_effective_batch(model, None, losses())
        assert math.isfinite(focal) and focal > 0 and regularizer == 0
        params = [p for p in model.parameters() if p.requires_grad]
        if start == 16:
            grads = {k: float(p.grad.norm()) for k, p in watched.items()}
            before = {k: p.detach().clone() for k, p in watched.items()}
        torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        opt.step()
    decay = 1 - opt.param_groups[1]['lr'] * opt.param_groups[1]['weight_decay']
    updates = {k: float((p.detach() - before[k] * decay).abs().max()) for k, p in watched.items()}
    assert all(math.isfinite(v) and v > 0 for v in [*grads.values(), *updates.values()])
    assert shared.frozen_hashes(model) == frozen
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    return dict(gradients=grads, beyond_decay=updates)


def main():
    weights, data, out = sys.argv[1:]
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    assert original.sha(weights) == original.WEIGHT_SHA
    for name, digest in original.DATA_SHA.items():
        if name.startswith('fold1/'):
            assert original.sha(Path(data) / name) == digest
    assert original.sha(Path(data) / 'fold1/tissue.npy') == shared.TISSUE_SHA
    fold = original.Fold(data, 1)
    fit, audit = shared.split_indices(fold.tissue, [any(g[0] == 4 and g[2] for g in row) for row in fold.gt])
    shared.validate_cohort(fit, audit)
    smoke_ids = shared.smoke_ids(fit, fold.gt)
    assert len(set(smoke_ids)) == 134 and not set(smoke_ids).intersection(audit)
    seeds = {}
    for seed in (19, 1):
        ids = np.random.default_rng(np.random.SeedSequence([seed, 0, 123])).permutation(smoke_ids).tolist()
        initial, counts, traces = {}, {}, {}
        for arm in ('P3', 'C', 'B'):
            reset(seed)
            model = original.Proposal(weights, True) if arm == 'P3' else p29.Proposal(weights, True, arm)
            initial[arm] = hashes(model.state_dict().items()), rng_hashes()
            counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.cuda()
            if arm == 'B':
                boundary = witness_boundary(model, fold, ids, seed)
            else:
                traces[arm] = run(model, fold, ids, seed, None if arm == 'P3' else 'C')
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assert initial['P3'] == initial['C'] == initial['B']
        assert counts['P3'] == counts['C'] == counts['B'] > 0
        assert traces['P3'] == traces['C']
        seeds[str(seed)] = dict(initial_state_equal=True, control_two_step_parity=True,
                                trainable_parameters=counts, boundary=boundary)
    shared.dump(out, dict(status='PASS', seeds=seeds, smoke_fit=smoke_ids, fold1_only=True))


if __name__ == '__main__':
    main()
