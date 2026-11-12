# Same-budget adaptation of early versus late encoder blocks

P22 obtained 18 and 20 additional interior Dead matches when it updated all 24 UNI blocks instead of the last 12. That comparison changed both the placement and number of trainable parameters. P25 also improved broad coverage with full-depth adaptation, while P26's fixed anchor on the last 12 blocks produced two additional interior Dead matches per seed and reduced overall recall and precision. P27 tests whether allocating the same trainable parameters to earlier blocks improves the final point predictions.

## Frozen comparison

- C updates UNI blocks 12–23 and the original point head. E updates blocks 0–11 and the same point head. Block indices are zero-based. The patch embedding and the rest of the encoder are frozen in both arms. E preserves the forward computation and backpropagates through the frozen later blocks.
- Both arms start from the same UNI weights and, within a seed, the same head initialization. Encoder/head learning rates are 1e-5/3e-4, cosine over 130 epochs, AdamW with decay .01 and clipping at 1. The original effective batch of 16 and microbatch of 4, training images, deterministic order and augmentation, center targets, focal loss, and native 256-pixel input are shared.
- Train fresh C19/E19/C1/E1 on the original 2,134 PanNuke fold-1 training images. Evaluate the last checkpoint on the frozen 522-image development set with the original peak decoding and one-to-one matching. Apply the original two-seed rule with C as F1 and E as T1. Report counts by tissue and fixed sample identities as descriptive follow-up. Save all four training and eight development examples per run with raw coordinates and matching, selected by the existing label-only rule.
- Equal trainable counts establish the parameter budget. Backpropagation through the frozen suffix can change execution time and memory. A successful result would support the fixed early-block recipe on this development split. P22 remains a historical comparator.

## Validation and submission

1. Use the isolated LM1 worktree and pinned environment. Test trainability boundaries, optimizer membership, CLI and checkpoint metadata, exact loading keys, and dispatch to the shared training loop. Preserve the original P26 CLI and C/R behavior.
2. In one V100 smoke job capped at 30 minutes, check both seeds for initial-state equality, original-C two-step state parity, E gradients in the first and twelfth blocks including gradient through the frozen suffix, real changes beyond AdamW decay, frozen hashes, optimizer membership, and BF16. Run the actual training loop over the existing 134 fixed training images for one epoch per arm. Check nine updates, last/final reloads, fixed images, 1.5× time estimate below 12 hours and training peak at or below 24 GiB.
3. Verify the tested source on Expanse with the existing inputs and runtime. Validate storage quotas, destination readback, account headroom, and scheduler admission. Submit exactly once after smoke completes with its bound receipt. Request one V100, four CPUs, 32 GiB and 12 hours per task, array concurrency two, for 48 requested GPU-hours. Save the smoke and formal receipts.
