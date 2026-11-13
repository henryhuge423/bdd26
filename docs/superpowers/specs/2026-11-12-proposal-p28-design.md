# Distribute the same encoder adaptation budget across depth

P22's full-depth updates improved broad coverage and matched more interior Dead nuclei. P25's full-depth layerwise rates also improved coverage, with different changes across tissues. Both comparisons changed the number of trainable parameters. P27 compares updating a contiguous early half or late half at equal parameter counts. P28 asks whether distributing those same 12 blocks across the whole depth improves development coverage while retaining adaptation in the late blocks, including the output-side block.

## Frozen comparison

- C updates blocks 12–23. I updates blocks 1, 3, 5, …, 23, using zero-based indices. Both update the original center head. The other encoder tensors stay frozen. The two arms share initialization and the number of trainable parameters. The odd-index convention retains block 23 and is fixed before any P28 output.
- Each arm uses seeds 19 and 1, 130 epochs, the original 2,134 fold-1 training images and 522 development images, effective batch 16, microbatch 4, identical focal loss, AdamW, clipping and cosine schedule. Encoder and head learning rates are 1e-5 and 3e-4. Apply the existing one-to-one decoder, last-checkpoint evaluation and two-seed decision rule. Preserve eight fixed development and four fixed training examples per run.
- This is adaptive exploration after prior development results. A difference estimates the effect of this fixed block allocation under the common budget. Record whether the broader allocation affects training coverage, aggregate development precision and recall, tissue-specific Dead counts, and time and memory.

## Validation and submission

- Verify exact trainable block names and equal counts, strict reload configuration, shared control behavior, and finite gradients and actual updates in both the earliest and final trainable blocks. On V100 verify both seeds over two effective batches and perform the full 134-training-image smoke in both arms, including last/final reload and fit-only images.
- Seal source and launcher hashes. Check three storage quotas, stripe capacity, small-file readback, allocation and scheduler admission. After the V100 smoke passes, submit C19/I19/C1/I1 once, concurrency two, one GPU, four CPUs, 32 GiB and 12 hours per task. Retain the receipt and exact source commit.
