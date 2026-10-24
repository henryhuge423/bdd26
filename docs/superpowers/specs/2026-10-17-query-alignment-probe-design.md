# E8a: fixed-query classifier mechanism probe

2026-10-17 America/New_York. This is the bounded train-only mechanism work requested after the E7 review, not a revised E7 recipe or a method confirmation.

Question: with masks, query features and Hungarian identities fixed, can the final linear query classifier recover useful masks? Does balancing foreground/background classification improve a second fixed training batch, rather than only fitting the optimized batch?

Use E7 Q final checkpoints, seeds19/1. Inputs are exclusively fold1: the first32 deterministic sample keys of data-seed19, epoch0. Fit on keys0:16, audit on keys16:32. The second batch is never optimized. Both checkpoint seeds see exactly the same images and augmentation. No fold2/3 arrays, predictions or metric summaries are read. Saved provenance records may mention fold2, but only current source, UNI and fold1 input bytes are verified.

Cache final query vectors, mask embeddings/features and original classifier logits in eval mode. Compute Hungarian assignments ONCE from the original checkpoint. Those labels remain fixed in both arms. No encoder, query-layer, mask, BN or tissue parameters/states are updated. No new checkpoint is saved.

Arms: A0 frozen original classifier; A1 same classifier initialization, original 2x weighted cross entropy (background weight .1); A2 same initialization, 2x equal means of foreground and background CE. Only the loss normalization differs between A1/A2. Each has a fresh AdamW, lr3e-4, betas(.85,.95), wd1e-4,300 full-cache steps, no LR scheduler. Record steps0/100/300, analyze final300 only. Run in order seed19 A1/A2 then seed1 A1/A2. Do not choose a better intermediate step.

Keep existing class argmax/score>=.5, mask threshold .5 and overlap decoder unchanged. Report fixed-assignment foreground accuracy/background accuracy and original weighted CE; best raw-mask recall, best accepted-mask recall and best final-instance recall at IoU>.5. Diagnostic maxima use GT, are not deployable or official metrics. Include all classes/support and canonical assignment retention; do not infer patient or seed-population uncertainty.

Allocation rule frozen before probe: a head-only alignment mechanism is promising ONLY if final A2 improves audit-batch best-final recall by>=.05 against A0 in BOTH checkpoint seeds, neither audit-batch fixed-assignment foreground accuracy drops, and fit-batch foreground accuracy reaches>=.90 in both. A2-minus-A1 is separately descriptive: A2 cannot be credited with improvement due only to training the head. Any failure closes this specific balanced-head mechanism at this stage; no larger dev training, extra seeds, loss menu or retries. A pass only permits designing a separate train/validation experiment, never automatically launching it.

Limits: frozen original assignments may themselves be suboptimal; the optimized batch is tiny and augmented; audit images are from the same training fold; checkpoint seeds are two fixed models. A failure does not disprove end-to-end class/mask alignment or set models. No conclusions about validation/test performance.

Execution: one LM2 GPU with>=24GiB currently free; cap this process to30% device memory,20-minute wall limit, no saved feature caches or weights, <=50MiB small outputs,20GiB disk reserve. Existing foreign GPU processes are preserved; this is not a latency benchmark. LM1 GPU2 prohibited. Existing brokers only, no connection changes. Verify exact snapshot/source/checkpoint hashes and pinned versions. New script identity is recorded independently of frozen scientific source.

Validation: test original versus balanced CE on analytic examples and background-replication invariance; test classifier-only fitting leaves source classifier/features unchanged, starts from equal logits and rejects invalid/empty labels. CPU regression with CUDA hidden before GPU dispatch. Validate original cached logits against the unchanged classifier under the same bf16 autocast before fitting. Verify mask/features unchanged after every arm; independently recalculate allocation from saved final results. Preserve logs on failure; no automatic replay.
