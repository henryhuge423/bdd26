# P1-R1 — Convergence-corrected addendum to the frozen P1 probe (pre-registered)

**Frozen:** 2026-10-18 (before the R1 LM2 run). **Status:** addendum to the frozen P1 design
(`docs/superpowers/specs/2026-10-18-p1-frozen-point-heatmap-design.md`, sha
19c29e78bea81823371df662856ae2747dcc32ec1a02f2e355fb18379ebf2a19). R1 re-runs the P1 primary
arm under a pre-declared convergence-based optimization budget because the frozen P1 run (v3)
demonstrated, by train-loss floor analysis, that the 10-epoch budget left the head grossly
under-trained — a null-by-under-training, not a valid test of the point paradigm. Full
disclosure of v3 and of every R1 constant is below. No fold2 metric influenced any R1 constant;
the stopping rule uses the training loss only.

## Why R1 exists (v3 record, disclosed)

The frozen P1 run (v3; LM2 ops `p1_point_probe_20261018`; launched 2026-10-18 04:49 +0800; probe
sha f52519c2…; receipt `launch.json`) completed BOTH primary-arm seeds and then exited rc=1 when
the descriptive secondary UNI arm hit a latent token-count bug in `extract_uni` (CLS+256-token
reshape; that code path had never executed before). Primary-arm gate results, as printed
(the run exited before writing results.json; the frozen verdict is fully determined by these):

- **seed 19:** Δrecall(internal Dead, D1) = −0.7463 (probe 0.00270 vs baseline 0.74899),
  bootstrap 95% CI [−0.7905, −0.6984], D1 precision 0.9775 (baseline 0.8333), non-degeneracy
  0.2833, candidates/image 0.335. **FAIL** (clauses 1, 2, 4).
- **seed 1:** Δ = −0.7395 (0.00945 vs 0.74899), CI [−0.78498, −0.69054], precision 0.9762,
  non-degeneracy 0.30, candidates/image 0.283. **FAIL** (clauses 1, 2, 4).
- Frozen verdict per P1: `pass_primary = false` (not_supported at the 10-epoch budget). This
  stands as the P1 result and is never rewritten.

**Root cause of the primary failure (established without using any fold2 gate value): the head
never converged.** The P1 penalty-reduced focal loss has a fitted floor ≲ 0.02 per batch
(positive term → 0 as p → 1; negatives weight-suppressed), yet the last printed training losses
were ≈1.08–1.37 (seed 19) and ≈1.27–1.50 (seed 1), still trending down with noise — about two
orders of magnitude above the floor. An under-converged head emits near-zero sigmoid maps
(observed 0.3–0.34 candidates/image at p ≥ 0.5), so P1's gates could not have passed for
optimization-budget reasons unrelated to the point-heatmap paradigm's capacity. A
null-by-under-training is not evidence about the paradigm; R1 removes this confound with a
train-loss-only, pre-declared stopping rule.

## R1 design (only these differ from P1; everything else byte-identical)

- **Script:** `scripts/probe_frozen_point_heatmap_r1.py`, sha256
  `2986cb1a3e11f728da775b1f230f47111a8dee53d79b47b3859ea8ace8e41035`; generated from the frozen
  P1 script (sha f52519c2…) by `make_probe_r1.py` with strict single-occurrence anchors; the
  173-line unified diff is stored beside it (`probe_r1.diff`). All frozen functions
  (gt_objects, gaussian_target, focal_center_loss, decode_peaks, decode_nms, hits_for_points,
  radius_curve, PointHead, feature extraction, baseline) are asserted byte-identical by tests.
- **Epoch rule (pre-declared; train-loss only):** up to `EPOCH_CAP = 100` epochs; checked at
  epochs 15, 20, …, 100; stop at the first check where either (a) the mean train loss of the
  last 5 epochs < 0.02 (`R1_LOSS_TARGET_5EP`), or (b) its relative improvement over the previous
  5-epoch mean is < 5% (`R1_PLATEAU_MIN_REL_DROP`). Seeds, shuffle RNG (per-seed default RNG),
  optimizer, lr, batch size, loss, decode, metrics, gates and bootstrap are unchanged from P1.
- **Scope:** primary arm only (UNI2-h; seeds 19 and 1). The descriptive secondary UNI arm is
  omitted (compute economy; its v3 execution was void by the latent bug above). Gates are
  primary-arm only by P1, so this removes no gated element.
- **Pre-declared non-gate diagnostics:** per-epoch mean train loss (printed + stored in the
  per-seed JSON), head checkpoints every 5 epochs, and an in-sample fold1 D1 fit table at the
  stopping point (clearly labeled; never a gate input).
- **Budget:** one LM2 GPU with ≥ 28 GiB free at launch, fresh
  `ops/p1r1_point_probe_20261018/{results,tmp}`, controller `timeout -k 30 12600`, in-script
  deadline 12,000 s. No LM1 GPU, no test fold (fold3 never read), no other jobs touched;
  failure stops with outputs preserved (no automatic retry).
- **Verdict semantics (unchanged from P1):** all four P1 clauses in both seeds ⇒ `pass_primary`,
  which permits a separately recorded P2 design; any failure ⇒ the point-head paradigm closes
  at this corrected budget. No threshold sweep, extra seeds, architecture/lr changes or rescue
  modifications may be added after seeing R1 results.

## Limitations (frozen statement)

As P1 (frozen features, reused fold2, two head seeds, no augmentation, image-level bootstrap,
small internal-Dead support ≈70 images), plus: the epoch rule is the only changed element and
was chosen from training-signal analysis alone; the in-sample fold1 table is diagnostic only;
a single optimization recipe (lr/arch frozen) bounds what "converged" means here.

## References

- P1 design: `docs/superpowers/specs/2026-10-18-p1-frozen-point-heatmap-design.md`
  (sha 19c29e78…).
- v3 evidence: LM2 `ops/p1_point_probe_20261018/{controller.log, probe.exit, launch.json,
  results/run1/}`; shas recorded in the ops record.
