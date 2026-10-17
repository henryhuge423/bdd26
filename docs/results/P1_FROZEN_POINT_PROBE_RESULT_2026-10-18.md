# P1 frozen-feature point-heatmap probe — run v3 result and under-training diagnosis (2026-10-18)

**Status:** primary arm completed, both seeds FAIL the frozen gates; secondary (descriptive)
UNI arm crashed on a latent implementation bug (recorded below) and did not run. Frozen verdict:
`pass_primary = false` — i.e. not_supported **at the frozen 10-epoch head budget**. This document
also records the train-loss evidence that the head never converged, which classifies v3 as a
**null-by-under-training**: it is not evidence about the point-heatmap paradigm, and does not by
itself close anything. A separately pre-registered, convergence-corrected re-run (P1-R1,
`docs/proposals/P1R1_CONVERGENCE_CORRECTED_2026-10-18.md`) addresses exactly that gap with all
scientific parameters unchanged.

## Run identity

- Frozen design: `docs/superpowers/specs/2026-10-18-p1-frozen-point-heatmap-design.md`
  (sha 19c29e78…). Probe: sha f52519c2… (LM1 scripts / LM1+LM2 ops verified equal at launch).
- LM2 ops `p1_point_probe_20261018`; launched 2026-10-18 04:49:05 +0800 on GPU 0
  (uuid GPU-f2fca365-…, 81,050 MiB free at pick; receipt `launch.json` sha 5dc6ecae…).
- Cohorts/inputs: fold1 2,656 (train), fold2 2,523 (dev validation); all input shas asserted
  in-run. fold3 never read. GT objects: fold1 63,218; fold2 59,872; fold2 internal-Dead
  instances 741. Baseline (CellViT-UNI `final.pth`, NP ≥ 0.5 components): 53,987 candidates.
- Feature caches (UNI2-h, folds 1/2): 10,574,364,800 / 10,044,850,304 bytes written 04:52 +0800.

## Frozen gate outcomes (primary arm, D1 decode)

| metric (pooled fold2) | baseline | seed 19 | seed 1 |
|---|---|---|---|
| recall internal Dead | 0.74897 | 0.00270 | 0.00945 |
| Δ internal Dead (gate 1, ≥ +0.10) | — | −0.74629 FAIL | −0.73954 FAIL |
| bootstrap 95% CI of Δ (gate 2, lower > 0) | — | [−0.7905, −0.6984] FAIL | [−0.7850, −0.6905] FAIL |
| candidate precision (gate 3, ≥ baseline − 0.10) | 0.83327 | 0.97754 PASS | 0.97616 PASS |
| non-degeneracy (gate 4, ≥ 0.95) | — | 0.2833 FAIL | 0.30 FAIL |
| recall all classes (D1) | 0.56526 | 0.01370 | 0.01149 |
| internal-Dead recall (D2, descriptive) | — | 0.02834 | 0.08502 |
| candidates/image D1 / D2 | 21.40 / — | 0.335 / 1.995 | 0.283 / 2.547 |

Verdict: `pass_primary = false` (frozen rule: all four clauses in both seeds). Precision passes
where the head fires, but the head almost never fires: only ~0.3 candidates/image exceed p ≥ 0.5.

## Under-training diagnosis (train-loss floor; no fold2 gate value used)

The frozen penalty-reduced focal loss has a fitted floor ≲ 0.02 per batch (positive term → 0 as
p → 1; negatives suppressed by (1−Y)⁴). Observed last-epoch losses: seed 19 printed
1.885→1.079–1.371 (steps 200–1600), seed 1 2.153→1.267–1.512 — still trending down, ~two orders
of magnitude above the floor. An under-converged head emits near-zero probability maps, so gates
1/2/4 could not pass for optimization-budget reasons. Root-cause statement frozen in the R1
addendum; code review (2026-10-18) confirmed no coordinate/convention defect in
gt_objects / gaussian_target / decode_peaks / hits_for_points (all strictly (row, col)).

## Incident log for this run family

1. **aborted1** (pre-GPU, < 2 min): FileNotFoundError at `fold1/images` — loader used `k` without
   `.npy`. Fixed; outputs preserved (`tmp_aborted1/`, `results/run1_aborted1/`, `probe_aborted1.exit`).
2. **aborted2**: `ValueError: too many values to unpack` in `extract_uni2h` (5-dim probe tensor).
   Fixed (3 sites verified); outputs preserved under `*_aborted2`. Commit abaf18c (pushed).
3. **v3 crash (this run, rc=1)**: after both primary seeds completed, the descriptive UNI arm hit
   `RuntimeError: shape '[2, 1024, 16, 16]' is invalid for input of size 526336` at
   `extract_uni` — the UNI path feeds CLS+256 tokens (257) and that code path had never executed
   before. Not exercised by R1 (primary arm only); noted for any future secondary-arm use.

Evidence (LM2, preserved; file shas): `controller.log` 0556701b…, `probe.exit` (content `1`)
4355a46b…, `launch.json` 5dc6ecae…, `results/run1/norm_stats_uni2h.npy` = `norm_reused_uni2h_19.npy`
= `norm_reused_uni2h_1.npy` = b0f7cbbb… (identical fold1 statistics across seeds, as frozen).
No results.json/per-seed JSONs exist for v3 (writes occur only after both arms; the crash
preceded them); all gated quantities above are from the printed per-seed summaries.

## What this run does and does not establish

Does: the frozen P1 computation, executed as pre-registered, fails its gates; the failure mode is
non-firing, under-converged maps; precision-when-firing is high.
Does not: falsify the point-heatmap paradigm — the head provably had not converged, so the gate
did not test the paradigm's capacity. No test fold was read; no method claim is made.

## Next step

P1-R1 (pre-registered): identical computation, convergence-stopped head training (train-loss-only
rule, cap 100 epochs), primary arm only, same gates. If R1 fails at convergence, the point line
closes at this budget.
