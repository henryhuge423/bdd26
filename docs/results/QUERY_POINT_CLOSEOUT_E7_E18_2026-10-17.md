# E7–E18 representation research closeout (2026-10-17)

## Decision

The query-mask research line has not produced a method that passes the frozen train-only quality gates. Do not begin large training from this evidence. E17 and E18 both fail the actual-inference allocation gate; E18's conditional-mask fitting witness does not override its failed detector and precision gates. This closeout uses only fold1 diagnostics and the selected 16-image cohort. No E18 validation or test result exists.

## Evidence across the sequence

| Experiment | Change / observation | Result and decision |
|---|---|---|
| E7 | Set-mask query model; four 130-epoch runs with final-checkpoint fold2 evaluation | Q mPQ .087958 vs matched B .472939; all quality gates fail. Close the trained recipe. [E7](SET_MASK_E7_DEV_2026-10-17.md) |
| E9 | Local-content query initialization | Both train-only allocation gates fail; the modest Dead-recall gains accompany precision losses and poor audit geometry. Stop. [E9](LOCAL_QUERY_E9_DEV_2026-10-17.md) |
| E10 | Predicted-mask constrained attention | Both allocation gates and A0 guard fail. Stop. [E10](MASKED_QUERY_E10_DEV_2026-10-17.md) |
| E11 | Equal-weight supervision at the initial and three decoder query stages | Both allocation gate and A0 guard fail. Stop. [E11](AUX_QUERY_E11_DEV_2026-10-17.md) |
| E12 | Free per-instance mask coefficients on frozen features, initialized with GT-matched queries | Fit IoU recall reaches 89.98/90.65% overall and 86.32/87.37% for Dead. This is an oracle fitting witness, not a detector or method score. [E12](FREE_MASK_E12_DIAGNOSTIC_2026-10-17.md) |
| E13 | Freeze the original mask projection | Audit Dead recall is only 1.48/7.38%; both gates fail. Free coefficients fitting well does not mean the shared projection can be frozen. [E13](STABLE_MASK_E13_DEV_2026-10-17.md) |
| E14 | Replace stride-sampled matching with exact native-resolution matching | The matching-cost correction fixes an off-grid failure mode, but does not rescue allocation; audit Dead recall is 2.95/3.16% and gates fail. [E14](NATIVE_MATCH_E14_DEV_2026-10-17.md) |
| E15 | Conditional mask readout given one GT point and a 3×3 local descriptor | Seed 19 misses the 80% fit witness; audit Dead recall remains 31.58/42.11%. Not a detector. [E15](POINT_READOUT_E15_DIAGNOSTIC_2026-10-17.md) |
| E16 | Expand the point descriptor to a 3×3 feature context | Both seeds pass the fit witness (fit Dead recall 84.21/80.26%), while audit Dead recall remains 31.58/42.11%. The improvement is conditional on GT point location. [E16](CONTEXT_READOUT_E16_DIAGNOSTIC_2026-10-17.md) |
| E17 | Predict sparse point heatmaps and decode their local maxima at fixed 0.5 | Zero candidates on all 16 images for both seeds; actual-inference recall and precision are zero. Close this recipe. [E17](PREDICTED_POINTS_E17_DIAGNOSTIC_2026-10-17.md) |
| E18 | Dense foreground-pixel offsets voting to predicted instance points | The fit witness passes, but dense votes produce thousands of proposals and only 3.5–4.3% fit precision and 3.5–3.7% audit precision. Both seeds fail the allocation gate and A0 guard. Close this recipe. [E18](DENSE_VOTES_E18_DIAGNOSTIC_2026-10-17.md) |

## What the evidence says

The repeated bottleneck is instance proposal allocation and calibration. E12 shows the frozen mask basis contains enough information to fit many masks when GT instance assignments are supplied. E15–E16 show that the mask readout can learn from a GT point, with local context improving train fit. Neither observation establishes that an inference system can generate a useful point set.

E17 and E18 test that missing stage at opposite extremes. The sparse heatmap emits no proposals at the fixed threshold; dense pixel votes emit up to 512 proposals per image, mostly unmatched or low quality. E18's overall audit recall improves slightly over A0 in both seeds, yet audit precision drops by 13.15 and 15.06 percentage points. Seed 19 also loses 5.26 points of Dead recall. Seed 1's Dead-recall gain of 21.05 points is insufficient because the precision and fit-recall gates fail. Typed Dead F1 stays below 2.1%.

These are deliberately small, train-only diagnostics. The 8-image audit split was held out from each probe's fitting but was already seen by E7, and it contains only 19 Dead objects. This evidence can reject the tested recipes; it cannot estimate independent validation performance or establish that all point-based methods fail.

## Reproducibility and next boundary

E18 A0 rows exactly reproduce E11 on the selected images. A standalone verifier independently rebuilt fit coverage, per-image detector counts, admission deltas, and gate outcomes from the saved result. E18 used LM2 GPU0, completed in 74.69 seconds, and reserved at most 1.43 GiB. LM1 GPU2 was not used.

This session ends at E18 as requested. No threshold sweep, extra steps, new seed, full fold training, validation evaluation, or large training was started. Any future method would need a predeclared proposal-count/precision gate on a fresh train-only cohort before it could justify a separately scoped full train/validation design.
