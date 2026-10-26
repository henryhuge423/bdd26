# P16 completed development results — 2026-10-26

At 11:29 EDT, array 55287551 tasks 0–3 were all `COMPLETED/0:0`. Every C/R run for seeds 19/1 completed the original 130 epochs. This supersedes the earlier partial snapshot without changing its historical record.

We recovered 28 small result files with transfer SHA256/size verification and checked all 401 sealed source files against the deployment manifest. The exact 2,134 fitting / 522 audit image lists, disjointness, cohort/input/UNI identities, per-image row identities and uninterrupted epoch sequences 1–130 agree across all four runs. Per-image aggregation exactly reproduces each saved summary. Large checkpoints were neither transferred nor independently rehashed; independent archival of the large Lustre artifacts remains outstanding.

| Seed / objective | Internal Dead hits / 145 | Audit overall recall | Unique precision |
|---|---:|---:|---:|
| 19 / original focal C | 70 = .482759 | .598041 | .694079 |
| 19 / squared-error R | 75 = .517241 | .602531 | .701149 |
| 1 / original focal C | 63 = .434483 | .593959 | .695137 |
| 1 / squared-error R | 72 = .496552 | .600898 | .705009 |

Replaying the unchanged original admission implementation and independently reproducing its 1,000-draw tissue-stratified image bootstrap (RNG 20261018) yields:

| Seed | R−C internal Dead recall | Original 95% CI | Original seed admission |
|---|---:|---:|---|
| 19 | +.034483 | [−.034091, +.131144] | false |
| 1 | +.062069 | [+.028249, +.115044] | false |

Seed 1 passes the gain and positive-lower-CI clauses, but both seeds fail absolute internal Dead recall ≥.60 and unique precision ≥.75. Seed 19 also fails gain ≥.05 and a strictly positive lower confidence bound. The conjunction fails in both seeds: close this fixed coefficient-1, sigma-1 Gaussian-regression recipe. No additional coefficient, sigma, threshold, seed or epoch menu, masks, Phase B or automatic confirmation follows.

R lowers fit overall recall from .9800/.9809 to .8721/.8727; fit internal Dead falls from .9619/.9604 to .8460/.8628. The positive audit differences are limited signals under the fixed objective, not a qualified method. Loss scale, background weighting and halo gradients change jointly; the comparison does not isolate halo supervision or establish calibration.

The audit is repeatedly reused fold-1 development data: 11 Dead-positive images, 145 internal Dead objects, 116 in Lung/Colon. These are geometric point metrics, not mask IoU, PQ, test, official three-split or independent-patient performance. Image bootstrap does not account for sequential hypothesis selection or training-seed population uncertainty.

P17 was independently frozen before P16 seed 1 completed. At the 11:29 EDT snapshot, array 55297061 C19/K19 was running at epochs 44/59 and C1/K1 was pending under the array concurrency limit. No P17 production outcome or new submission is claimed. See the [P17 protocol](PROPOSAL_P17_SUBMISSION_2026-10-26.md), [P16 protocol](PROPOSAL_P16_SUBMISSION_2026-10-26.md) and [earlier P16 snapshot](PROPOSAL_P16_PROGRESS_2026-10-26.md).
