# P16 partial development results — October 26, 2026

At 09:16 EDT, array 55287551 tasks 0/1 (C/R, seed 19) had completed 130 epochs with SLURM status `COMPLETED/0:0`. Tasks 2/3 (seed 1) were running at epochs 80/68. This is a partial result, not a completed two-seed study or a new test result. Existing submission guards and receipts must not be replayed.

The two completed runs returned 18 small files whose sizes and SHA256 values match their preservation manifests. All 401 sealed source files were independently hashed. We checked the exact 2,134 fit and 522 audit image identities, their disjointness, the 130-epoch histories, configurations and per-image results. Large checkpoints were neither downloaded nor independently rehashed.

| Seed 19 endpoint | Original focal C | Gaussian regression R |
|---|---:|---:|
| Fit overall recall | .980007 | .872135 |
| Fit internal Dead recall | .961890 | .846037 |
| Audit overall recall | .598041 | .602531 |
| Audit unique precision | .694079 | .701149 |
| Audit internal Dead recall | 70/145 = .482759 | 75/145 = .517241 |

Independent aggregation from the preserved per-image rows exactly reproduces the saved summaries. Replaying the original 1,000-draw tissue-stratified image bootstrap (RNG 20261018) gives a Dead recall change of **+.034483**, with 95% CI **[-.034091, +.131144]**. The original admission implementation gives the same values. This seed fails absolute Dead recall ≥.60, gain ≥.05, strictly positive lower confidence bound and unique precision ≥.75. An improved point estimate cannot replace the original conjunction.

R fits less well and has a small positive audit difference in this fixed seed. The treatment changes loss scale, background weighting and Gaussian gradients together; it does not isolate halo supervision or establish probability calibration. Seed 1 completion and full preservation/configuration/source/input checks remain necessary for a complete closeout. No extra coefficient, sigma, threshold, seed or epoch menu is opened, and no masks, Phase B or confirmation are promoted.

The repeatedly reused fold-1 audit contains 11 Dead-positive images and 145 internal Dead objects, 116 from Lung and Colon. These point metrics are not mask IoU, PQ, test, independent-patient or official three-split results. Image bootstrap does not account for sequential hypothesis selection or training-seed population uncertainty.

See the [frozen P16 protocol](PROPOSAL_P16_SUBMISSION_2026-10-26.md), [stage report 6](../report/stage_report_6/README.md) and [independent P17 study](PROPOSAL_P17_SUBMISSION_2026-10-26.md). P17 was frozen before P16 seed 1 completed; it does not promote this partial result or reopen the original Phase B. Exact collection, request and job receipts remain in ignored operations notes. External archival of the large Lustre artifacts remains outstanding.
