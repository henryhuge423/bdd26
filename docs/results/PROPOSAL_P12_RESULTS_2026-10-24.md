# P12 decoder Dropout2d result — 2026-10-24

Array 55251406 completed all four fresh 130-epoch runs, each COMPLETED / 0:0 with empty SLURM stderr. C is the original control; R adds channel Dropout2d(p=0.2) after the stem ReLU. The frozen decoder, optimizer, adaptation budget, split and admission rule were retained.

| Seed | C Dead recall | R Dead recall | R−C | Original paired CI95 | C / R precision |
|---|---:|---:|---:|---|---|
| 19 | 0.482759 | 0.448276 | -0.034483 | [-0.106195, +0.015385] | 0.691729 / 0.689088 |
| 1 | 0.468966 | 0.441379 | -0.027586 | [-0.063636, +0.000000] | 0.697158 / 0.690146 |

Both seeds fail the original gate: R Dead recall is below .60, gain is below .05, bootstrap lower bound is not positive, and precision is below .75. Support and nondegeneracy pass. Close this exact probability/location/budget recipe; no additional seed, epoch, threshold menu, Phase B or mask experiment is admitted.

R reduces fit recall (~.941/.945 versus C ~.975/.979) without improving Dead audit recall. This supports a negative decision on the fixed recipe, not a general causal conclusion about dropout or regularization. No candidate cap saturation occurred. Both arms retain a large fit–audit gap.

These are reused fold1 development point metrics (522 audit images, only 11 Dead-positive images / 145 internal Dead objects), not PQ, independent validation, or test performance. The tissue-stratified image bootstrap is conditional on these trained models and the small observed cohort; it does not establish seed-population uncertainty. No fold2/fold3 was read.

## Verification

40 small files (2,985,680 bytes) collected; all 36 payload SHA256/byte counts match their preservation manifests. All four frozen run configurations, fit/audit identities, input and UNI hashes match; each history has epochs 1–130 with finite nonnegative losses. Large checkpoints remain on Expanse and were not independently rehashed in this review.

Training commit: `77cfa9ac17b278c67b750762191e6bdd5b9b4cc9`; source manifest: `35748e383fa57d626c3c8a03fc09871627872c19599d2a3003559a1845bb1a3d`. SLURM elapsed allocation totals 17h00m57s across four one-GPU jobs. First pair finished 11:30 EDT; final pair finished 15:52 EDT on October 24.

Before publishing, the P3–P12 CPU suite passed 153 tests in 24.31 seconds. The historical E8–E18 diagnostic suite passed 46 tests in 13.69 seconds. Historical diagnostic sources are archived on their original research branch, not promoted into the P12 training snapshot. A broader attempted check including E19 could not collect because the worktree lacks the legacy `misc` vendor import; no full-suite pass is claimed.
