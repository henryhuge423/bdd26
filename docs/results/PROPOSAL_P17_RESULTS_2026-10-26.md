# P17 completed results and P18 research decision — 2026-10-26

All four P17 array 55297061 tasks completed 130 epochs with SLURM COMPLETED/0:0. Both original admission rules fail; close this fixed from-scratch convolutional recipe. P18 was frozen at commit c679f4b before any P17 production scores were read; its independent C/H design has now passed the V100 gate and was submitted once as array **55311007**. Submission is not completion or method admission.

## Verified P17 evidence

The collection verified every one of 407 sealed source files against the immutable manifest, and 36 small payload files by bytes/SHA256 against the preservation records. Exact fit2134/audit522 identities, disjoint lists, weights/input/cohort hashes, configuration, row identities and uninterrupted epoch1–130 records agree. Reaggregating every evaluation row exactly reproduces all four summaries. The original admission implementation and an independent replay of its 1000-draw tissue-stratified image bootstrap RNG20261018 agree exactly. Large checkpoints were excluded from text collection regardless of size and were not independently transferred or rehashed; the K final checkpoint is small enough to defeat a size-only text filter, which was corrected before the successful collection. Original artifacts remain on project and scratch; external archival is outstanding.

| Seed / arm | Internal Dead hits /145 | Overall recall | Unique precision | Fit overall recall | Fit Dead recall |
|---|---:|---:|---:|---:|---:|
| 19 / C | 60 = 0.413793 | 0.594694 | 0.684423 | 0.978928 | 0.967988 |
| 19 / K | 30 = 0.206897 | 0.139510 | 0.952089 | 0.144051 | 0.230183 |
| 1 / C | 67 = 0.462069 | 0.594857 | 0.694662 | 0.976946 | 0.948171 |
| 1 / K | 31 = 0.213793 | 0.118939 | 0.949186 | 0.119212 | 0.211890 |

| Seed | K−C internal Dead recall | Original 95% CI | Original admission |
|---|---:|---|---|
| 19 | -0.206897 | [-0.287037, -0.157609] | false |
| 1 | -0.248276 | [-0.336283, -0.190217] | false |

The arithmetic mean paired Dead difference is -0.227586; this is a descriptive average of two fixed seeds, not population significance. Both K arms fail absolute Dead>=.60, gain>=.05, positive lower CI and overall-recall>=C−.02. Seed1 additionally fails nondegeneracy>=.95 (10/11 positive images have any candidate). Its precision alone exceeds .75. Both controls are freshly trained within P17; cross-round control values are not interchangeable. Source/weight/cohort identity and smoke parity do not imply bit-identical full training across devices or rounds.

## Precision rose because coverage contracted

Seed19 C emitted10644 proposals and uniquely matched7285 of12250 objects; K emitted1795 and matched1709. Precision therefore rose .684423→.952089 while recall fell .594694→.139510. K's median audit-image count was1 rather than C's17, and it emitted no proposals in198/522 images versus22/522. These counts explain why a 95% precision headline would be misleading. Neither arm reaches cap512 (K max41); candidate clipping cannot explain the observed K deficit. Small internal Dead hits fell7/32→2/32 in seed19; K seed1 found3/32. This is a fixed <67-native-pixel descriptive subgroup, with no new selection rule.

K also fits the training images poorly: overall fit recall .144051/.119212, versus the matched C values in the table. Its fit internal-Dead recall is .230183/.211890. The near-equal low fit and audit coverage differs from the original UNI controls' high-fit/low-audit profile. The evidence supports inadequate fitting at the prescribed operating point within this recipe and budget, not a general claim that local features cannot generalize. Representation, pathology pretraining, initialization, optimizer rates, parameter count and output-grid resolution change jointly: K predicts at64x64 before interpolation, while C's head reaches128x128. The experiment does not isolate which factor caused the failure, and unchanged threshold.5 does not establish probability calibration. No post-hoc threshold, width, learning-rate or epoch search follows.

Across represented tissues, K's low overall recall is broad. Dead-specific evidence remains concentrated: Lung/Colon carry116/145 internal objects in only5 images. Seed19 Dead hits in Lung fall36/60→18/60 and Colon13/56→4/56, while Cervix rises5/15→7/15. Such small tissue cells support descriptive localization only. Bootstrap intervals remain conditional on the reused image collection and omit sequential hypothesis selection and training-seed population uncertainty.

## P16 and the accumulated evidence

[Verified P16](PROPOSAL_P16_RESULTS_2026-10-26.md) changed Dead recall by+.034483/+.062069, but failed the original conjunction in both seeds; absolute recall .517241/.496552 and precision .701149/.705009 remained insufficient. Its improved audit coverage accompanied lower fit coverage. Earlier [P3–P15 synthesis](PROPOSAL_PROGRESS_2026-10-26.md) documents high fit but weak transfer in many controls, the failure of added rare-positive exposure, and coverage losses under translation. P17 adds a distinct failure profile: replacing the pretrained representation with this scratch CNN severely reduces fit as well as audit recall. These results close exact recipes; they do not rank whole method families or provide an unbiased search over hypotheses.

## Why P18 is a separate, bounded experiment

P18 retains the original adapted UNI and point head and adds a zero-output-initialized RGB residual path in the point logits. It preserves original normalization, cohort, loss/denominator, full permutation, D4/color augmentation, optimizer schedule,130epochs,lastcheckpoint,decoder and BOTH-seed admission. C/H x19/1 are fresh; new path parameters use head LR3e-4, UNI remains last12 LR1e-5. Global clipping includes the new parameters. This tests an added image path/capacity/optimization interaction, not pure locality or a compute-matched comparison. GN uses spatial statistics. It is neither P17 width/LR tuning nor E19 mask-skip retraining. See the [frozen design](../superpowers/specs/2026-10-26-proposal-p18-design.md) and [submission](PROPOSAL_P18_SUBMISSION_2026-10-26.md).

The user authorized progression through the next Expanse submission. The next action after P18 completion is to verify its exact outputs and apply the original gate, with no automatic masks, PhaseB, extra seeds or menus. Even a pass would be development evidence on repeatedly reused fold1, not independent generalization validation. Future independent evaluation requires a separate frozen design; no fold2/fold3 model evaluation occurred here.

Current CPU regression: **906 passed, 3 skipped, 19073 warnings in 474.97s (0:07:54)** (exit0); CUDA was hidden for this CPU run, so skipped GPU-only checks are not CPU passes. The real V100 P18 protocol was checked separately.
