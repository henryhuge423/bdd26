# P3 Phase A: proposal generalization closeout (2026-10-20)

P3 completed all eight 130-epoch runs. **Original admission=false; the tested recipe closes. Phase B was not run.** P4 is a separately frozen auxiliary-supervision hypothesis, not a reinterpretation of P3's negative result. These are fold1 image-held-out development results, not test or mask/PQ performance.

The frozen design lives in the P3 study's provenance (commit856b713). Four arms compare frozen/trainable UNI features (F/T) with absent/present augmentation (0/1), seeds19/1. Fit2134 images and audit522 images; audit has145 internalDead objects in11 images. Original radius3 maximum-cardinality unique matching and decoder remain unchanged.

## Two-seed means

| arm | fit recall | audit recall | audit precision | audit internalDead recall | audit smallDead recall |
|---|---:|---:|---:|---:|---:|
| F0 | .999922 | .442776 | .706998 | .351724 | .187500 |
| F1 | .538848 | .428857 | .815171 | .396552 | .312500 |
| T0 | .999833 | .509918 | .724614 | .375862 | .218750 |
| T1 | .977035 | .588939 | .692705 | .437931 | .234375 |

SmallDead means internalDead with area<67 native pixels, a previously training-defined descriptive cut; only32 audit objects. Do not use it to select a new threshold or winner.

## Frozen primary comparison T1−F1

| seed | Dead recall F1/T1 | change | stratified image-bootstrap95%CI | precision F1/T1 |
|---|---|---:|---|---|
| 19 | .358621 / .427586 | +.068966 | [−.013793,+.166667] | .821038 / .691090 |
| 1 | .434483 / .448276 | +.013793 | [−.038462,+.044248] | .809304 / .694319 |

Both seeds fail absolute Dead recall>=.60, precision>=.75, precision>=F1−.02 and positive bootstrap lower bound. Seed1 additionally fails change>=.05. Support and nondegeneracy are sufficient; larger overall recall cannot substitute for these failed clauses. No extra epochs/seeds, threshold search, promoted arm or automatic Phase B.

## Interpretation

Without augmentation, F0/T0 essentially memorize fit images yet audit recall remains .44/.51. With augmentation, F1 fits only .54 recall while T1 fits .98, indicating an optimization/representation difference under the frozen recipes. T1−F1 improves overall audit recall by about.160 but reduces precision by about.1225; mean internalDead change is only.04138 with uncertain seed-specific intervals. This is a transfer/precision limitation of this recipe, not proof that all point detectors or pathology encoders fail.

No fit/audit image in any run reaches the512 candidate cap. Therefore cap truncation is not supported as the explanation. T1's candidates lie inside annotated nuclei at roughly.895 versus strict radius3 unique precision.693; localization error, duplicates and annotation differences could contribute, but this gap alone identifies none causally. These point endpoints cannot be subtracted from E19's maskIoU metrics as pipeline attrition.

T1/F1 primary comparisons were all completed on A100. F0/seed1 ran on V100; T0/seed1 resumed from epoch83 on V100 after source-disk headroom protection stopped its original run. Original precision/settings were preserved, but cross-device arithmetic remains an additional secondary-comparison limitation. Reused development images, image rather than patient splitting, two seeds and only11 audit Dead-positive images constrain inference. No fold2/fold3 model evaluation was added.

## Next study

[P4 design](../superpowers/specs/2026-10-20-proposal-p4-design.md) freezes a fresh C/N×seed19/1,130-epoch comparison of center-only versus center+balanced dense foreground BCE. Shared decoder/main recipe, cohorts, original point decoder and strict admission are fixed. No auxiliary output is used for point inference. This tests whether dense supervision helps transferable representation; it adds a small auxiliary parameter capacity and does not establish a causal mechanism or full PM-Net. Passing this train-only study still does not automatically authorize Phase B or three-split confirmation.

## Numeric evidence

Lightweight aggregation reads only frozen fit/evaluation/loss JSON files, with individual SHA256 recorded. Eight-run summary SHA256: `9b3e552a6736795a7a08fd6b0e5894eaf03d4f696b477957d0a150efc13cfa25`. Scientific data/weights, seeds and the failed original gate are unchanged. Detailed machine paths, original/continuation receipts, analysis script and provenance remain in ignored operations records; no large artifacts are committed.
