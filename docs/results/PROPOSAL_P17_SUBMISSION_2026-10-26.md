# P17 convolutional proposal representation — submitted October 26, 2026

Formal array **55297061** was submitted once at **09:08:19 EDT** (`2026-10-26T13:08:19.357784Z`). Tasks 0–3 are C/seed19, K/seed19, C/seed1 and K/seed1. Each is a fresh 130-epoch run with one V100, four CPUs, 32 GiB and a 12-hour limit. Concurrency is two; the account is `jhu159`, partition/QoS `gpu-shared/gpu-shared-normal`, with Lustre and no requeue. The formal upper bound is 48 GPU-hours, with smoke separately limited to 0.5 GPU-hour.

At 09:16 EDT, all four tasks were `PENDING/Priority`. Submission is not training completion or method admission. Create-only submission guards, job ID and receipts exist; never replay the deployment or submission scripts.

## Fixed research question

Does a small, from-scratch convolutional proposal recipe merit further work relative to the adapted UNI detector? C exactly retains original P3 T1. K replaces the representation with three convolutional stages at widths 32/64/128, each containing two 3×3 convolution, GroupNorm(8) and ReLU layers. Average pooling downsamples between stages. Quarter-resolution features from all stages are concatenated into 224 channels, followed by a 64/32-channel fusion head and native bilinear output. Inputs use RGB/255 followed once by the original ImageNet normalization. Output weights start at zero, with bias `log(.01/.99)`.

All K parameters use learning rate 3e-4. C retains head LR 3e-4 and last-12-block encoder LR 1e-5. Both preserve original AdamW, cosine schedule, clipping, effective batch 16/microbatch 4, full permutation, D4/color augmentation, focal loss divided by effective-batch object count, sigma 1/radius 3 targets, 130 epochs, last checkpoint, decoding and admission. No P16 regression objective or other closed treatment is stacked into K.

Architecture, initialization, pretraining and optimization change together. This is a complete recipe comparison, not an isolated locality effect, capacity-matched experiment or controlled speed comparison. GroupNorm uses spatial statistics. The design was chosen independently before P16 seed 1 completion; it neither promotes P16's partial point estimate nor reopens P3–P15 closed recipes.

The original admission must pass separately in **both** seeds, with F1 mapped to C and T1 to K: internal Dead recall ≥.60, gain ≥.05, original tissue-image bootstrap lower bound >0, unique precision ≥.75 and ≥C−.02, overall recall ≥C−.02, nondegeneracy ≥.95 and support ≥10 images/100 objects. Failure or an inconclusive result closes this exact width/LR/objective/budget without menus. Success permits discussion of a future design; it does not automatically authorize masks, Phase B, fold-2/fold-3 evaluation or confirmation.

The fixed fold-1 split remains 2,134 fit and 522 audit images. The audit has 11 Dead-positive images and 145 internal Dead objects, 116 from Lung and Colon. Point matching is not mask IoU, PQ, test, independent-patient or official three-split performance. The bootstrap does not correct sequential development reuse or estimate training-seed population uncertainty.

## Validation

After the missing-module failure, 31 new checks and 289 proposal checks passed. Independent review found that cumulative parameter changes could include first-step weight decay. Eighteen counterexamples failed before the fix; second-step-only changes and an explicit decay-only comparison then passed with 291 focused checks. Initial full CPU regression was 859 passed/3 skipped and preceded two additional counterexamples; final full CPU regression passed with 861 tests and 3 skips (19,073 pre-existing dependency deprecation warnings; 784.55s, exit 0), with the full receipt retained in the stage-6 handoff. Metrics, HoVer-Net and pinned dependencies were unchanged.

Real V100 smoke **55297045 completed with `COMPLETED/0:0`**, taking 6m12s on `exp-7-59`; 291 node checks passed in 78.17s and strict readiness passed. Both seeds reproduce C/P3 initial tensors and RNG, then two effective steps of every trainable gradient, parameter, AdamW state and RNG state. Every K encoder stage, fusion head and output layer has finite nonzero second-step gradients and measured updates beyond pure weight decay. Actual data/augmentation/loss/count/gradient, optimizer membership, frozen state, AMP, last/final reload, cross-configuration rejection, source/input/cohort identity and timing were checked. Twenty-five small evidence files were returned with byte/SHA checks; smoke made no audit predictions.

The original 256-fit-image timing formula, including a 1.5 safety factor, projects C/K to **6.952075/4.771626 hours**, below 12 hours. Training peaks were **4.070870/.351575 GiB**. These estimates are not completion guarantees. The current SDSC guide and live allocation, partition, QoS, quota, disk and queue were checked. Existing P16 tasks and unrelated job 55275190 were preserved.

## Provenance

Training commit: `5a7e85ba7e876109449aab85767bc66942952daf`.

The 407-file source manifest is `dd4649a13035c303c7596ba32eede4c00f9228c18dcf4c1fe285ae6a42c9c885`. Two inherited portability-test fixes were refreshed only in the new snapshot; the P16 parent remains immutable. Existing data, UNI weights and runtime were reused. No bulk transfer, dependency upgrade, deletion, merge, push or new SSH authentication occurred. Exact machine paths and receipts remain in ignored operations notes; large Lustre checkpoints still need external archival.

[Frozen design](../superpowers/specs/2026-10-26-proposal-p17-design.md) · [implementation plan](../superpowers/plans/2026-10-26-proposal-p17.md) · [stage report 6](../report/stage_report_6/README.md) · [P16 partial result](PROPOSAL_P16_PROGRESS_2026-10-26.md).
