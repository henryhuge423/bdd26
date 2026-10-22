# P9 Dead-center weighting: verified submission — 2026-10-22

Formal array **55208619** submitted once at **2026-10-22 19:55:04 EDT**. Submission is not completed training or an experimental result. C19/D19/C1/D1 are four fresh130epoch fold1-development runs; original admission F1=C/T1=D remains frozen. No automatic PhaseB,mask,fold2/fold3 or confirmation.

## Frozen intervention

C is original center focal/actual effective-batch instance count. D multiplies only annotated Dead-center positive focal terms by4; original negative terms and denominator stay unchanged. Includes boundaryDead,maximum4 on collisions; original D4 geometry synchronizes positive weights without consuming image/target RNG. Inference remains type-agnostic and unchanged. This independent point-loss recipe differs from historical failed dense-model Dead-pixel weighting; it does not establish imbalance or gradient conflict as a cause. [Design](../superpowers/specs/2026-10-22-proposal-p9-design.md), [P7/P8 closeout](PROPOSAL_P7_P8_RESULTS_2026-10-22.md).

Original fit2134/audit522,cohort55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22,seeds19/1,130epochs,originalaugmentation,UNI/head/optimizer/BF16,finalcheckpoint,decode and original admission retained. Both seeds must meet all original Dead recall/gain/CI/precision/relativequality/support gates. Failure/inconclusive closes this fixedweight4 budget without a weight sweep or extra runs. Reused development audit11images/145internalDead,two seeds and image-level splitting limit evidence; no PQ/test/independent validation claim.

## Validation

Original75focused checks pass12.67s. Initial new-module tests RED; original implementation96passed11.47s. Predeployment scan found first256fit images contain0Dead (fullfit780Dead),so inherited smoke would not exercise real weighting. Before any P9GPU submission,freeze validation-only first240fit plus first16distinct subsequent fit images with anyDead,identical for C/D/verifier. Two membership checks RED then final focusedP3-P9 **98passed11.58s**. Formal complete-fit membership/order never changed. Independent design and code review found no blockers. No metrics/HoVer-Net edits; historical5dense_votes failures disclosed,no full-suite pass claimed.

RealV100 smoke **55208331 COMPLETED/0:0**. Nodefocused98passed293.72s; initial states equal in bothseeds; C/P3 two real effective optimizer steps have all trainablegradients,adaptedparameters and AdamWstates bitwise equal. SyntheticGPU Dvalue/gradient agrees analytically; both arms have nonzero finite encoder/head updates and gradients. Actual internalBF16/terminalbilinearFP32 checked; last/finalreload predictions equal. Identical256smokeFit IDs recorded; D observed461augmented weightedpositivepixels,C0,all mapped to target==1. No audit predictions.

|Measure|C|D|
|---|---:|---:|
|1.5x safety estimated130epoch hours|7.077|7.033|
|TrainingpeakGiB (before reload)|4.060|4.060|
|Reloadconstruction+verification seconds|5.384|5.253|

Both pass unchanged12h timing gate; estimates do not guarantee runtime. strictready binds source/scripts/input/cohort/UNI plus tests/parity before formal. Same pending smokeID moved from gpu-shared to legalgpu-debug/gpu-debug-normal with identical30min/1GPU/4CPU/32GiB resources and sealed script hashes; no cancellation,newsubmission or source reseal. Override receipt retained. Official [SDSC guide](https://www.sdsc.edu/systems/expanse/user_guide.html) and current allocation/quota/queue/QoS checked.

## Submission provenance

Training source commit 051082692dce2dd1f29a8b6383d9cf4a9dad1988; manifest 85ef261efe90a495cbdbf16a6465369cbc076f8eebfe5c0c4585727e1d9edb32. Formal array0-3%2,mapping0=C19,1=D19,2=C1,3=D1; per task1V100/4CPU/32GiB/12h,gpu-shared/gpu-shared-normal/jhu159,Lustre,no-requeue,max48GPU-hours. Durable submissionguard/jobID/receipt saved. Do not rerun once-only scripts; inspect exact IDs before action. Shared runtime/data/UNI reused without upgrades or large transfers. Completed results copied with project quota guards and preservationmanifest while retaining scratch; Lustre is not archival backup.

P7/P8 both close under original gates. Main checkout,all previous branches/output and unrelatedjobs preserved. Research branch stays isolated; no merge/push. After allfourP9 complete,verify130epochs,exitcodes,preservation/input/config hashes,collect small artifacts,apply originaladmission once. No automatic expansion.
