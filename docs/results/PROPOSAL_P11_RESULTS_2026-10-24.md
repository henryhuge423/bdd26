# P11 positive focal redistribution: completed, original admission failed

2026-10-24 America/New_York. All four130epoch runs, array55244166 tasks0-3, COMPLETED/0:0, empty SLURM stderr. Summed allocated one-GPU runtime16h54m40s. Forty smallfiles/2,985,074bytes returned to LM1; all36payloads match preserved SHA256/bytes. Exact config, fit2134/audit522, cohort/input/UNI identities agree; loss histories exactlyepochs1-130 finite nonnegative. Last/final multi-GB checkpoints remain on Expanse; this review did not download or independently rehash them.

| Seed | C Dead | G Dead | G-C | Original paired95%CI | C/G auditrecall | C/G precision | Gate |
|---|---:|---:|---:|---|---|---|---|
|19|68/145=.468966|74/145=.510345|+.041379|[+.025381,+.075310]|.588816/.576490|.692692/.700040|false|
|1|67/145=.462069|67/145=.462069|.000000|[-.038462,+.035398]|.589224/.578612|.699961/.705063|false|

Original admission mapping F1=C/T1=G is false for BOTH seeds. Both fail absoluteDead>=.60,delta>=.05,precision>=.75. Seed1 also fails strictly-positive bootstraplower. Both pass relative precision floor-.02, overallrecall floor-.02, support11Dead-positive images/145internalDead,nondeg1.0. Seed19's positive interval does not override the conjunction or substitute for seed1. Two-seed mean change+.020690 is descriptive, not a pooled statistical test. Close this exact mass-redistribution recipe; no bandwidth/mass/threshold menu, extra seeds or epochs, originalPhaseB or masks.

|Run|Fit recall|Fit Dead recall|Fit precision|SmallDead hits/32|Candidates|Trainingh|TrainpeakGiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|C19|.975259|.952744|.998694|8|10413|4.029|4.047|
|G19|.943631|.896341|.999480|8|10088|4.014|4.047|
|C1|.980046|.958841|.998660|7|10312|4.046|4.047|
|G1|.951774|.907012|.999485|8|10053|4.031|4.047|

No512candidate-cap saturation. Redistribution reduces fit recall and audit overallrecall while slightly raising candidateprecision. The objectives differ, so raw loss values C~.05/G~.63 cannot rank quality. InternalDead improvements are seed-dependent; smallDead+0/+1 is descriptive. These aggregates do not isolate a causal failure mechanism.

## Research synthesis and next experiment

P4foregroundBCE, P5nativegrid, P6negativehalo, P7RGBgains, P8imagenormalization, P9Deadweight4, P10last4adaptation and P11massredistribution all failed original admission. Do not pool fresh controls across these rounds as pairedreplicates. Near-perfect C fit versus audit~.59 continues to motivate a bounded regularization experiment; it does not prove overfitting or establish a winning method. P12 tests exactly one .2 channel-dropout after decoderstemReLU against a fresh original control, with parameter/init/RNG/inference-equivalence controls. It retains last12adaptation and the original loss, unlike P10/P11. No closed menu is reopened.

All P-series point results are repeatedly reused fold1 image-development audit, only11Dead-positive images/145objects, 80%objects in5Lung/Colon images from previous descriptive audit. They are not independent validation, officialPQ/test or three-split performance. Imagebootstrap does not establish patient independence or correct sequential hypothesis selection. No fold2/fold3 model evaluations in this review.

Trainingcommit3fa1d6550d884729c96c380b5b78a93d7ad28fbf; sourcemanifest78cb80d7552091ade0630fd46f50fa3280aba7c09d037f5d3ae891d1c918a989. Evidence LM1 ignoredops/review_20261024/{p11_collection_manifest.json,p11_gate_01.json,p11_analysis_01.json,p11_live_01.json,p11_collected_01/}. Expanse project/scratchbdd26/p11_20261023_01 preserves checkpoints/results; Lustre externalarchive still outstanding.
