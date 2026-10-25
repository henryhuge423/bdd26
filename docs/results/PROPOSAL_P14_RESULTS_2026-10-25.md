# P14 fixed encoder learning rate: negative development result

All four fresh C/L x seeds19/1 runs in array55270432 completed130epochs, SLURM COMPLETED/0:0. Summed allocated one-GPU walltime17h06m38s. Original last-checkpoint evaluation and admission recomputed from preserved rows: false in BOTH seeds. Close the exact encoder1e-6 recipe. No extra LR, seeds, epochs, thresholds, PhaseB or masks.

|Seed|C internal Dead|L internal Dead|L-C|Original paired95%CI|C/L overall recall|C/L unique precision|
|---|---:|---:|---:|---|---|---|
|19|74/145=.510345|64/145=.441379|-.068966|[-.114583,+.021517]|.583429/.495102|.689067/.776569|
|1|69/145=.475862|56/145=.386207|-.089655|[-.154545,-.049451]|.595755/.475673|.697172/.786264|

L passes the absolute precision floor. Both seeds fail Dead>=.60, delta>=.05, strictly-positive bootstrap lower and relative overall recall>=C-.02. Nondegenerate fraction1 and support11positive images145objects pass. Seed19's interval includes zero; seed1's is entirely negative. Two-seed mean Dead change-.079310 is descriptive, not a pooled significance estimate.

|Run|Fit recall|Fit Dead recall|Fit precision|Audit candidates|Matched/12250|Small Dead hits/32|Training hours|PeakGiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|C19|.978045|.957317|.998638|10372|7147|6|4.0272|4.0474|
|L19|.662906|.743902|.970417|7810|6065|10|4.0973|4.0474|
|C1|.978398|.963415|.998258|10468|7298|9|4.0310|4.0474|
|L1|.630611|.695122|.973704|7411|5827|6|4.0974|4.0474|

No candidate-cap saturation. Lower adaptation strength produces fewer proposals and better candidate precision while sacrificing fit and audit recall under the fixed130epoch budget. The observations are compatible with insufficient adaptation at that budget; they do not prove optimizer convergence or identify a universal regularization mechanism. Do not extend epochs to rescue this closed recipe. Small-Dead effects disagree by seed (+4/-3), so they do not provide a stable improvement.

Descriptive tissue breakdown: seed19 Colon internalDead23/56→10/56, Lung39/60→39/60; seed1 Colon17/56→12/56, Lung39/60→33/60. Other tissues have only1–15 internalDead objects each; no tissue-specific selection or thresholds were introduced. The loss is not uniform across tissues, and this tiny repeatedly reused image audit cannot identify a causal tissue mechanism.

Evidence:41smallfiles/3,025,326bytes returned to LM1; all36payloads matched project preservation byte counts and SHA256. Exactly130 finite nonnegative loss records per run; cohort membership/evaluation ordering/config/UNI/input identities verified; original summary and admission independently recomputed. Training source9a2533e1103ebd8db1ec79b2f8c2804a5588f2bb; source manifest630c58d842ee84a48f0b16bd92919412ebef68758a171ba6d102db59cbea943d. Large checkpoints remain on Expanse; writer hashes are not independent rehashing. Lustre external archival is outstanding.

This is point matching on reused fold1 development images, not maskPQ/test/independent validation or official three-split confirmation. Only522audit images,11Dead-positive images145internalDead, with image-level rather than proven patient-level separation. Original1000tissue-image bootstrap is conditional on the trained models, not uncertainty over seeds or correction for sequential selection. Historical P3–P13 controls are not pooled with this round's fresh controls.

P15 is separately frozen as a training integer-translation hypothesis, not a low-LR continuation. C/J use original encoderLR1e-5/head3e-4/full image coverage and original model/loss; only J adds +/-16 integer translation with white fill before original augmentations and correct retained-object targets/count. Original decoder/admission remain. Prior recipes stay closed. Validation and formal submission are reported separately.
