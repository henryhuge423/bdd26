# P4 result review — 2026-10-20

All four P4 runs finished130epochs with SLURM COMPLETED/0:0 (array55103975). Last-checkpoint evaluation only, original fold1 fit2134/audit522 images. Audit internalDead support11images/145objects. These are reused development images and radius3 point matches, not test/PQ results or independent patient validation.

| Arm | Seed | Audit overall recall | Unique precision | InternalDead recall | SmallDead recall |
|---|---:|---:|---:|---:|---:|
| C center-only |19|.601959|.691875|.503448|.281250|
| N + foreground BCE |19|.607265|.686001|.475862|.218750|
| C center-only |1|.582041|.694120|.448276|.218750|
| N + foreground BCE |1|.603347|.690812|.455172|.187500|

Original `proposal_p3_utils.admission`, mapping F1=C/T1=N, returns **false in both seeds**. Dead changes -0.027586 and+0.006897; tissue-stratified paired image bootstrap95%CI[-.054455,+.010753] and[-.032276,+.030864],1000draws/seed20261018. Both N Dead recalls<.60, both gains<.05, both CI lower bounds<=0, and both precision values<.75. Support/nondegeneracy and relative overall quality guards pass; these do not compensate for failed primary clauses.

Two-seed means: C/N overall recall .592000/.605306; precision .692997/.688407; internalDead recall .475862/.465517 (change-.010345); smallDead recall .250000/.203125. With only32small internalDead objects on the reused audit cohort, do not infer a universal small-object effect. No run reached the512proposal cap. Counts C19/N19/C1/N1=10658/10844/10272/10699 audit candidates.

Fit generalization gap remains substantial. C fit overall recall .982460/.976063 and unique precision .998823/.998655; N .977358/.974827 and .998657/.998774. Fit internalDead recalls C .963415/.955793 and N .961890/.949695. N audit inside-labeled-nucleus fraction .893858/.892700 exceeds unique radius3 precision, but this alone cannot identify duplicate peaks, localization error, annotation incompleteness or a causal mechanism. Do not subtract unlike point/mask metrics or use fit quality as method admission.

Losses fell in all runs. C first->last3.5023->.04902 (seed19),24.3791->.05757 (seed1); N4.4167->.13200 and26.7437->.13148. N includes auxiliary BCE and totals cannot be compared as a quality ranking. Actual run elapsed4.15-4.21hours; four array tasks total about16.81GPU-hours including preservation. Successful completion does not make the recipe successful.

**Decision:** close this foreground-BCE recipe at its frozen budget. Do not extend epochs, add seeds, change decoder/threshold, launch P3 PhaseB, or claim foreground supervision generally cannot help. Earlier P3 PhaseA also remains closed. E19 and query-line closures remain unchanged. The surviving question is how to improve actual inferred object localization/generalization with a distinct controlled hypothesis.

**Next independent experiment:** P5 equal-capacity spatial refinement, L128grid vs R256grid; seeds19/1, fresh130epochs, unchanged fit/audit/targets/optimizer/decoder/admission. This tests refinement grid, not a demonstrated diagnosis of P4. Native feature interpolation cannot restore missing image information. No automatic fullfold/PM-Net expansion, even on P5 admission.

Provenance: P4 training commit536f1e15de77482fa6a30c52c9ba4c5ccca88d2d, source manifest ac1330996d1f08724bf4e7d4af8c430177e7e1671008f7947b5ddb5fcc4abf57. Source/data/runtime retained on Expanse; final/last checkpoints preserved under the P4 project results with per-file SHA256 manifests. Thirty-five small evidence files (2,783,255bytes) collected and independently hash-checked on LM1. Four complete.json/epoch130 states checked, original admission recomputed. No new model inference, metric/HoVer-Net changes, test-fold reads or dependency changes were needed for this closeout. P4 historical full CPU suite was588passed/3skipped/5known dense_votes failures; it was not rerun for this JSON analysis.
