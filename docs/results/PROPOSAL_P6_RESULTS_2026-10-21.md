# P6 center-target bandwidth: completed, admission failed

2026-10-21 America/New_York. Four array55148533 tasks COMPLETED/0:0, each130epochs, Slurm stderr0bytes. C19/W19/C1/W1 elapsed4:17:18/4:12:09/4:15:37/4:14:30; summed16h59m34s of one-GPU task allocation. Scheduler timezone is preserved in raw evidence; UTC retrieval2026-10-22T01:45 (21:45EDT October21).

40smallfiles totaling2,963,361bytes returned to LM1;36payload SHA256/size values match project preservation manifests, and all four input/weight/cohort identities agree. Each history contains exactlyepochs1–130, finite nonnegative loss, complete exit0/130. Large checkpoints remain on Expanse; no new model inference or fold2/fold3 evaluation.

| Seed | Arm | Fit recall | Audit recall | Audit unique precision | Audit internalDead recall | SmallDead hits/32 |
|---|---|---:|---:|---:|---:|---:|
|19|C|.973591|.588653|.689520|68/145=.468966|8|
|19|W|.995350|.614041|.660346|71/145=.489655|7|
|1|C|.980596|.597061|.691827|68/145=.468966|8|
|1|W|.996763|.615510|.673455|71/145=.489655|7|

Original unmodified admission mapping F1=C/T1=W is false for each seed. W-C Dead increase+.020689655 in both seeds, below+.05; original tissue-stratified paired1000draw95%CI seed19[-.02151719,+.06172840], seed1[0,+.05309735] fail strictly positive lower bound. W absoluteDead .489655<.60 and precision.660346/.673455<.75. Seed19 precision change-.029174 also violates relative-.02 guard; seed1 change-.018373 passes that clause. Both pass overall-recall relative floor, support11images/145objects, and nondeg1.0. Report all clauses rather than choosing the improving endpoints.

W improves training fit and descriptive overall audit recall, with lower unique candidate precision; larger target halos can change off-center/duplicate output, but these aggregates do not establish which error mechanism is responsible. SmallDead recall drops from8/32 to7/32 in both seeds; the subset is small and descriptive. Gaussian halo values were negative-penalty suppression, not additional positive labels. Loss magnitudes across arms therefore differ in objective weighting and cannot rank model quality. Observed training peaks4,345,896,960bytes (~4.05GiB) each.

Close this bandwidth recipe at this budget. Do not add epochs/seeds, retune thresholds or initiate original P3PhaseB/mask training. P3/P4/P5 closures remain. The repeatedly reused fold1 internal audit is development evidence, not independent validation, test/PQ, or official three-split performance; no broad method-family rejection follows.

P7 is a separately frozen engineering augmentation control: identical C versus training-only per-image independent RGB gains[.8,1.2], retaining original model/loss/center targets/cohort/seeds/epochs/decode/admission. It tests the full augmentation recipe (including brightness variation), not a proven stain or generalization mechanism. Expanse source/runtime/data reused only after identity checks and real V100 smoke; max48GPU-hours for four formal runs.

Provenance: trainingcommitc1b3bc9b7d1c942b2cd1d0b06c19eaa4fc47054e, source manifestf41c9868070894a9d93dc5028da8d74d2abbdd059992e77d97aa6463558c103e; cohort55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22. Operational receipts/logs/collection/gate/analysis in LM1 ignoredops/review_20261021_evening and ops/p6_development_20261021. Source remains on research/proposal-p6 documentationHEAD5fca107; current report is recorded on the next research branch. No shared branch merge/push.
