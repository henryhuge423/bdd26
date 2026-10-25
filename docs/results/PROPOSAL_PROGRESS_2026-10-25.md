> Latest closeout: [P14](PROPOSAL_P14_RESULTS_2026-10-25.md) verified41smallfiles36payloads; precision rises but fit/audit recall falls; bothoriginalgatesfail. The P14 design text below records its pre-result rationale. P15 independently tests training integer translation with originalLR/model/loss/cohort and original admission; smoke validation is pending, no claimed improvement. Full boundaries in [P15 frozen design](../superpowers/specs/2026-10-25-proposal-p15-design.md).

# Proposal research review — 2026-10-25

The tested P3–P14 recipes have not produced an admitted proposal detector. This is a bounded negative result on repeatedly reused fold1 development images, not a conclusion that point detection, foundation encoders or class balancing cannot work. P13 is newly verified here; earlier rows below summarize their committed closeout reports rather than independently rehash all historical artifacts again.

|Round|Fixed treatment vs its own fresh control|Internal Dead recall change seed19 / seed1|Original admission|
|---|---|---|---|
|P4|Dense foreground auxiliary BCE|−.027586 / +.006897|false|
|P5|Matched-capacity native-grid refinement|.000000 / −.006897|false|
|P6|Wider negative-suppression Gaussian|+.020690 / +.020690|false|
|P7|Training RGB gains|+.034483 / −.013793|false|
|P8|Per-image nucleus normalization|−.034483 / −.048276|false|
|P9|Dead-center positive focal weight4|+.006897 / +.048276|false|
|P10|Last4 rather than last12 adaptation|+.027586 / −.013793|false|
|P11|Positive focal mass redistribution|+.041379 / .000000|false|
|P12|Post-stem channel dropout.2|−.034483 / −.027586|false|
|P13|Fixed25%rare-positive replacement|−.137931 / −.048276|false|
|P14|Encoder initialLR1e-6 instead of1e-5|−.068966 / −.089655|false|

P3's original T1−F1 was+.068966/+.013793 but failed absolute Dead, precision and uncertainty clauses. The later within-round comparisons must not be pooled with those arms, ranked as a hyperparameter search, or treated as independent patient replications. Fresh controls vary; only each round's own matched control enters its original admission.

The consistent practical gap is fit quality versus held-out proposal reliability. P13 raises fit internal-Dead recall to.998476/1 while audit falls to.351724/.358621. P10 and P12 instead reduce fit recall without sufficient audit improvement. P9 and P11 have some positive changes but remain below original absolute recall/precision requirements; a positive interval in one seed does not satisfy the two-seed conjunction. Thus neither added rare exposure nor the tested capacity/regularization/target modifications resolves the frozen precision/recall gap. These observations motivate controlled hypotheses, not a proven causal mechanism.

Audit support is only11Dead-positive images/145internalDead,80%in5Lung/Colon images in the prior descriptive decomposition. Image-level rather than patient-level separation, sequential hypothesis selection and repeat use of522audit images constrain inference. Conditional tissue-image bootstrap intervals do not repair these limits or represent uncertainty across random seeds. All endpoints here are point matching, not maskIoU/PQ/test or official three-split performance. Earlier E-series mask scores and three-split baselines cannot be subtracted from these point rates as pipeline loss.

P14 separately freezes one adaptation-strength control: keep last12 blocks and original full image coverage, reduce only encoder initial LR1e-5→1e-6, same head3e-4 and cosine schedule. This avoids interpreting P10's restricted trainable scope as a learning-rate result; it is not a rescue or extra menu for any closed recipe. Fourfresh130epoch runs C/L×19/1 retain original cohort, loss, augmentation, last checkpoint, decoding and admission. Real V100 parity/optimizer/AMP/reload/frozen/source/timing checks gate submission. No claimed scientific improvement before results, no automatic masks/PhaseB/fold2/fold3/extra seeds.

Detailed original sources: [P3](PROPOSAL_P3_PHASE_A_2026-10-20.md), [P4](PROPOSAL_P4_PHASE_A_2026-10-20.md), [P5](PROPOSAL_P5_RESULTS_2026-10-21.md), [P6](PROPOSAL_P6_RESULTS_2026-10-21.md), [P7/P8](PROPOSAL_P7_P8_RESULTS_2026-10-22.md), [P9](PROPOSAL_P9_RESULTS_2026-10-23.md), [P10](PROPOSAL_P10_RESULTS_2026-10-23.md), [P11](PROPOSAL_P11_RESULTS_2026-10-24.md), [P12](PROPOSAL_P12_RESULTS_2026-10-24.md), [newly verified P13](PROPOSAL_P13_RESULTS_2026-10-25.md), [P14 design](../superpowers/specs/2026-10-25-proposal-p14-design.md).
