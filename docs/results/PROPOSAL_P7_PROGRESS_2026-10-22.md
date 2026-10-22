# P7 RGB gain: verified partial results and research interpretation

2026-10-22 America/New_York. Evidence cutoff08:09EDT (12:09UTC). Array55176183: C19/J19 COMPLETED/0:0,130epochs each; C1/J1 still RUNNING, last completed epochs89/85. All four Slurm stderr files were empty. This report is explicitly partial; no completed two-seed result is claimed.

## Verified completed seed19

20smallfiles totaling1,481,539bytes returned to LM1;18payloads match the per-run preservation manifests by SHA256 and size. Exact fit2134/audit522, input/UNI weight/cohort identities agree. Histories contain epochs1–130 exactly with finite nonnegative losses. Last checkpoints, unchanged decode and original point matching; no new model inference or fold2/fold3 evaluation.

| Endpoint | C | J | J−C |
|---|---:|---:|---:|
| Fit overall recall | .975612 | .949341 | −.026271 |
| Fit unique precision | .998835 | .996048 | −.002787 |
| Fit internalDead recall | .942073 | .922256 | −.019817 |
| Audit overall recall | .587918 | .594367 | +.006449 |
| Audit unique precision | .690906 | .714244 | +.023338 |
| Audit internalDead recall |71/145=.489655|76/145=.524138|+5/145=.034483|
| Audit smallDead recall |7/32|8/32|+1/32|
| Audit candidates |10,424|10,194|−230|
| Images reaching512candidate cap |0|0|0|

Original tissue-stratified paired1000draw95%CI for J−C internalDead recall: [.00543478,.07964602], RNG20261018,11Dead-positive images/145objects. It is a reused development cohort; this positive interval is not independent validation or a multiplicity-corrected discovery.

Original seed19 admission fails: absoluteDead .524138<.60, improvement .034483<.05, precision .714244<.75. Its positive CI, relative precision/recall floors, support and nondegeneracy pass. Report all clauses. Seed1 remains pending; a full two-seed table and original full admission are not yet available. Since the rule requires both seeds to pass, the completed seed19 failure already prevents admission of this fixed recipe, but retain the other jobs and report their final outcomes before full closeout.

Trainable peak memory4,345,896,960bytes (~4.05GiB) for both completed runs; GPU task elapsed4:15:07/4:13:53. Loss values are descriptive within each objective; J changes training appearance and its higher training loss is not evidence of inferior final quality. SmallDead has only32objects and one extra hit, so avoid subgroup claims.

## What the result supports

For seed19, the full RGB-gain recipe improves both held-out overall recall and precision while reducing fit recall. That pattern is compatible with regularization but does not identify a causal mechanism, optical stain robustness or a stable two-seed gain. There is no evidence here for a candidate-cap bottleneck. Other baseline repeats are not numerically identical despite matching configurations; compare within each study rather than attributing between-study baseline shifts to augmentation. Long-run nondeterminism/hardware effects have not been isolated.

P3's trainable augmentation condition improved overall recall over frozen features but failed precision/Dead admission. P4 foreground auxiliary supervision, P5 equal-capacity grid refinement and P6 broader center targets also failed the original gate. These negative results bound the tested recipes, not whole method families. The persistent gap is converting excellent fit into simultaneous held-out recall and precision under a fixed decoder.

## Independently frozen next question

P8 was frozen and uploaded before reading these P7 endpoint scores. C retains effective-batch total-object normalization; I averages the original per-image focal sum/max(1,true object count), including empty-image full negative terms and actual tail batch size. This is an objective-weighting experiment, not class/tissue balancing. Same original augmentation (no P7 gains), data/seed/130epochs/architecture/decode/admission. No new thresholds or mask stage. [Frozen P8 design](../superpowers/specs/2026-10-22-proposal-p8-design.md).

Provenance: P7 trainingcommit73190915d67f658d4573e5f601b9cdc48a0f0079, source manifest83947384b842c3989fda71e2c6663b2fb958742e567dc113057003e23f5a9a45, cohort55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22. Collection/analysis/gate/live-state evidence remains in LM1 ignored ops/review_20261022. Full checkpoints remain on Expanse project/scratch. Point-development endpoints are not maskPQ,test or official three-split performance.
