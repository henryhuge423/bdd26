# Visibility supervision: fixed-C2 mechanism closeout — 2026-10-15

All eight final checkpoints evaluated on 2,523 fixed E4 C2 windows each; split1 validation fold2 only. No training, added seeds, test-fold access or confirmation. Original-image frozen admission clauses 2 and 3 already fail.

## Frozen primary endpoint

Positive A1−A2 means fewer misses under selective HV supervision. Miss = maximum IoU against any predicted instance <.1 in the 192-pixel crop frame. C2 image crops resize to 256 for inference and predicted maps resize back with nearest-neighbour sampling.

| Seed | A0 miss | A1 miss | A2 miss | A3 miss | A1−A2 improvement |
|---|---:|---:|---:|---:|---:|
|19|0.474138|0.439655|0.465517|0.456897|-0.025862|
|1|0.439655|0.448276|0.439655|0.465517|+0.008621|

Mean A1−A2 improvement **-0.008621**, paired 95% CI **[-0.040918, +0.017699]**. Frozen clause1 pass=False. Requirements remain: each seed >0, mean≥.03, CI lower>0.

Bootstrap: 1,000 draws, seed20261013, tissue-stratified image clusters with identical draws across arms and the two fixed seeds, then average the per-seed pooled nuclear rate differences. As in the E4 record bootstrap, the sampling support is images containing the eligible endpoint objects. This is conditional on these two trained seeds and these label-defined objects, not a patient-level or training-seed-population interval.

## Support and definition audit

Only **116 eligible Dead objects from 38 images** support this endpoint, despite inference over 2,523 images. Tissue counts: Lung77, Colon15, Stomach7, Esophagus6, HeadNeck6, Breast3, Cervix2. Native fragment areas: <30:40, 30–<100:43, ≥100:33.

Frozen E4 grouping uses crop-border contact on originally complete objects: 5/116 Dead touch the crop edge without losing pixels (111 actually lose pixels). Training A2 uses actual pixel loss. The primary is not redefined after observing results. The same distinction affects 495/11,502 objects over all classes. Neither crop-native area bins nor this group definition are identical to the final-working-area<30 training eligibility rule.

## Original-image and crop diagnostics

Interior match means max-IoU>.5 against any prediction; miss means max-IoU<.1. Neither is type-specific detection F1. Typing accuracy below is pooled over IoU>.5 matches, all classes. Split/merged counts are GT statuses from the existing error taxonomy; no new definitions or postprocessing.

|Run|Interior Dead n|Matched rate|Miss rate|H1 area-standardized Dead gap [CI]|C0 split/merged|C0 typing correct/matched|C2 split/merged|C2 typing correct/matched|
|---|---:|---:|---:|---|---|---|---|---|
|A0_seed19|741|0.635628|0.242915|+0.18120 [+0.11585,+0.26347]|635/2840|40676/47492|667/1377|24547/28936|
|A1_seed19|741|0.623482|0.263158|+0.17538 [+0.10636,+0.25090]|642/2802|40488/47420|553/1450|24467/28852|
|A2_seed19|741|0.632928|0.267206|+0.16501 [+0.09919,+0.24407]|672/2822|40566/47500|576/1433|24544/28929|
|A3_seed19|741|0.623482|0.259109|+0.17992 [+0.11282,+0.26042]|665/2816|40567/47472|592/1450|24518/28886|
|A0_seed1|741|0.591093|0.302294|+0.14145 [+0.07727,+0.22117]|668/2830|40529/47378|663/1346|24500/28849|
|A1_seed1|741|0.607287|0.279352|+0.14139 [+0.06822,+0.22779]|654/2849|40303/47110|611/1437|24366/28687|
|A2_seed1|741|0.584345|0.299595|+0.12084 [+0.04690,+0.20379]|654/2821|40252/47179|601/1449|24351/28714|
|A3_seed1|741|0.599190|0.292848|+0.12273 [+0.05556,+0.20306]|646/2878|40293/47107|601/1481|24336/28615|

## Actual training exposure (not unique objects)

Across all130 epochs, each A2 run logged **33,424 eligible small newly truncated exposures (seed19)** and **33,660 (seed1)**; zero all-zero low-threshold HV targets logged in these eligible sets. A1/A2/A3 logged count vectors are identical at every epoch within each seed. This supports matched exposure accounting, not proof of every individual augmented image identity. Counts are all-class repeated exposures; logs do not provide Dead-specific counts. A3 counts describe the same selective eligibility subset, not all additional A3 small-object interventions.

The earlier one-augmentation-per-image audit (A2 touches7 Dead vs A3 touches70) is a different deterministic label audit, not the actual130-epoch exposure distribution. It cannot justify a claim that A2 never meaningfully changed targets.

## Interpretation and decision

The tested selective-target intervention does not meet the frozen crop-mechanism criterion. E4 showed crop-associated vulnerability; it did not establish that HV zeroing caused that vulnerability. This training experiment does not establish a sufficient rescue under the tested recipe, and does not exclude every visibility-aware or HV-based approach.

Descriptive A3−A2 miss difference (positive favors A2): +0.008621 [-0.014713,+0.034834]. An interval containing zero is not equivalence or proof of indistinguishability. No specificity claim follows solely from it.

**Decision: stop this dev recipe; no extra seeds, altered gates or three-split confirmation.** PM-Net remains a proposal. Train-only feasibility work found276 instances in one patch and23 rounded centers outside their own masks; see [geometry audit](PMNET_TRAIN_GEOMETRY_AUDIT_2026-10-15.md).

## Reproduction and validation

First set `CUDA_VISIBLE_DEVICES` to one resource-admitted GPU index (a single device, not a comma-separated list), then run:

`~/.conda/envs/nuclei/bin/python scripts/analyze_visibility_supervision.py --fold 2 --out NEW_OUTPUT_DIRECTORY`

Runner refuses other rounds/folds and existing outputs. Uses corrected E4 geometry and inference helpers, eval mode, no-grad, bf16 forward, batch16 and CPU8workers. Each checkpoint is resource-admitted serially on one GPU. No scientific metric or HoVerNet implementation changed. The CUDA allocation field is the process cumulative peak, not separately reset per-run peak.

Raw compressed C0/C2 rows, summary diagnostics, checkpoint/config/image/GT/prediction/geometry hashes, primary comparison, support/definition audits and status are under `runs/analysis/visibility_supervision_mechanism_20261015/`. Code and source manifest are in ignored ops handoff notes. Focused tests:52 passed (19 pre-existing deprecation warnings); independent pre-execution review found no blockers. All retained source/input hashes and paired endpoint recomputation were checked during report construction.
