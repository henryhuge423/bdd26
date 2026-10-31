## 2026-10-30 — P21 operational recovery (no new scientific result)

Original C19 failed in quota preflight before training; three unstarted tasks replaced by array 55440148. Frozen source and research protocol unchanged. Storage guard uses bounded retries with invalid-data rejection. Focused regression28passed. [Details](results/p20_p21_2026-10-30/P21_RECOVERY_2026-10-30.md).

> **2026-10-30 — P20 closed; P21 submitted:** P20 four130epoch runs completed with42small payload/source checks and426sealed-source checks. Original two-seed gate failed despite Dead gains+.089655/+.041379; close fixed512recipe. P21 training-only semantic-center auxiliary C/A×19/1,original256/cohort/loss denominator/last checkpoint/gate. V10055416052COMPLETED0:0/746s,actual control/learning/AMP/reload/visual checks passed; fullCPU 968 passed, 3 skipped, 19073 warnings in 1426.74s (0:23:46). Formal array **55416129**,four130epoch tasks,concurrency2,max48GPUh,submitted once. Submission is not completion. [EN/ZH/reader](results/p20_p21_2026-10-30/README.md),[validation/receipt](results/p20_p21_2026-10-30/P21_SUBMISSION.md). Point examples auto-save at final checkpoint; smoke previews are one-epoch fit-only. Reused fold1 point development,notPQ/test; no masks/PhaseB/menu.

> **2026-10-28 — P19 closed; P20 submitted:** P19 four130epoch runs completed; original Dead B−C gate failed both seeds (−.062069/0). Close fixed effective32 recipe. P20 input256/512, effective16/physical4: fullCPU 950 passed, 3 skipped, 19073 warnings in 646.58s (0:10:46); original smoke55343640 failed the U24hour timing guard, followed by resource-only C12/U36 redistribution with unchanged96GPUhour total. Third validation55357273 passed after the separately frozen1GPUhour validation budget amendment;55356965 TIMEOUT remains preserved. Formal arrays **C55357457 / U55357476**, combined concurrency2, four fresh130epoch runs. Validation actual acceptance increased from.5 to1GPUhour after the recorded849second TIMEOUT; formal96GPUhours unchanged. Submission is not completion. [EN/ZH reader](results/p19_p20_2026-10-27/README.md), [actual validation/resource amendment/receipts](results/p19_p20_2026-10-27/P20_SUBMISSION.md). Reused fold1 point development, notPQ/test; no automatic PhaseB/masks/menu.

> **2026-10-27 — P18 closed; larger-batch P19 submitted:** All four P18 130-epoch runs completed; both original gates failed (Dead H−C −.006897/+.020690). Fitting coverage improved without qualified rare-class transfer. V100 validation55326654 passed; fitting-only microbatch16 improved measured throughput12.27% over4. Formal effective16/32 ×seeds19/1 array55326699 submitted once, concurrency2, max48GPUh; submission is not completion. [Detailed EN/ZH/Chinese reader](results/p18_p19_2026-10-27/README.md), [P19 protocol](results/p18_p19_2026-10-27/P19_SUBMISSION.md). Reused fold1 point metrics, not PQ/test; no automatic PhaseB/masks/menu.

> **2026-10-26 evening — P17 closed; P18 submitted:** Four P17 130-epoch runs completed/0:0; original admission failed in both seeds. The fixed scratch CNN has low fit/audit coverage despite high precision. Independently frozen RGB residual P18 passed V100 smoke 55310632 and formal array 55311007 was submitted once (C/H x19/1,concurrency2,max48GPUh). Submission is not completion. Reused fold1 point metrics are not PQ/test evidence; no automatic masks/PhaseB/menu. [P17 results](results/PROPOSAL_P17_RESULTS_2026-10-26.md), [P18 protocol](results/PROPOSAL_P18_SUBMISSION_2026-10-26.md), [中文分析](results/RESEARCH_REVIEW_2026-10-26_EVENING_ZH.md), [中文导读](results/RESEARCH_READER_2026-10-26_EVENING_ZH.md).

# Experiment findings

> **2026-10-26 — P16 partial results, P17 submitted, stage report 6:** P16 seed 19 C/R completed 130 epochs and passed preservation/source checks. Dead matches changed from 70/145 to 75/145, with original CI [-.034091,+.131144]; this seed failed admission. Seed 1 remains running. Independent P17 passed V100 smoke 55297045, 291 node checks and strict readiness. Formal array 55297061 was submitted once at 09:08:19 EDT: four 130-epoch tasks, concurrency 2, maximum 48 GPU-hours. Submission is not completion or admission. These are reused fold-1 point metrics, not PQ/test results. No automatic Phase B, masks or menus. See [P16 partial results](results/PROPOSAL_P16_PROGRESS_2026-10-26.md), [P17 submission](results/PROPOSAL_P17_SUBMISSION_2026-10-26.md), and [Stage report 6: English, Chinese and Chinese reader](report/stage_report_6/README.md).

The [current results index](results/README.md), [P3–P15 synthesis](results/PROPOSAL_PROGRESS_2026-10-26.md)
and [documentation audit](audits/DOCUMENTATION_RESULTS_GIT_AUDIT_2026-10-26.md) provide the latest overview.
P13–P15 failed their original gates; P16 submission is verified, with completion not rechecked
in this documentation audit. The full dated journal below preserves the historical order
and execution states; later closeouts supersede earlier pending statements.

> **2026-10-25 — P14 submitted:** Fixed encoder learning rate 1e-6 versus original 1e-5, C/L × seeds 19/1, four fresh 130-epoch runs. V100 smoke job 55270073 completed/0:0; 214 node checks passed, including exact-control parity, actual learning rate, AMP, reload, frozen parameters, timing, and strict readiness. Formal array 55270432 was submitted once at 2026-10-25T04:16:21.488338-04:00, concurrency 2, maximum 48 GPU-hours. Submission is not training completion. See the [protocol](results/PROPOSAL_P14_SUBMISSION_2026-10-25.md) and [P3–P13 synthesis](results/PROPOSAL_PROGRESS_2026-10-25.md). Reused fold-1 development data are not PQ or test results.

> **2026-10-25 — P13 closed:** All four 130-epoch runs completed/0:0. Fixed 25% rare-positive replacement worsened audit Dead recall by .137931/.048276 despite fit recall .998476/1; both original gates failed. See the [final result](results/PROPOSAL_P13_RESULTS_2026-10-25.md). P14 independently freezes encoder learning rate 1e-6 versus original 1e-5 with the same last 12 blocks; validation/submission was pending. Reused fold-1 development data are not PQ or test results.

# 2026-10-24 P12 formal submission verified

Array 55251406 was submitted once at 2026-10-24T07:20:40.533905-04:00: C19/R19/C1/R1, four fresh 130-epoch runs, maximum 48 GPU-hours. Smoke job 55251401 completed/0:0; 153 checks passed, stderr was empty, and all strict-readiness checks passed. Source commit 77cfa9ac17b278c67b750762191e6bdd5b9b4cc9; manifest 35748e383fa57d626c3c8a03fc09871627872c19599d2a3003559a1845bb1a3d. No formal completion or performance claim. See the [full review](results/EXPERIMENT_REVIEW_2026-10-24_ZH.md).

# 2026-10-24: P11 complete, original admission failed

All 4 runs completed 130 epochs (SLURM 0:0; empty stderr). We verified 40 small files totaling 2,985,074 bytes and 36 payload-preservation hashes; config/cohort/input/UNI were identical. C/G Dead recall was .468966/.510345 (seed 19) and .462069/.462069 (seed 1); G−C changes were +.041379/0. Original paired 95% CIs were [.025381,.075310]/[−.038462,.035398]. Both seeds failed the absolute Dead .60, gain .05, and precision .75 thresholds; seed 1 also failed the CI gate. Relative precision/recall, support, and nondegeneracy passed. Close the exact mass-redistribution recipe. C fit recall .975/.980 versus G audit recall .576/.579 is not causal evidence. No PQ/test, fold2/fold3, or new mask stage. See the full [results](results/PROPOSAL_P11_RESULTS_2026-10-24.md).

P12 applies decoder `Dropout2d(.2)` after stem ReLU, C/R × seeds 19/1, four new 130-epoch runs, original gates. It was chosen by the agent under the user's autonomous experiment request. Design review checked initialization RNG, per-update RNG traces, and the ReLU-zero confound; all were implemented. Baseline: 139 passed in 12.46 s; RED: missing P12 module; focused GREEN: 152 passed in 14.98 s. No metrics/HoVer-Net edits; historical 5 dense-vote failures are disclosed, and no full-suite pass is claimed. Real V100 smoke/formal submission was pending. Evidence: ignored `ops/p12_development_20261024` and `ops/review_20261024`.

> **2026-10-23 — P10 submitted:** Restricted encoder adaptation, C (last 12 blocks)/S (last 4 blocks), seeds 19/1, four fresh 130-epoch runs. Formal array 55227144, concurrency 2. V100 smoke 55226781 completed/0:0; 119 focused tests, parity, frozen parameters, reload, AMP, timing, and strict readiness were verified. Submission is not training completion. See the [submission and protocol](results/PROPOSAL_P10_SUBMISSION_2026-10-23.md). P9 closed under the original gate. Reused fold-1 development data are not PQ or test results.

> **2026-10-23 — P9 closed:** All four 130-epoch runs completed; original admission was false for both seeds. Dead-positive weight 4 changed audit recall by +.006897/+.048276 but failed the absolute Dead, precision, and CI gates. See the [audited results](results/PROPOSAL_P9_RESULTS_2026-10-23.md). P10 restricted encoder adaptation was independently frozen and submitted; see the latest entry. Reused fold-1 development data are not PQ or test results.

## 2026-10-22 evening: P7/P8 closeout and independent P9 submission

> **2026-10-22 evening:** All eight P7/P8 130-epoch runs completed, and both original two-seed gates failed. See the [final results](results/PROPOSAL_P7_P8_RESULTS_2026-10-22.md). The P9 fixed Dead-center positive-weight-4 control passed 98 focused checks and real V100 smoke 55208331. Formal array **55208619** was submitted once at 19:55:04 EDT for four fresh 130-epoch runs. See the [submission and limits](results/PROPOSAL_P9_SUBMISSION_2026-10-22.md). Training completion is not claimed; no automatic Phase B, masks, or confirmation.

# P9 Dead-center weighting: verified submission — 2026-10-22

Formal array **55208619** was submitted once at **2026-10-22 19:55:04 EDT**. Submission is not completed training or an experimental result. C19/D19/C1/D1 are four fresh 130-epoch fold-1 development runs; original admission F1=C/T1=D remains frozen. No automatic Phase B, mask, fold2/fold3 evaluation, or confirmation.

## Frozen intervention

C is the original center-focal/actual effective-batch instance count. D multiplies only annotated Dead-center positive focal terms by 4; original negative terms and denominator remain unchanged. This includes boundary Dead, with a maximum of 4 on collisions. Original D4 geometry synchronizes positive weights without consuming image/target RNG. Inference remains type-agnostic and unchanged. This independent point-loss recipe differs from the failed historical dense-model Dead-pixel weighting; it does not establish imbalance or gradient conflict as a cause. See the [design](superpowers/specs/2026-10-22-proposal-p9-design.md) and [P7/P8 closeout](results/PROPOSAL_P7_P8_RESULTS_2026-10-22.md).

Original fit/audit: 2134/522; cohort 55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22; seeds 19/1; 130 epochs; original augmentation, UNI/head/optimizer/BF16, final checkpoint, decoding, and original admission retained. Both seeds must meet all original Dead recall/gain/CI/precision/relative-quality/support gates. Failure or inconclusive results close this fixed-weight-4 budget without a weight sweep or extra runs. Reused development audit: 11 images/145 interior Dead objects; two seeds and image-level splitting limit evidence. No PQ/test/independent validation claim.

## Validation

Original: 75 focused checks passed in 12.67 s. Initial new-module tests RED; original implementation: 96 passed in 11.47 s. Predeployment scan found 0 Dead objects in the first 256 fit images (780 Dead in the full fit set), so the inherited smoke would not exercise real weighting. Before any P9 GPU submission, freeze a validation-only smoke set: first 240 fit images plus the first 16 distinct later fit images containing any Dead, identical for C/D/verifier. Two membership checks were RED; final focused P3–P9 suite: **98 passed in 11.58 s**. Formal complete-fit membership/order never changed. Independent design and code review found no blockers. No metrics/HoVer-Net edits; historical 5 dense-vote failures are disclosed; no full-suite pass is claimed.

Real V100 smoke **55208331 COMPLETED/0:0**. Node-focused suite: 98 passed in 293.72 s. Initial states matched in both seeds. C/P3's two real effective optimizer steps had bitwise-equal trainable gradients, adapted parameters, and AdamW states. Synthetic-GPU D values/gradients matched the analytic results. Both arms had nonzero finite encoder/head updates and gradients. Internal BF16 and terminal bilinear FP32 were verified; last/final reload predictions matched. Identical 256 smoke-fit IDs were recorded. D observed 461 augmented weighted-positive pixels; C had 0; all mapped to target==1. No audit predictions.

|Measure|C|D|
|---|---:|---:|
|1.5x safety estimate for 130-epoch runs (hours)|7.077|7.033|
|Training peak (GiB, before reload)|4.060|4.060|
|Reload reconstruction and verification (seconds)|5.384|5.253|

Both pass the unchanged 12 h timing gate; estimates do not guarantee runtime. Strict readiness binds source/scripts/input/cohort/UNI plus tests/parity before formal submission. The same pending smoke ID was moved from gpu-shared to legalgpu-debug/gpu-debug-normal with identical 30 min/1 GPU/4 CPU/32 GiB resources and sealed script hashes; no cancellation, new submission, or source reseal. Override receipt retained. The official [SDSC guide](https://www.sdsc.edu/systems/expanse/user_guide.html) and current allocation/quota/queue/QoS were checked.

## Submission provenance

Training source commit 051082692dce2dd1f29a8b6383d9cf4a9dad1988; manifest 85ef261efe90a495cbdbf16a6465369cbc076f8eebfe5c0c4585727e1d9edb32. Formal array 0–3%2, mapping 0=C19, 1=D19, 2=C1, 3=D1; per task: 1 V100/4 CPU/32 GiB/12 h, gpu-shared/gpu-shared-normal/jhu159, Lustre, no requeue, maximum 48 GPU-hours. Durable submission guard/job ID/receipt saved. Do not rerun once-only scripts; inspect exact IDs before action. Shared runtime/data/UNI reused without upgrades or large transfers. Completed results were copied with project quota guards and a preservation manifest while retaining scratch; Lustre is not archival backup.

P7/P8 both close under the original gates. Main checkout, all previous branches/output, and unrelated jobs were preserved. Research branch remains isolated; no merge/push. After all four P9 runs complete, verify 130 epochs, exit codes, preservation/input/config hashes, collect small artifacts, and apply the original admission rule once. No automatic expansion.


> **2026-10-21 — P5 closeout:** All four 130-epoch L/R runs for seeds 19/1 completed. Original admission was false for both seeds: R−L internal Dead recall changed by 0.000000/−.006897, paired 95% CIs [−.032680,+.053846]/[−.031579,+.013514]; R unique precision .685397/.691238. Fit recall >.98/precision >.998 did not transfer. Close this matched-capacity native-grid refinement recipe. See [P5 results](results/PROPOSAL_P5_RESULTS_2026-10-21.md). The independent P6 center-target bandwidth control passed real V100 smoke 55148203 and was submitted once as array 55148533 at 09:56:52 EDT, C/W × seeds 19/1, four 130-epoch runs, concurrency 2; it was initially PENDING and results were pending. No Phase B or fold2/fold3 model evaluation.

# Findings log

## 2026-10-17 — E17/E18 closeout

E17's fixed sparse heatmap detector produced zero candidates across 16 images and two seeds. E18 retained the conditional-mask readout and replaced only the point detector with dense foreground-to-instance-point voting. GT-point fit recall exceeded 80% for all/Dead objects on fit images in both seeds, but actual detector fit recall was 34.18%/41.84%; fit precision was 3.67%/4.26%. Audit precision was 3.67%/3.52%. Seed 19 lost 5.26 pp audit Dead recall; seed 1 gained 21.05 pp but lost 15.06 pp precision. Both allocation gates and A0 guards failed. A0 matches E11 on the selected images, and the independent result verifier passed. This is train-only evidence (audit images were seen by E7; 19 audit Dead objects), not independent validation/test. Stop at E18; no large training. See the [E7–E18 synthesis](results/QUERY_POINT_CLOSEOUT_E7_E18_2026-10-17.md), [E17](results/PREDICTED_POINTS_E17_DIAGNOSTIC_2026-10-17.md), and [E18](results/DENSE_VOTES_E18_DIAGNOSTIC_2026-10-17.md).


On 2026-10-17, E12/E13's GT-assisted fixed-feature fitting witness passed (Dead 86.32/87.37%); this was not method admission. Freezing the mask projection failed both quality gates. E14 exact-native matching was completed later, but did not pass the allocation or A0 guard. No large training. See [E12](results/FREE_MASK_E12_DIAGNOSTIC_2026-10-17.md) and [E13](results/STABLE_MASK_E13_DEV_2026-10-17.md).

E11 (2026-10-17): Four-stage auxiliary supervision failed both original allocation and A0 guards (two seeds, 300 steps). Full CPU suite: 512 passed/3 skipped. No large training. See the [results](results/AUX_QUERY_E11_DEV_2026-10-17.md).

 > **2026-10-17 E10:** Predicted-mask constrained attention failed both train-only allocation gates and the separate A0 guard. Audit Dead recall M−G changes: −.067511/+.008439; overall recall: −.095944/−.056694. A0 was reproduced exactly in both seeds before reusing E9 G. Stop this 300-step recipe; no expanded training. Full regression: 507 passed/3 skipped. See [E10 results and mechanism limits](results/MASKED_QUERY_E10_DEV_2026-10-17.md).

 > **2026-10-17 E9:** Local-content initialization failed the frozen train-only allocation gate in both seeds. L−G audit Dead recall gains were +.004219/+.018987, with precision losses −.027199/−.023349; fit recall was .134321/.168889. Both adapted arms also underperformed frozen A0 on audit geometry. Stop this recipe; no full train/validation or large training. Current-source regression: 501 passed/3 skipped. See the [full E9 review](results/LOCAL_QUERY_E9_DEV_2026-10-17.md).

## 2026-10-17 E8a/E8b query mechanism closeout (train-only)

E8a fixed-query classifier probe: A2 fit foreground accuracy 95.17%/97.58%, but audit foreground changes −7.69/−6.87 pp and final untyped recall gains +2.20/+2.47 pp failed the frozen allocation rule; stop this balanced-head recipe. Both fixed batches contain zero Dead. E8b separately audits all 65 Dead-positive fold1 images/967 Dead (801 interior, 166 border) with original E7 Q19/Q1: any-raw IoU>.5 coverage 35.57%/36.40%; final typed matches 2.79%/12.20%; canonical background 76.32%/32.37%. Raw coverage is not a strict final-decoder ceiling (8/37 raw-bad objects improve after competition). No validation/test performance, additional full training, or saved weights.

See [complete results and limits](results/QUERY_MECHANISM_2026-10-17.md). Independent per-object recounts passed. Final full CPU suite: 494 passed/3 skipped/19073 warnings, exit 0. Invalid import-route, bf16 flattened-input, and root-full fixture-error runs are preserved separately. Existing E7 source/weights and scientific gates were unchanged.


## 2026-10-16 E6 complete; train-only optimization probe; E7 allocation

All 4 B/Q × seed 19/1 runs completed using final checkpoints; fold-2 development only. Frozen 741-object primary endpoint, Q−B = −.507422 (95% CI [−.563164,−.451449]); every quality criterion failed, while latency passed (.7712/.7955×). Two-seed mean B/Q mPQ = .481432/.102699, bPQ = .650773/.192614, Dead PQ = .179635/.005951. Paired 130-epoch data digests and returned completion hashes were verified. E6 recipe closed; no threshold/seed expansion or test confirmation.

Fixed 16-image fold1/Q19-final trainability probe; identical Adam state and frozen encoder, 300 steps each. Resetting LR to 3e-4 reduced evaluation loss 36.24%; assigned-decoded recall changed from 18.73% to 36.86% versus 20.24% for the terminal-LR control. The allocation gate passed; this does not prove schedule causality, gamma .99 validity, or generalization. The independent E7 round changes only exponential gamma .85→.99; both B/Q arms were retrained from matched original initialization with the same seeds and quality/cost gates. Formal queue launched on LM2 GPU6 after full CPU 487 passed/3 skipped and a fresh admitted smoke; first B19 epoch completed. See the [complete analysis](results/SET_MASK_E6_DEV_2026-10-16.md). No metric/HoVer-Net code changes, dependency upgrade, or publishing.

## 2026-10-15 — point/mask representation feasibility, fold1 only

New CPU analysis covers 63,218 training instances (967 Dead). Under a hard block-mask, nearest-upsample representation, stride 4 cannot reach IoU > .5 for 65 Dead instances, including 29/801 interior instances; stride 8 affects 460 Dead instances, including 373 interior instances. This is not a bound for bilinear logits or native refinement, and it does not explain the existing baseline mechanism. Fixed stride-4 3x3 point-neighbor conflicts involve 191 targets/19 Dead; 94 vertex-disjoint edges provide a representation-exclusion lower bound under strict local-winner decoding. HV30 is a target-support rule, not a final-mask area filter. Independent checks on 438 real objects and all-image sparse-graph reconstruction agree. [Analysis](results/POINT_MASK_FEASIBILITY_2026-10-15.md), [next draft](proposals/RESEARCH_NEXT_2026-10-15-instance-representation.md). No model fitting/inference/fold3; existing stop rules unchanged.


## 2026-10-15 — fusion gating frozen-gate failure and point-grid feasibility

Both G-seed runs are complete. Interior-Dead matched changes were −.032389/+.025641; the mean was −.003374 [−.014273,+.007486], so clause 1 fails. Mean Dead PQ changed by +.017897, and strict Dead PQ by +.014202; clauses 2/3 pass. The actual mean miss-rate change was +.009447 (worse); `comparison.miss_improvement` in the frozen artifact is an invalid auxiliary complement of matching and was not used for admission. Stop this recipe; no C controls or extra seeds. [Full results](results/FUSION_GATING_DEV_2026-10-15.md). A new fold-1-only point audit covers 63,218 objects and legal nearest-visible-centroid points; single-slot excess for strides 2/4/8 is 0/1/45. [Audit](results/PMNET_POINT_GRID_AUDIT_2026-10-15.md). No new training/inference/test access.

## 2026-10-15 — fixed-C2 mechanism closeout and train-only geometry

Eight final checkpoints and 2,523 fixed E4 C2 windows are complete. Primary Dead support is 116 objects across 38 images (77 in Lung; 5 border contacts without pixel loss). A1/A2 misses were 51/54 for seed 19 and 52/51 for seed 1. Mean A1−A2 improvement was −.008621 [−.040918,+.017699], so frozen clause 1 fails; the original-image guards already fail. The descriptive A3−A2 change was +.008621 [−.014713,+.034834]; this does not establish equivalence. Stop the recipe; no added seeds, tuning, confirmation, or test-fold access.

Actual 130-epoch A2 eligible exposures were 33,424/33,660, with all-class repeats; there was no Dead-specific training count. The single-view audit’s 7/70 Dead exposure is a separate measure. PM-Net fold-1 geometry: 63,218 instances, maximum 276, and 20 instances overflowed the 256-query limit in one patch; 23 rounded centers fell outside their own masks (1/967 Dead), and 116 images were empty. No new model training. See [mechanism details](results/VISIBILITY_SUPERVISION_MECHANISM_2026-10-15.md) and the [geometry audit](results/PMNET_TRAIN_GEOMETRY_AUDIT_2026-10-15.md). All 52 focused checks pass; input hashes and paired-endpoint recomputation match. Only the latest progress PDF/PPTX and proposal PDF are present; Git tracks current PDFs only in these delivery directories.


## 2026-10-15 — visibility dev original-image gate closeout

Eight runs complete. Admission fails on frozen original-image clauses 2/3; C2 remains unexecuted. [Full tables, arithmetic and limits](results/VISIBILITY_SUPERVISION_DEV_2026-10-15.md). No new training, test access, threshold selection or scientific implementation changes.


 > **2026-10-14 — Visible-Fragment Supervision development started (execution record, not a performance result):** The user explicitly authorized four arms × two seeds.
 > The specification and recipes were frozen. Full regression passed (403 checks); an independent review's A3 validation-target issue was fixed. Real fold1 batch-16
 > freeze/unfreeze smoke passed. A0–A3 seed19 and A0 seed1 completed 130-epoch training and fold2 evaluation; A1 seed1 was training, with 2 runs queued (A2/A3 seed1). Source eee3248 and command/config/input fingerprints are bound.
 > No test-fold reads, no TTA, final checkpoint only; closed routes remain closed. Full status is recorded in
 > [research status](audits/RESEARCH_STATUS_2026-10-14.md). No new method performance or admission decision was available.

 > **2026-10-14 — E4 verified after correction; documentation closed:** The active artifact is
 > `runs/analysis/visibility_probe_20261014/`, bound to code `988c0cd`. The older 10-13 artifact includes train-mode
 > inference and an H1 result not stratified by area; retain it as invalidated history and do not cite it as the formal result. H1 area-standardized Dead change:
 > **+.1738 [.1060,.2528]** (the coarse estimate +.2913 is descriptive only); H2 DiD **+.1474 [.1392,.1557]**
 > (interpolation control −.0006); H3 **+.8175 [.7968,.8393]**. All three frozen gates were independently recomputed and passed. Code and
 > checkpoint/prediction/GT/eval SHA256 values and both sets of 2523 windows were verified identical. This result permits drafting the next
 > protocol only; it does not establish HV zeroing as the cause of missed detections and does not authorize training. The next-round
 > [visible-fragment supervision design draft](superpowers/specs/2026-10-14-visibility-supervision-design.md)
 > was prepared for review; the closed research line and unavailable E1 experts remain unchanged.

 > **2026-10-11 — E1 revised execution (E1-C + E1-M, spec §7):** With no expert available, the user approved replacing that step with
 > detector consensus + non-expert model prescreening. See the same-day entry below; the human-review obligation remains, and the materials package is unchanged.

 > **2026-10-09 audit:** See the [review and corrections](audits/RESEARCH_AUDIT_2026-10-09.md).
 > Rechecked 108 stage-III endpoint deltas and reproduced the frozen stage-IV verification snapshot.
 > Corrected E0's unequal-chunk mean. From saved throughput, image-weighted two-model/x1 =
 > **6.084468×**, TTA/x1 = **4.663162×** (historical chunk means 6.086869× / 4.666531×).
 > No new GPU measurement; approximately 6.1× and the no-deployment-win conclusion remain.
 > Actual forward precision was bf16 autocast, not the archived fp32 label.
 > Added missing gradient-probe/x2-cost fold guards, respected EF identity in cost replay,
 > and rejected unsupported HoVer-NeXt external TTA; **277 tests passed**.
 > Historical P0–P3 full-provenance revalidation is blocked by the old split2_seed2 seed1
 > inventory versus the later seed2 directory; the guard was not bypassed. Frozen results remain intact.

 > **2026-10-05 follow-up audit:** The typing-audit, matched-seed, and EF-P2 entries below
 > include corrected denominators, rounding, and interpretation. These are post-execution
 > documentation corrections, not changes to frozen menus or archived experimental results.

 > **2026-10-02 audit correction — supersedes conflicting historical interpretations below.**
 > `lever_postproc.py` used an off-by-one class lookup and included background edges in the
 > final instance's perimeter. Existing lever scores describe the historical implementation,
 > not a correctly implemented same-class merge. Pre-merge review also found lost
 > shared contacts across directions and lossy validation GT channels. The current P0 v2
 > selection/evaluation fixes both; v1 and older outputs remain archived. The alleged 3×3 seed grid
 > has nine run directories but split2 seeds `{19,1,1}` (launch script and configs agree),
 > and the base has seeds on split1 only (3/1/1 coverage). Do not use directory names as seed
 > provenance or quote the old noise ratios as fully paired evidence. Recomputed seed19
 > means: base strict Dead PQ .132989; du2 .121388; historical du2lev .123754, despite
 > official Dead PQ .176012/.183115/.183190. `mPQ+` remains class-aware; `F_c` is not pure
 > detection recall; `1 - (missed_bg + missed_shape)` is not matched recall because merge/split
 > are separate states. `missed_bg` means max IoU < .1, not necessarily zero overlap.
 > LKCell/HoVer-NeXt released val-selected checkpoints are not our own final checkpoints.
 > The bPQ “5.3× maximum seed std” claim overlooks baseline split1 std .001548 (versus
 > x2 maximum .001308); this and duplicate-seed coverage must be relabelled/recomputed.
 > Execution: `docs/superpowers/plans/2026-10-02-p0-p3-research.md`. Corrected results and
 > limitations: [P0–P3 report](results/P0_P3_RESULTS_2026-10-02.md).

## 2026-10-14 — E4 corrected full-results closeout: all three gates pass; prior inference and H1 definitions superseded

The active artifact is `runs/analysis/visibility_probe_20261014/`, code `988c0cd`.
See [complete results, denominators, and limits](results/E4_VISIBILITY_PROBE_2026-10-13.md).
H1 area-standardized +.1738 [.1060,.2528]; H2 +.1474 [.1392,.1557] (C-int −.0006);
H3 +.8175 [.7968,.8393]. Retain the old coarse H1 +.2913 as descriptive only; old train-mode inference artifacts are invalidated history.
Closeout rechecked input/code SHA, 192/160 windows, and frozen-gate arithmetic; no inference was repeated and no new training ran.
C2/C3 mPQ/bPQ .4633/.6183 and .4518/.5884; unified nucleus-level type accuracy .8477/.8315.
Original-image C0 .4835/.6531 remains bound; H3 miss rates for the three bins are .8677/.3774/.0502.
Passing the gates permits drafting the next round only: [visible-fragment supervision design draft](superpowers/specs/2026-10-14-visibility-supervision-design.md).
The new design was not approved. Limits remain: truncation position/context and residual zoom confounding, one checkpoint, repeated development-set use, and an unverified HV mechanism.
Do not reuse the old claims “DiD completely separates amplification” or “the old H1 is the gap after area control.”

## 2026-10-13 — M merged-class metric diagnostic: merging {Inf,Conn} ranks first among 10 class pairs (Δofficial +.0519); typing-oracle gap shrinks 35% in the merged space

Executed the [spec](superpowers/specs/2026-10-13-merged-class-m-design.md) (user-approved 2026-10-13; see
[plan](superpowers/plans/2026-10-13-merged-class-m.md)). Artifacts are in
`runs/analysis/merged_class_20261013/` (results.json / tables.md / commands.md; input SHA256 and
code fingerprint in provenance). GT and predictions were both remapped through the merge, and official aggregation is structurally equivalent (4 classes);
descriptive diagnostic only: no gate, no method claim, and no change to the 5-class score table.

**Correction record (2026-10-14, prompted by review):** All 69 bootstrap CIs in the first artifact were invalid because stratified-resampling
blocks were misaligned with stratum positions (tissue strata mixed when interleaved). Point estimates were unaffected; all conclusion-driving values remain unchanged.
`merged_class.bootstrap_delta` and the similarly affected `fast_retype.mpq_bootstrap_delta` were fixed
(TDD regression test `test_bootstrap_delta_stratum_blocks_aligned` ×2); this round was regenerated and verified
(66/66 CIs contain their point estimates). **Prior-round note:** typing_oracle 2026-10-09
and T (2026-10-10) CIs recorded through `mpq_bootstrap_delta` were also invalid. Both rounds' gate decisions used
point estimates (T2 point +.0002; corrected CI still spans zero), so no decision flips. Recompute CIs with the corrected code
if citing them.

- **All bindings pass:** M0 identity re-evaluation is bitwise identical in mPQ to frozen eval (split1/2/3 eval_test_*);
  per-class PQ / bPQ differ by ≤1e-12 (frozen values use pannuke_eval matrix-summation order; the fast path accumulates bincount
  rows and has an inherent 1–2 ulp per-class difference—also seen in typing_oracle 2026-10-09; mPQ happens to be bitwise identical);
  Dead per-class PQ is bitwise unchanged under merge(2,3); bPQ is independently bound from the **inst map**.
- **Δofficial for all 10 class-pair merges** (3-split mean, paired-image bootstrap with 1000 tissue-stratified resamples;
  (2,3) per-split point/CI: +0.0553 [.0492,.0621] / +0.0524 [.0460,.0581] / +0.0481
  [.0423,.0538], with every CI lower bound > +.04):
  **(2,3) ranks first at +.0519**; next are (1,3) +.0499 and (1,2) +.0190; the other 7 pairs are ≤ +.0099,
  including (4,5) +.0002. Strict aggregation agrees ((2,3) +.0647, all per-split CI lower bounds > +.055).
- **Merged-class PQ .5079 ≫ original two-class mean .4280** (Inf .4377 / Conn .4184).
- **oracle-under-merge(2,3):** current .5514 → oracle .6575 (gap .1061), versus the 5-class-space
  gap .1637 (typing_oracle round) ⇒ **merging shrinks the typing-oracle gap by 35.2%**; Inf/Conn-only
  intervention gap .1146 → .0497 (a 56.6% reduction).

**Interpretation limits (spec §8 preregistered):** The first-place result for (2,3) is directionally consistent with E1-T detector-consensus evidence,
but the reference distribution shows the top three pairs all merge two large classes (mechanical effect + confusion structure). Do not read +.0519 as “loss from ambiguous labels”;
the 35% oracle-gap reduction is a metric change in merged-class space (mPQ is nonlinear), **not an additive share decomposition**.

**Implementation note (reusable):** PanNuke `gt_channels` and `inst.npy` disagree on a small number of pixels (fold3
173/2722 images; e.g. img77 has 3 atypical pixels inside an Epi nucleus assigned to the Inf channel object, so channel geometry ≠ inst-map geometry).
Official bPQ matches on the inst map, while per-class PQ matches on channels—**bPQ cannot be derived from a channel-pair table** (the
preregistered binding caught this; `bpq_from_pairs` was removed from `merged_class.py`, and `merged_class_diag.py` now recomputes from the inst map).
No training, tuning, or selection occurred. Test artifacts were read-only diagnostics and are not method-score results.

## 2026-10-12 — E3 candidate-quality learning development round: all three arms failed admission; stop under preregistered rules; utility is predictable, strict-FP risk is not

Executed the [spec](superpowers/specs/2026-10-11-e3-candidate-quality-design.md) (user-approved 2026-10-11; A2 semantics
revised under spec §8 to a (floor,cap) grid). Artifacts: `runs/analysis/e3_quality_20261011/` (local; CPU,
8 workers). Training = 13,332 fold1 OOF candidates (three split2 seeds; matched labels IoU>0.5); application =
4,084 frozen fold2 candidates from split1_seed1; A1 = frozen EF-P2 (a30d: non-Dead floor 30, Dead exempt;
**A1 replay was hard-asserted within ≤1e-12 across 18 sweep rows and the p2ef frozen row**; the a30 variant once reproduced the EF row bit-exactly at
mPQ .48345570337477284, anchoring pipeline correctness). No fold3 reads.

- **Decision: primary/secondary gates both false (A2/A3/A4 all fail)** ⇒ stop under spec §5: no confirmation round and no extra menu to rescue the result;
  the utility-learning line closes at the development stage as a negative result. EF-P2 remains a mechanism-evidence/teacher candidate.
- **Failure-layer localization** (required closeout content):
  1. **Utility itself is cheap to predict:** fold1 GroupKFold CV AUC **.721** (12 low-dimensional features) / **.737**
     (+support/sup_frac); the low-dimensional arm has a good fold2 reliability curve (mean .45→rate .49, .65→.62).
     E1-C’s “support is monotonic” assumption holds for the learned model.
  2. **Selection rule conflicts with the endpoint:** the verbatim EF-round `select_fusion` rule (strict Dead ≥ identity .12957
     and interior Dead matched > 441 and mPQ/bPQ ≥ identity−.002) rejected **all 16
     non-identity rows** in A2/A3/A4 on the strict-Dead threshold (best learned row .1245 < .12957); all three arms therefore fell back to
     identity, so gate deltas equal identity−A1 (ΔDead −.0061, ΔbPQ −.0012, ΔmPQ −.0012; bootstrap CIs
     all below or touching zero). **A1 itself is among the few rows that pass the gate** (strictD .1345)—the frozen rule’s
     strict axis naturally favors adding fewer candidates.
  3. **Descriptive comparison (not part of the gate):** a4_p0.5 versus A1 = ΔmPQ +.0014, ΔbPQ +.0017, ΔDead +.0030,
     adding 2134 candidates (A1 1374) with unmatched share 46.8% (A1 53.3%); a3_p0.4 = ΔmPQ +.0010,
     ΔbPQ +.0012, ΔDead +.0014, unmatched 45.8%. Thus **official metrics show inexpensive
     bPQ/Dead gains, but they all cost strict Dead −.012 to −.014**—the Dead-FP penalty on images lacking that class
     is invisible to candidate-level features (area/shape/confidence/support). Candidate-level derivation P(match)·E[IoU|match] > q/2
     (10-06 §Route B) fails exactly where it ignores epsilon/missing-class penalties, matching its own warning.
  4. **The support-arm shift was measured:** +support calibration degraded (fold2 bins with .84 confidence had actual rate .58—
     a composition shift from 6 CellViT seats in fold1 to 13 seats across folds/architectures in fold2); the CV AUC gain (+.016) did not transfer
     to the endpoint. The HoVer-Net split2 checkpoint was lost during cleanup on both disks; the fold1 jury had no HN seat (spec deviation,
     disclosed; retraining one seat is disproportionate, and using split1/3 HN would break fold1 OOF).
- **Mapping to the 10-09 plan §5:** Close the “candidate-quality learning” branch supported by E1: a simple quality model
  (including jury support) cannot replace EF on the joint official+strict endpoint. Reopening would require
  **image-level missing-class risk features** (e.g. image Dead prior/tissue condition), not a more complex candidate-level head—this is
  a new testable hypothesis, but does not automatically authorize a project.
- Run log (three genuine failures were caught by guards/assertions, fixed, and rerun): (first) class domain is 1..5 (the eligible pool
  was already typed), causing one-hot shift; (second) distance values included inf sentinels (no base instance in image, 17+19 rows), clipped to the 256² diagonal
  constant 362 for images (data-independent, same transform for both folds); (third) first A1 used a30+stage-1 row—bit-exact reproduction of a30
  exposed that the current row was a30d; derive floors/exemptions from p2ef selection.json and assert them. Also, mix counts
  (records column 1376) and fuse runtime (1374) differ by 2 rows (probability-table/runtime-overlap difference); use
  fuse assertions for A1 endpoint, and treat mix as descriptive. Full pytest: 332 passed (including 17 new tests).
- **Post-review revision (evening 2026-10-12, after independent branch review):** Full pytest **335 passed**; final artifacts
  were regenerated (`e3_quality_20261011/`, `e1t_typing_20261011/`; old versions saved under `_v1*`) and were
  **bitwise identical** to the first version (all E1-T; all E3 except the bootstrap mPQ column). That column now follows official NaN exclusion
  (average over classes present in GT only): identity−A1 ΔmPQ CI changed from [−.00232,−.00002] to
  **[−.00242,−.00014]**; the conclusion (below or touching zero) is unchanged. Provenance now includes ef_selection,
  dependency-module fingerprints, and E1-T fold2 GT SHA; two regression tests were strengthened (A1-drift assertion failure path and conditional
  bootstrap denominator check—under old semantics lo=0.2≠1.0 must fail).

## 2026-10-12 — E1-T cross-detector typing consensus: the jury systematically favors the base class on base-misclassified instances; consensus evidence for an Inf/Conn label ambiguity

Executed the [spec](superpowers/specs/2026-10-11-e1t-typing-consensus-design.md) (user approved the direction on 2026-10-11;
“please continue” was treated as approval of the spec). Artifacts: `runs/analysis/e1t_typing_20261011/` (local; `*_v1_bad_ci/` is the first run,
invalidated due to a bootstrap-denominator bug; the four cell point estimates match across runs). Cache replay only, CPU-only, fold2 read-only
(split1 validation). Base = split1_seed1 validation base identity (x1); jury = E1-C 13 seats minus the base run = **12 seats**
(10 CellViT + 2 HoVer-Net, trained across folds {1,3}). Paired IoU > 0.5 (ties reported separately; 0 this round);
vote = member’s pixel-majority class for its maximum-IoU instance (IoU ≥ 0.3) against that GT (**pixel majority class**,
canonical `instance_classes` semantics; type-0=valid untyped; base data contain 164 type-0 instances, which caused the first run’s “unique value” requirement to abort);
majority class requires ≥3 total votes, a strict win, and votes from both training folds (defined independently of base/GT; fixes the first run’s
degenerate “consensus against GT” in G0). Group accounting: G1 6,858 + G0 40,558 + ambiguous 0 + unmatched 12,456 =
**59,872 = all fold2 GT instances** ✓. Manual checks on three images (image 0/1200/2500) matched grouping and vote-level outputs
(`scripts/e1t_verify_spot.py`, output in tasks cache; image 0 GT9 is a typical G1-maj=base example).

- **Main result:** In G1 (matched instances misclassified by base, n=6,858), jury majority class = base class **4,970 (72.5%, image-
  cluster bootstrap 95% CI [.706,.742)**; jury majority = GT class for only 1,246 (18.2% [.167,.197], CIs separate).
  In G0 (base classified correctly, n=40,558), jury’s intrinsic flip rate was only **1.9% [.017,.021]**. Per preregistered interpretation key:
  **evidence for an ambiguous Inf/Conn label boundary in detector consensus is supported**—cross-fold, cross-architecture detectors systematically favor the base’s
  “wrong” label, pointing to a systematically ambiguous label boundary rather than noise from one model (**not a biological claim**; correlation between jury and base family
  is disclosed. G0 partly constrains a “pure echo” explanation: if the jury only echoed base, G0 flips should be ≈0 and G1 ≈100%).
- Inf↔Conn is largest in both directions: (2,3) (GT=Inf, base=Conn) 74.8% [.718,.779]; (3,2) 78.5% [.756,.811];
  Neo (1,2)/(1,3) are also aligned (69.8%/66.4%). **The only large-cell exception is (5,1) (GT=Epi, base=Neo): GT leads
  (51.6% vs 30.1%)**; report this separately, outside the aggregate conclusion.
- **Dead separately** (small sample and heterogeneous cells; descriptive only): combined Dead-related cells, maj=base [.573,.703] vs
  maj=GT [.173,.282], same direction as Inf/Conn but smaller; (3,4) 81.4% versus (4,2) 48.9% is heterogeneous.
- **All three sensitivity analyses agree in direction:** vote threshold IoU 0.5 (base 72.0%); min_votes 5 (72.3%); HoVer-Net-
  only 2-seat jury (base 36.0% vs GT 20.2%, no-majority 36.2%, G0 flip rate 6.7%).
- **Conclusion mapping (spec §2 preregistered key):** Close the typing direction as “**ambiguous Inf/Conn label boundary (detector-consensus evidence)**”—
  the bPQ−mPQ typing gap (typing oracle .4995→.6632) contains a **component that current label supervision cannot remove**;
  this is compatible with T0–T3 (no incremental post-hoc readout from internal features) and the negative CONCH result: the readout ceiling is limited by label noise.
  The joint-training hypothesis remains deferred; this result does not start training. Together with E1-C (candidate-presence consensus), the typing-axis
  label-quality evidence chain is complete. The final report must state “not a biological claim.”
- Run log (all RED→GREEN fixes were rerun): type unique-value requirement → majority-vote semantics; two conditional-bootstrap
  bugs (rows outside a group counted as 0 rather than NaN; evaluate event before member in a three-way condition). The second was exposed by a miniature fixture
  (G0 CI reached 0.5). No test-fold reads and no new inference/training.



## 2026-10-11 — E1-C/E1-M revised execution without experts: candidate cross-detector consensus far exceeds a random baseline; model prescreening is unusable

Executed the [revised spec §7](superpowers/specs/2026-10-10-e1-blind-audit-design.md) (user-approved 2026-10-10).
Artifacts: `runs/analysis/e1_consensus_20261010/` (local). Jury: 13 seats: split1/split3 ×
seed{19,1,2} × {x1,x2} fold2 identity predictions (11 seats; source run split1_seed1-x2 excluded) +
HoVer-Net split1/split3 fold2 (2 new inference seats, val-fold sanity mPQ .4509). All members trained on folds
∈{1,3}; fold2 was never used for training. Support = number of members with IoU≥0.3; candidate set = frozen 4,084 from split1_seed1.

- **Synthetic null control (added before writing any conclusion):** Place same-size random blobs more than 8 px from GT instances
  (area resampled from the “zero-overlap candidate” distribution, n=2000) → mean support **0.07**, ≥3 rate **0.7%**,
  ≤1 rate 99.1%. This control tests only a “random-location baseline”; it does not control for shared nucleus-like false positives across detector families.
- **Observation** (S2 accepted n=1515 / S3 rejected n=2569): mean support 5.47 / 5.04; GT match rate rises
  monotonically with support (support 0–2: 5–12%; ≥5: 40–60%). Stratified by relation to GT:
  - Zero overlap with GT (S2 518 / S3 1146): mean 4.8/4.4; **≥3 and cross-fold/cross-architecture support: 380/751** (expected
    ~0.7% under the null)—nucleus-like objects detected consistently by multiple detectors where GT has no object, evidence of **questionable-label detector
    consensus**; not a biological claim (possible explanations include annotation-unit mismatch, omitted degenerate/apoptotic objects, or family-shared FP).
  - Overlap exists but IoU<0.5 (S2 328 / S3 582): support is similarly high (mean ~4.6)—a contour/instance-unit mismatch
    subgroup, consistent with the second row of §4 (geometric mismatch).
  - Debris-compatible (support≤1): S2 151 (10.0%) / S3 398 (15.5%)—EF filtering points in the same direction but has limited discrimination.
- **Engineering implication (mapping from revised §4):** Support strongly predicts candidate quality (monotonic, consistent across both sources) →
  “low-dimensional features + jury-support labels” now has controlled evidence supporting candidate-quality learning and merits prototyping;
  retain prerequisites: training-only cross-fit, comparison with frozen EF-P2/pure area filtering, and no test-fold reads.
- **E1-M (two independent model blind reviews; annotators were non-experts):** passA (analyze_image channel), on S1—known to be
  almost entirely true nuclei in x1 detections—called only 40.9% true (systematic “not a nucleus” bias, with highly templated reasons);
  passB (sonnet native vision) had 34–70% uncertainty but a more reasonable ranking (S1 .915 > S2 .506/S3 .573 >
  S4 .408). Kappa across rounds −.09–.35; raw agreement .18–.38. **Conclusion: model reviews do not even agree with one another; their signal is unusable**—
  E1-M served only to validate its workflow and retrospectively supports excluding model review from formal conclusions in the spec’s decision.

- **E1 status:** Human-review obligation remains (the package is unchanged and the original protocol can be run at any time); the engineering branch is supported by E1-C
  alone. Limits: one pair (split1_seed1), CellViT family occupies 11/13 seats, support threshold 0.3 preregistered,
  synthetic-disk null control (not nucleus-like shapes). No test-fold reads; HoVer-Net fold2 is validation-fold inference.

## 2026-10-10 (later) — T/E1 pre-registered specs approved; E1 packet built; typing-probe T0–T3 implemented

- User approved both specs (2026-10-10): T ([spec](superpowers/specs/2026-10-10-typing-probe-t-design.md),
  [plan](superpowers/plans/2026-10-10-typing-probe-t.md)) and E1
  ([spec](superpowers/specs/2026-10-10-e1-blind-audit-design.md),
  [plan](superpowers/plans/2026-10-10-e1-packet.md)). All values frozen before any run.
- **E1 packet complete** (`runs/analysis/e1_packet_20261010/`, local): 200 windows
  (S1 x1 objects / S2 EF-accepted / S3 EF-rejected / S4 random background, 50 each),
  tissue×border stratified with recorded inclusion probabilities, seed 20261010, built from the
  split1_seed1 frozen selection via the same replay code path as the morning's readiness check
  (column binding passed against `e1_input_check_20261010_split1_seed1/records.npz`).
  Stage-1 material is image + cross only (VLM-render check + programmatic cross-position check
  passed on sampled windows; one sampled S2 window is a 63 px border sliver, pmax .95, unmatched —
  exactly the population E1 is meant to characterize). Analysis script `scripts/e1_analyze.py`
  is committed BEFORE any reviewer answers exist. **E1 awaits two qualified reviewers (user to
  appoint); until then E1 is incomplete by definition.**
- **Typing-probe T0–T3 implementation complete on branch `research/typing-probe-e1`**
  (committed, tests in `tests/test_typing_probe.py`): `RetypeEvaluator.mpq_strict` +
  tissue-stratified paired-image bootstrap (both mirrored bit-exactly against canonical
  `pannuke_eval` on synthetic fixtures), `nucseg.typing_probe` (instance/24px-ring pooling,
  T1 rule, torch linear head seeds {0,1} + sklearn cross-check), fold-guarded feature export
  (`scripts/export_tp_features.py`, inst_prob binding check) and the pre-registered runner
  (`scripts/run_typing_probe.py`, decision gate = spec §2 six conditions).
  Preflight on the real val predictions: T0 table alignment holds (59,593 rows, 132 untyped);
  T1 changes only 0.38% of instances vs T0 — the aggregation rule itself is near-saturated,
  so any material gain must come from T2/T3 features.
- Incidents during the round (all fixed RED→GREEN where behavioral): NpzFile per-image indexing
  re-decompressed the full member array (repeat of the 2026-10-08 E0 pitfall; 9 h lost on the
  first packet build), zero-instance images wrote 64-wide rows into the 6-wide prob column,
  tp softmax missed the NCHW→NHWC permute, and a stale `__pycache__` from a `git stash` round-trip
  briefly executed old bytecode at identical file size. Full local suite after the fixes: passed
  (typing-probe + e1-audit files; full `pytest -q` re-run closing the round).
- No test fold read in this round; T results are dev-fold (split1/fold2) only and pending below.

## 2026-10-10 (T verdict) — typing probe T0–T3: pre-registered gate FAILED; simple readouts carry no incremental typing signal

Ran per [spec](superpowers/specs/2026-10-10-typing-probe-t-design.md) / [plan](superpowers/plans/2026-10-10-typing-probe-t.md)
(artifacts `runs/analysis/typing_probe_20261010/`, local). Split1 dev-only: fold1 trains the
heads (63,218 GT instances), fold2 validation (59,593 predicted instances), fold3 untouched.
Feature export verified against the cached instance-probability table (max |Δ| = 6e-8).

| config | mPQ | Δ vs T0 | notes |
|---|---:|---:|---|
| T0 majority vote (identity) | .483486 | — | reproduces official eval_fold2 exactly |
| T1 instance-mean probability | .483182 | −.0003 | boot CI [−.0010, +.0003]; aggregation rule saturated |
| T2 pooled tp_feat + linear head | .483707 | **+.0002** | 2 seeds same direction; CI [−.0027, +.0019] |
| T3 T2 + 24px context ring (sum) | .481796 | −.0017 | both seeds negative |

- **Gate verdict: pass_gate = false** (c1 Δ≥.003 ✗ at .0002; c2 CI-lo>0 ✗ at −.0027;
  c4 Dead ≥−.002 ✗ at **−.0053** — the unweighted linear head types Dead worse than the
  majority vote; only c3 strict +.0020, c5 beats-T1, c6 seed-consistency pass). sklearn
  deterministic cross-check agrees in direction (val mPQ .4817, worse).
- **Failure position (per spec stop-loss):** not the aggregation (T1 null); not unreadable
  features (GT-region diagnostic accuracy **85.3%** ≈ the T2 head's own matched typing accuracy
  **85.8%**, current decoder type accuracy 85.6%);
  the linear readout extracts the SAME information the TP head already uses, with no increment
  on predicted contours — and it under-serves the rare class. Per the pre-registered rule: no
  bigger classifier, no added configs; a new testable explanation is required first (candidates:
  Inf/Conn label ambiguity, IoU-marginal matches, joint training rather than post-hoc readout).
- Combined with the CONCH re-typing null: two independent feature sources (external VLM,
  internal frozen TP features) both fail post-hoc re-typing — the bPQ−mPQ typing gap is not
  recoverable by post-hoc instance-level readouts of frozen features.
- Verification: fast `RetypeEvaluator` == canonical `pannuke_eval` on the retyped T2 map
  (|Δ| = 1e-16); **bPQ bit-identical** to eval_fold2 (geometry untouched). Decision rule values
  and all inputs SHA-bound in results.json.
- Cost (reported, not gated): export pooling 790 ms/img unoptimized single-thread (51× the
  ≤3 ms/img design target; a deployment path would need vectorized pooling if ever revived);
  head forward 8.2 ms for the whole fold. GT-contour-trained heads applied to predicted
  contours (spec'd domain gap) — recorded, and the GT-region diagnostic column separates it.
  sklearn cross-check hit lbfgs max_iter=1000 (ConvergenceWarning; direction unchanged).
- Incidents fixed en route (all RED→GREEN pinned): `_gt_classes` +1 label shift (100% of cached
  label rows repaired deterministically, features byte-identical), retyped type map built with
  a global id LUT that cross-wired images (ids restart per image; fixed per-image, canonical
  check re-run), NpzFile re-decompression hang, zero-instance column widths, tp softmax
  NCHW→NHWC, stale `__pycache__` after a `git stash` round-trip.
- Dev-fold conclusion only: single split, single segmenter seed (19), head seeds cover the
  linear head alone; no test fold was read; this entry makes no performance claim beyond the
  validation fold.

## 2026-10-10 — verification and validation-candidate readiness

- Fresh full suite: **277 passed, 64 warnings**; Python 3.10.21, torch 2.5.1+cu124,
  numpy 1.23.5 unchanged. Stage-III's 108 endpoint deltas and stage-IV's frozen snapshot
  (96 E2a endpoint cells) reproduced; the image-weighted cost correction also reproduced.
  Both revised Chinese PDF/TeX hashes match the revision ledger; all seven formal-report
  table numeric sequences, figure references, labels and bibliography keys are unchanged.
  This checks the existing PDFs, not a new compilation or independent all-page visual review.
- Continued with a **CPU-only replay of frozen P2 validation candidates**, not a new
  selection or performance experiment: split1, seed1, fold2, frozen
  `a200_p0.9_interior0`. Checked input/GT SHA-256, byte sizes, and stored run-context
  hashes before executing `audit_existence_candidates.py --pairs split1_seed1 --workers 2`.
  Its old CLI does not itself enforce all those provenance checks. Checkpoint binding
  remains legacy run provenance, not a new hash-bound inference.
- Recomputed **4,084 eligible / 1,515 accepted** candidates; among accepted candidates,
  **669 match GT at IoU > .5, 589 are also correctly typed, and 62 are correctly typed
  Dead**. Every saved candidate column exactly matches the same pair's subset in the
  frozen `existence_audit_20261005_v2/records.npz`. This is P2 acceptance, not EF-P2.
  Output and verification: `runs/analysis/e1_input_check_20261010_split1_seed1/` (local).
- **E1 remains incomplete:** replayed GT-match labels are not independent biological
  judgments. The blinded two-stage packet, two qualified reviewers and sampling protocol
  are still needed. The seed19 identity selection cannot supply an accepted-candidate group.
- Next proposed bounded control is T0/T1 on cached split1 validation instances, comparing
  majority vote with mean TP probabilities while preserving geometry and all predictions.
  It needs a guarded runner and a fixed background rule; the existing CONCH retyping CLI
  also trains and reads test data, so was not used. The experiment spec remains pending;
  no typing probe, new training or inference was launched. This readiness replay read
  validation only; the separate report verification also read historical test summaries.

## 2026-10-09 — typing-oracle ceiling diagnostic (read-only, unregistered)

Conversation-driven analysis ([digest](background/STRUCTURE_AND_STRATEGY_2026-10-09.md) §5): how far official
mPQ could rise if every IoU>0.5 matched instance were typed correctly, instance geometry untouched.
Scored with `RetypeEvaluator` on the frozen baseline (CellViT-UNI, no TTA, last checkpoint) test
predictions — audit-type access to already-evaluated test folds, no tuning, no new inference;
current values reproduce the official summaries (.4951/.4930/.5104).

- **Typing oracle**: 3-split mean mPQ .4995 → **.6632 (+.164)**; per-split gains are
  +.1692/+.1671/+.1548. This is the gain from one GT-assisted retyping intervention with
  geometry fixed. Its numerical proximity to bPQ .6653 is not an additive decomposition
  of bPQ−mPQ: the metrics use different matching/aggregation, and segmentation errors remain.
- **Inf/Conn-GT-restricted oracle** (all matches whose GT is Inflammatory or Connective
  corrected, regardless of their predicted class): → **.6141 (+.115)**, approximately 70%
  of the all-class intervention's gain. **Correction on 2026-10-09:** this does not isolate
  Infl↔Conn swaps; `typing_oracle.py` selects on GT class only. The gain can also change
  other classes' FP counts. Separate descriptive confusion rates were 11.8%/10.9% in the
  two directions, with overall matched typing accuracy .864; those rates do not establish
  a 70% pairwise contribution.
- Per-class (split1, current→oracle): Neo .570→.666, Inf .440→.720, Conn .418→.595,
  Dead .142→.411, Epi .575→.659.
- Interpretation limits: oracle ≠ achievable performance. Label ambiguity and residual
  feature separability need independent tests; no universal literature ceiling is established
  here. The tested CONCH retyping methods were null, which argues against simply repeating
  those recipes but does not establish feature saturation. The revised proposal in
  [RESEARCH_NEXT_2026-10-09.md](proposals/RESEARCH_NEXT_2026-10-09.md) separates candidate-existence
  review from an instance-typing feasibility probe. All proposals remain unapproved;
  selection must stay on training/development data, and a reused benchmark is not a fresh test.
- Reproduce: `python scripts/typing_oracle.py --out-json runs/analysis/typing_oracle_20261009/summary.json`
  (artifact: `runs/analysis/typing_oracle_20261009/summary.json`, local). Relevant pytest subset
  (`-k "retype or typing"`): 10 passed.

## 2026-10-02 — P0–P3 v2: pre-merge defects fixed and dependent CPU experiments regenerated

Current source: `runs/analysis/p0p3_20261002_v2/results_verified.json` and `tables_verified.md`.
The same frozen validation menus were rerun, not expanded. Original model predictions and all
41 P3 files are unchanged. No training or new GPU inference; v2 runs on CPU.

- **Repair:** accumulate horizontal and vertical contacts instead of overwriting; sweep directly
  against preserved official GT class channels, not lossy merged-label reconstruction. Nine
  new geometry/GT regression cases first failed, then passed. P0 also binds test to validation
  GT hashes; eight bootstrap guard cases bind split, comparison paths and point estimates.
- **P0:** base/MS **.499834 mPQ / .665622 bPQ**, x2/MS **.490576/.660088**.
  Base validation frac/area = (.5,40)/(.25,40)/(.5,40); x2 = (.5,40)/(.25,20)/(.25,40).
  Base/MS Dead PQ / strict Dead PQ = .176688/.134047; x2/MS = .184987/.124931.
  Mean mPQ/bPQ gains are small, not universal per-split or per-metric improvements; mPQ+ decreases.
- **P1:** validation typed TP = 983/1262/1034, typed FP = 2321/2808/2120. Test interior Dead
  match rate .532667→.599961; stable x1 types on x2 geometry give mPQ .497402 versus .490576.
  The reverse gives .492525. Oracle values remain GT-assisted diagnostics, not deployment scores.
- **P2:** validation again selects identity / a200_p0.9_interior0 / a200_p0.7_interior0.
  Test means **.499906 mPQ / .665175 bPQ / .183534 Dead PQ / .137050 strict Dead PQ**.
  Deltas versus base/MS: +.000072 / −.000448 / +.006846 / +.003003; strict mPQ −.000411.
  2000-replicate paired image bootstrap: split2 Dead CI [−.0016,+.0116], split3
  [+ .0076,+.0256]. Only split3 excludes zero; strict Dead has no CI here, and no multi-seed
  or independent-split pooled claim is justified. P2 remains a candidate, not an overall win.
- **P3:** unchanged 128-image validation probe, not a full test benchmark.
- **Verification:** 126 CPU tests passed, 3 CUDA-only skips, 62 dependency warnings; four related
  GPU tests passed in the earlier execution. Independent pixel-neighbour references checked
  500 maps and 2500 transpose transforms. All six selected P0 validation arms and twelve P1
  input-arm comparisons agree with canonical mPQ/bPQ within 1e-12. Commands and logs are versioned.
  Updated report/PDF front matter supersedes both historical bodies and v1's completion assurance.

## 2026-10-02 — P0–P3 v1 execution (superseded by the pre-merge v2 correction)

**Historical record, not the final corrected method:** the subsequent pre-merge review found
that shared-boundary lengths were overwritten between axes and P0 sweep GT channels were
reconstructed lossily. The values and completion assurance below describe the first execution;
they are retained rather than silently relabelled as v2. Current results belong to the v2 entry
and the linked P0–P3 report.

All numbers below are from `runs/analysis/p0p3_20261002/results_verified.json`, generated by
`scripts/p0p3_numbers.py`; the source prediction hashes and frozen selection→test links are checked.
No training or A100 GPU jobs were launched. This stage used CPU sweeps/evaluation and bounded L4 inference.

- **P0 (seed19, three-split test means):** base .499510/.665345 (mPQ/bPQ); corrected base+M/S
  **.500717/.666834**; du2 .489381/.658274; corrected du2+M/S **.491539/.661488**.
  Base M/S chooses (.25,40) on every validation fold; x2 chooses (.25,40)/(.25,20)/(.25,40).
  Thus M/S is not an x2-specific gain. Base/MS Dead PQ / strict Dead PQ .176721/.134068;
  x2/MS .183009/.123632. Old M/S scores remain separate historical measurements.
- **P1:** using current corrected x2 masks, validation disjoint candidate pools contain
  983/1259/1032 correctly typed matches but 2318/2848/2152 typed FPs. GT-assisted addition
  gives validation ΔmPQ +.009194/+.010702/+.010492, a diagnostic only. On test, interior
  Dead IoU>.5 match rate .532667→.603543. Stable one-to-one type exchange gives x2 geometry
  + x1 types mPQ .498380 (vs x2 .491539); the reverse gives .492490 (vs base .499510).
- **P2:** stronger reference = corrected base/MS; candidates = original du2 with valid
  per-ID type probabilities. Frozen validation choices: split1 identity (NO_GO), split2
  area≤200/p≥.9/allow-border, split3 area≤200/p≥.7/allow-border. Test means
  **mPQ .500793 / bPQ .666382 / Dead PQ .183571 / strict Dead PQ .137074**.
  Relative to base/MS: +.000075 / −.000452 / +.006851 / +.003006 respectively;
  strict mPQ −.000406. This is a candidate, not an established overall win.
  Paired 2000-replicate image bootstrap: Dead Δ split2 CI [−.0016,+.0116], split3
  [+ .0076,+.0257]; split1 is identical to reference. No seed-uncertainty or independent-split
  pooled significance claim; strict Dead is a point estimate without a CI here.
- **P3:** fixed tissue-stratified 128-image split1/fold2 validation subset, 3693 nuclei,
  84 Dead in 11 images. Train/infer scales 1/1,1/2,2/1,2/2 mPQ =
  **.486872/.306309/.367467/.487009**, bPQ = .664229/.447545/.554316/.654835.
  Strong scale-adaptation effect, but not a full-fold or three-split benchmark.
- **Verification:** CPU suite 109 passed / 3 CUDA-only skips; related GPU tests 4 passed separately. Independent review approved scientific claims and found one result-link guard gap;
  it was fixed with red→green tests, then all selections/parameters/folds were revalidated
  without changing result values. Stage reports carry dated front-matter corrections.


## 2026-09-24 — CONCH text space at nucleus level (pillar A feasibility)
Probe: fold 1 (a TRAINING fold), class-balanced random GT nuclei, CONCH v1, prompt bank
`configs/text/nuclei_prompts.yaml` (4 templates; 2 names / 8 LLM descriptions per class), per-class
mean-logit calibration (transductive; subtract each class's mean logit over the probe set).
Chance = 0.20. Scripts: `scripts/encode_text_prototypes.py` (crops), `scripts/probe_conch_dense.py`.

Nucleus-centred crops (CONCH preprocess: bicubic resize to 448), one CONCH pass per nucleus,
400 nuclei/class, balanced zero-shot accuracy:

| crop | names | descriptions | desc, tissue-conditioned | linear probe (5-fold CV) |
|---|---|---|---|---|
| 32 px  | 0.27 | 0.23 | 0.21 | 0.57 |
| 64 px  | 0.31 | 0.38 | 0.29 | 0.60 |
| 128 px | 0.39 | 0.48 | 0.37 | 0.69 |
| 256 px | 0.54 | 0.65 | 0.54 | 0.73 |

Radius-restricted attentional pooling (ONE pass per 256 px patch, bilinear resize to 448 = 28x28
tokens; CONCH's contrastive attentional pooler with key-mask = tokens whose centre lies within R px
(original 40x pixels) of the nucleus centroid, nearest token always kept), 800 nuclei/class:

| R (px) | names | descriptions | desc per-class recall (Neo/Inf/Con/Dead/Epi) |
|---|---|---|---|
| 16   | 0.603 | 0.685 | .60 .66 .52 .87 .77 |
| 32   | 0.608 | **0.695** | .65 .63 .55 .88 .76 |
| 64   | 0.604 | 0.689 | .65 .62 .54 .88 .76 |
| 128  | 0.577 | 0.673 | .65 .60 .52 .84 .74 |
| all  | 0.548 | 0.651 | .63 .59 .50 .82 .71 |

Takeaways
- CONCH class semantics live at the context scale: tight crops are near chance; tokens contextualised
  by the whole patch + local pooling are best (0.695) and need ~24x fewer CONCH passes than
  per-nucleus crops (one per patch vs. one per nucleus; ~24 nuclei/patch).
- LLM descriptions beat class names by 8-11 points everywhere; tissue-conditioning HURTS zero-shot.
- Dead is the most text-recognisable class (recall 0.88) although it is the worst class for
  segmentation models (PQ ~0.15) -> pillar A redesigned as a dense CONCH prior fused with / adapted
  into the per-instance type decision (see docs/RESEARCH_PLAN.md §2.1 update).
- Caveat: balanced sampling; real Dead prevalence is ~1.5%, so precision must be checked in-pipeline.

Natural class distribution (ALL GT nuclei of each fold, `nucseg.text.conch_prior.ConchNucleusPrior`,
R = 32, bf16 trunk; cached in `runs/conch_prior/gt_fold{k}_r32.npz`), calibrated zero-shot:

| fold | nuclei | balanced acc | Dead precision / recall |
|---|---|---|---|
| 1 | 63,218 | 0.688 | 0.104 / 0.950 |
| 2 | 59,872 | 0.654 | 0.088 / 0.864 |
| 3 | 66,654 | 0.664 | 0.095 / 0.863 |

Zero-shot alone is far too imprecise on Dead (~10% precision); its use must be as a complementary signal.

### Supervised heads on the cached region-pooled CONCH embeddings (GT nuclei; train fold 1 -> test fold 3)
Full-batch logistic regression, class-balanced CE. `text-init` = W initialised at 20 x desc prototypes;
`anchor` = L2 pull of W towards that init.

| train nuclei | random-init | text-init | text-init + anchor 1e-3 | names-init + anchor 1e-3 |
|---|---|---|---|---|
| 125 (0.2%)   | bal .626 / Dead F1 .382 | .638 / .421 | .688 / .450 | **.704 / .485** |
| 630 (1%)     | .717 / .530 | .719 / .538 | .715 / .480 | .726 / .477 |
| 6.3k (10%)   | .767 / .550 | .769 / .561 | .730 / .470 | .737 / .454 |
| 63k (100%)   | .782 / .568 | .783 / .568 | .734 / .464 | .738 / .444 |

- Frozen CONCH region embeddings + linear head: Dead F1 ~0.57 on ALL GT instances of fold 3.
  Rough reference only (different population, leaky model): the official HoVer-Net checkpoint's
  Dead F1 among its IoU>0.5-matched nuclei on fold 3 is 0.40 (recall 313/519, precision 313/1060).
  The fair test is in-pipeline re-typing of predicted instances (below).
- The text prior only helps in the few-shot regime (~25 nuclei/class); with full PanNuke labels,
  init is irrelevant and anchoring hurts. => On PanNuke the gain is the *representation*
  (region-pooled VLM features); language matters for few/zero-shot (cross-dataset, new classes).

## 2026-09-25 — In-pipeline re-typing on CellViT-UNI (split 1: train 1 / val 2 / test 3)
Segmenter: our CellViT-UNI reproduction (`scripts/train_cellvit.py`, CellViT paper recipe), last
checkpoint, no TTA: test mPQ 0.4951 / bPQ 0.6676 (TTA 0.5013 / 0.6728); CellViT++ reports 0.492 for
CellViT-UNI. `scripts/retype_conch.py`, all knobs tuned on val fold 2 with the exact fast mPQ
(`nucseg.metrics.fast_retype`, unit-tested against the full evaluator):

| method | val mPQ | test mPQ | test Dead PQ |
|---|---|---|---|
| seg_vote (pipeline) | 0.4835 | 0.4951 | 0.142 |
| seg + class bias (recalibration control) | 0.4873 | 0.4953 | 0.135 |
| seg x zero-shot CONCH (alpha .1, tau 10) | 0.4835 | 0.4938 | 0.141 |
| seg x CONCH linear (alpha .3) | 0.4836 | 0.4920 | 0.141 |
| CONCH linear alone | 0.3645 | 0.3821 | 0.145 |
| seg x CONCH linear + bias | 0.4864 | 0.4943 | 0.134 |

With 8x TTA (same protocol): seg_vote test 0.5013; val tuning picks alpha = 0 for both CONCH
fusions (i.e. CONCH gets no weight); seg + bias 0.5024 (Dead PQ 0.146 -> 0.134); CONCH linear alone 0.3855.

=> **Negative**: post-hoc CONCH re-typing does not improve a UNI-based segmenter (its type branch
already carries the pathology-FM information). Why: Dead PQ is lost in *detection*, not typing.

Error decomposition of CellViT-UNI on fold 3 (IoU 0.5; fraction of GT nuclei per class):

| GT class | matched | merged | missed (no overlap) | missed (poor shape) | split | correct type when matched |
|---|---|---|---|---|---|---|
| Neoplastic   | .811 | .048 | .092 | .040 | .009 | .905 |
| Inflammatory | .849 | .046 | .079 | .023 | .004 | .817 |
| Connective   | .752 | .038 | .153 | .044 | .013 | .794 |
| Dead         | **.470** | .089 | **.385** | .048 | .008 | .690 |
| Epithelial   | .837 | .042 | .075 | .036 | .011 | .930 |

Dead GT median area 107 px (others 368). Inside the *missed-without-overlap* GT nuclei:

| class | n | mean area | mean NP prob | NP fires (>50% px > .5) | mean TP P(Dead) |
|---|---|---|---|---|---|
| Neoplastic   | 2615 | 210 | .205 | .18 | .005 |
| Inflammatory |  853 |  73 | .437 | .46 | .015 |
| Connective   | 2676 | 162 | .258 | .25 | .008 |
| Dead         |  407 |  97 | .164 | .17 | **.263** |
| Epithelial   |  664 | 171 | .329 | .32 | .000 |

- Missed Dead nuclei are invisible to the NP (foreground) branch, yet the TP branch assigns them
  P(Dead) ~ .26 (vs ~.01 elsewhere): the type head "sees" dead cells that the foreground head drops.
- ~Half of the missed Inflammatory nuclei have NP firing -> lost in post-processing (small objects).
- Implication: improvements for Dead must act on detection / foreground (class-aware foreground,
  small-object recovery, targeted synthesis), not on re-typing detected instances.

## Baselines (official 3-split protocol, last checkpoint; filled in as runs finish)
Test fold per split: 1 -> 3, 2 -> 3, 3 -> 1. Values: mPQ / bPQ / Dead PQ (no TTA; TTA in brackets).

| model | split 1 | split 2 | split 3 | mean |
|---|---|---|---|---|
| HoVer-Net (ours, full-patch) | .4558 / .6601 / .103 [.4664 / .6707 / .087] | .4529 / .6610 / .108 [.4633 / .6696 / .105] | .4606 / .6627 / .170 [.4700 / .6739 / .191] | .4564 / .6613 / .127 [.4666 / .6714 / .128] |
| CellViT-UNI (ours) | .4951 / .6676 / .142 [.5013 / .6728 / .146] | .4930 / .6638 / .162 [.4973 / .6683 / .163] | .5104 / .6647 / .224 [.5154 / .6707 / .221] | .4995 / .6653 / .176 [.5047 / .6706 / .177] |

Both baselines reproduce (HoVer-Net paper .463/.660, ours .456/.661; CellViT++ UNI .492, ours .500).
Course requirement (a) — reproducible HoVer-Net baseline with full official evaluation — DONE.

Reference (paper-reported): HoVer-Net .463 / .660; CellViT-UNI (CellViT++) .492.

## 2026-09-25 — Training-free recovery of missed nuclei (CellViT-UNI, 3 splits)
`scripts/dump_cellvit_maps.py` (fp16 NP/HV/TP maps) + `scripts/sweep_recovery.py`
(`nucseg.postproc.recovery`; default config tested identical to the official post-processing; light
metric tested equal to the full evaluator). 96 configs: beta (NP mixed with TP-branch foreground)
{0,.5,1} x k_dead (fg = max(fg, k P(Dead))) {0,1,2,4} x blob threshold {.3,.4,.5,.6} x orphan-blob
recovery. Selection per family by VALIDATION mPQ, one test scoring. Sanity: official config from the
fp16 maps reproduces the pipeline test mPQ within 0.0003 on every split.

Test mPQ (val-selected config per family):

| family | split 1 (test f3) | split 2 (test f3) | split 3 (test f1) | mean | mean Dead PQ |
|---|---|---|---|---|---|
| official | .4950 | .4927 | .5104 | .4994 | .176 |
| thr (tuning control) | .4953 (t.3) | .4929 (t.3) | .5108 (t.3) | .4996 | .176 |
| orphans | .4951 | .4927 | .5094 | .4991 | .177 |
| beta | .4953 (b.5) | .4929 (b0,t.3) | .5108 (b0,t.3) | .4996 | .177 |
| dead | .4953 (k0,t.3) | .4929 (k0,t.3) | .5108 (k0,t.3) | .4996 | .176 |
| all | .4953 (b.5) | .4929 (t.3,orph) | .5092 (t.3,orph) | .4991 | .178 |

Validation grid, mean over splits (beta 0, no orphans): mPQ is flat (.4902-.4906) across k and thr;
Dead PQ moves only .181-.184 even at k = 4. Orphans: val Dead +.004, Inflammatory +.003, bPQ -.0015;
no test gain.

=> **Negative**: missed Dead nuclei cannot be recovered post hoc from this model's outputs; every
change is within -0.0012/+0.0004 test mPQ (the threshold-only control does as well as any recovery
variant; the -0.0012 is the split-3 "all" family, visible in the table above). The detection deficit has to be addressed in training (foreground supervision / data).

## 2026-09-25 — Why Dead nuclei are missed (CellViT-UNI, 3 test folds pooled)
`scripts/show_nuclei.py` montage: `runs/analysis/dead_split1.png` (split 1, test fold 3; green = GT
nucleus, yellow = predictions). Qualitative (visual, not quantified): missed Dead GT are a mix of
(a) tiny dark nuclear fragments (karyorrhectic debris, often < 50 px), (b) round eosinophilic
apoptotic bodies without dark chromatin that do not look like nuclei, (c) a few ambiguous annotations;
matched Dead look like compact dark pyknotic nuclei.

Detection vs. GT area (fraction of GT nuclei with no overlapping prediction):

| GT area | Dead missed | Dead n | other classes missed |
|---|---|---|---|
| < 30 px   | .869 | 268  | .831 |
| 30-60     | .680 | 488  | .601 |
| 60-100    | .318 | 739  | .328 |
| 100-200   | .212 | 1073 | .125 |
| > 200     | .214 | 513  | .035 |

24% of Dead GT are < 60 px (other classes 6%) -> a size effect shared by all classes (IoU 0.5 is
very strict for such objects), plus an appearance effect: large Dead nuclei are missed 6x more often
than large nuclei of other classes. Training-side ablations on split 1 (running):
M1 NP pixel-CE with Dead pixels x11 (appearance), M2 same with nuclei < 100 px x6 (size),
C1 sampler gamma 1.0 (resampling control).

## 2026-09-25 — Training-side foreground supervision, split 1 (single seed; seed test running)
CellViT-UNI recipe + pixel-weighted NP cross-entropy (`--np-wce 1`), last checkpoint. Deltas vs. the
baseline (seed 19) with 95% paired, tissue-stratified image-bootstrap CIs (`scripts/compare_runs.py`;
test-set sampling noise only, NOT training noise). Decisions use the VAL fold (2); test (3) reported once.

| run | val mPQ | val d mPQ [CI] | val d Dead | test mPQ | test d mPQ [CI] | test d Dead [CI] |
|---|---|---|---|---|---|---|
| baseline (seed 19) | .4835 | | | .4951 | | |
| M1 Dead pixels x11 | .4798 | -.0036 [-.0076, .0000] | -.002 | .4915 | -.0036 [-.0085, +.0009] | +.009 [-.012, +.030] |
| M2 nuclei < 100 px x6 | .4822 | -.0013 [-.0050, +.0025] | +.001 | .4971 | +.0019 [-.0016, +.0059] | +.011 [-.004, +.027] |
| C1 sampler gamma 1.0 | .4840 | +.0005 [-.0041, +.0050] | -.010 | .4906 | -.0045 [-.0092, +.0003] | .000 [-.032, +.025] |

TTA test mPQ: baseline .5013, M1 .4975, M2 .4998, C1 .4953.

### Seed noise (baseline, split 1, seeds 19/1/2; `scripts/compare_runs.py`, runs/analysis/compare_split1_*.json)
| | val mPQ | val Dead | test mPQ | test Dead |
|---|---|---|---|---|
| mean | .4816 | .1643 | .4928 | .1393 |
| std over seeds | .0019 | .0029 | .0021 | **.0074** |

### Ablations vs. the 3-seed baseline (deltas, [95% image-bootstrap CI])
| run | val d mPQ | test d mPQ | test d Dead |
|---|---|---|---|
| M1 Dead x11 | -.0018 [-.0057,+.0019] | -.0013 [-.0061,+.0029] | +.0125 [-.0055,+.0295] |
| M2 small x6 | +.0006 [-.0035,+.0047] | **+.0042 [+.0009,+.0076]** | +.0141 [-.0027,+.0319] |
| C1 sampler 1.0 | +.0024 [-.0021,+.0067] | -.0022 [-.0066,+.0020] | +.0030 [-.0282,+.0287] |

Conclusions (single seed):
- **M1 (Dead-weighted CE) and C1 (full-balance sampler) are dead ends**: no val gain, no test gain.
- M2 flagged as the only candidate (single-seed test mPQ +.0042, CI excluding 0) pending seeds.

### M2 three-seed verdict (seeds 19/20/21 vs baseline seeds 19/1/2; split 1)
Runs `split1_npwce_small5{,_s20,_s21}` (`small_w 5, small_area 100`). The single-seed test gain
does NOT replicate (95% paired image-bootstrap CIs):

| | val d mPQ | test d mPQ | test d bPQ | test d Dead |
|---|---|---|---|---|
| 3-seed M2 | -.0007 [-.0029,+.0016] | -.0010 [-.0043,+.0022] | **-.0024 [-.0048,-.0002]** | +.0108 [-.0021,+.0238] |

Per-seed test Dead: M2 .1533/.1444/.1523 vs base .1424/.1308/.1446 — the Dead direction is
consistent (3/3 seeds above the baseline mean) but not significant, and it is bought with a small
significant bPQ loss; mPQ is unchanged.

=> **M2 rejected as a contribution** (a trade, not a gain). All pillar-A training-side ablations
(M1 Dead-weight, M2 small-weight, C1 balanced sampler) are null. The +.0042 single-seed effect was
seed luck — validating the rule below. The Dead detection deficit needs data-level intervention
(pillar B synthesis / copy-paste) or architecture changes, not loss reweighting.

General rule (validated): on this benchmark a single-seed mPQ change below ~.004 and any Dead-PQ
change below ~.01-.015 are indistinguishable from noise (seed std mPQ .0021 / Dead .0074,
test-sampling CI half-width .02-.03 on Dead). Every claim must be a multi-seed, 3-split mean.

## 2026-09-26 — Failure mining on out-of-fold predictions (pillar B input; `scripts/mine_failures.py`)
OOF setting: split-2 CellViT-UNI predicting fold 1 (fold 1 = split 2's val fold, never trained on).
fold-1 OOF reference: mPQ .5019 / Dead PQ .180, missed_bg .104. Identical patterns on fold 3 test
(split-1 model), so the layout statistics generalise. Per-nucleus records:
`runs/analysis/failures_fold{1,3}.csv.gz`.

missed_bg rate by class x area bin (fold 1 / fold 3 agree):

| class | <60px | 60-100 | 100-200 | >200 |
|---|---|---|---|---|
| Neoplastic | .75 | .35 | .13 | .03 |
| Inflammatory | .68 | .18 | .05 | .01 |
| Connective | .77 | .40 | .18 | .06 |
| Dead | .76 | .38 | **.15** | **.12** |
| Epithelial | .75 | .37 | .09 | .02 |

Key layout findings (both folds):
- **Missed Dead nuclei are ISOLATED, not touching**: Dead with 0 touching neighbours miss .39/.42
  (n=863/882 = ~90% of all Dead); missed_bg Dead touch another nucleus in <5% of cases. The
  originally hypothesised "Dead glued to Neoplastic" layout is NOT the failure mode.
- **Merge is an intra-class cluster problem**: of merged Dead, 54/74 (fold 1) and 77/94 (fold 3)
  touch another Dead (karyorrhexis clusters); cross-class touching is rare overall in PanNuke
  (per-nucleus touching-class flags are almost diagonal).
- Size dominates everything (<60px miss .68-.77 for ALL classes), but large Dead are still missed
  .12-.25 vs .01-.06 for other classes -> appearance effect on top of size.

=> Layout sampler priorities for pillar B synthesis: (1) insert ISOLATED small nuclei (Dead-looking,
  60-200px) into empty stroma; (2) dense same-class clusters (Dead chains, Connective fields) to
  train separation; (3) cross-class touching de-prioritised (rare in the data and not the failure).

## 2026-09-26 — PixCell-256 Cell-ControlNet smoke test (20x generator on 40x PanNuke; `scripts/pixcell_sample.py`)
Released ControlNet + UNI2-h context embedding, real fold-1 instance masks as condition, fp32,
20 steps, guidance 2.5 (`runs/pixcell/smoke/smoke_fold1.png`, 6 patches x 2 seeds; UNI2-h
cosine(gen, real) .25-.77). Runs in ~10 s/patch on one A100 — large-scale synthesis is cheap.

Visual + quantitative review (440 mask objects, per-object fill/IoU, off-mask hallucination scan,
Laplacian/colour stats; details in the session log):
- **Placement is faithful and training-usable**: ~95% of mask objects get a nucleus at the right
  place/size/orientation (median per-object IoU .68); phantom nuclei off-mask are rare (36 slivers
  in 12 samples, mostly boundary artefacts). Tiny hyperchromatic dots — our failure case — are
  reproduced reliably.
- **Systematic under-rendering of LARGE masks**: 4.5% of objects dropped + 13% partial, and the
  failures are size-biased (dropped median 322 px vs 137 px for well-rendered). Using these as
  labels would under-supervise exactly the big atypical/clumped nuclei and teach "label extends
  past the visible object" at large-nucleus boundaries. (Same direction as our Dead finding: large
  Dead are missed .12-.25 — the generator is weakest where the segmenter is weakest.)
- **Appearance gap is structural, not just colour**: chromatin rendered as flat blobs/shells/rim
  (no granular texture or nucleoli; Laplacian variance 0.76x real, 0.45x on sharp patches),
  consistent pink->violet shift with higher saturation and milky background, flat magenta
  eosinophilic fills, mechanical cobblestone stroma. A stain-normalisation augmentation would NOT
  fix the chromatin/boundary part.

=> **Decision: LoRA-adapt the generator to 40x PanNuke (train fold only)**, with adaptation
targeting nuclear internal texture and boundary realism at least as much as colour; before/after
adaptation run the per-object fill-rate, hallucination and Laplacian metrics over a few hundred
stratified patches (Dead/large nuclei stratum first). At synthesis time: filter or regenerate
samples whose large objects under-render (fill-rate gate), stratify by size.

## 2026-09-26 (morning) — Pillar B round 1 executing: CP1 copy-paste control + PixCell LoRA pipeline
Two tracks launched in parallel, per the pillar-B decision that Dead detection
needs data-level intervention:

**CP1 — failure-driven copy-paste (control arm; `src/nucseg/augment/copy_paste.py`)**
- Layout straight from failure mining: donors (50-400 px, class-weighted Dead 0.4 / others 0.15)
  pasted into nucleus-free stroma with an 8 px clearance (never touching, matching the "missed Dead
  are ISOLATED" finding); prob 0.5, Poisson(3) insertions per augmented patch, applied BEFORE the
  geometric/photometric augmentations so pasted nuclei are rotated/colour-jittered with the patch.
- Montage review (sonnet subagent, `runs/analysis/copy_paste_{noaugs,full}_v2.png`) caught real bugs
  in v1 that unit tests missed: donors clipped at the patch border became straight-edged fragments;
  pasting after augmentation let fakes keep raw colour/orientation (a learnable shortcut); occasional
  collision with UNANNOTATED nuclei (PanNuke annotation gaps) and gross hue mismatch. v2 fixes:
  border-clipped donors excluded from the bank, per-channel stroma colour shift (donor stroma ->
  destination stroma), Otsu-darkness guard against unannotated nuclei, paste-before-augment order.
  6 unit tests (`tests/test_copy_paste.py`): bank/class counts, isolation clearance, label
  consistency, donor-class mix, border-donor exclusion.
- Run `runs/cellvit_abl/split1_cp1` (split 1, seed 19, CellViT-UNI, --cp-prob 0.5, TTA),
  ~3 h on a shared A100. Compare vs baseline seeds 19/1/2 (`scripts/compare_runs.py`). This is the
  decisive experiment for "is the Dead detection deficit data-fixable at all".

**PixCell LoRA — 40x adaptation (`scripts/train_pixcell_lora.py`, `src/nucseg/pixcell.py`)**
- LoRA (r=16, alpha=16) on the DiT attention projections only (8.3 M trainable / 616 M), ControlNet
  and VAE frozen, epsilon objective, CFG-condition dropout 0.1 implemented exactly like inference
  (dropped rows get zero controlnet contribution — the block injection is additive, verified in
  `pixcell_controlnet_transformer.py`). Conditions cached once per fold (VAE latents of image and
  binary mask + UNI2-h CLS embedding of the real image; `cache_fold{k}.npz`).
- Fold 1 (= split-1 TRAIN fold; no leakage) 5000 steps bs 16 lr 1e-4 (~40 min on a shared A100):
  `runs/pixcell/lora_fold1`.
- Quantitative generator eval harness (`scripts/pixcell_eval.py`): 200 stratified patches
  (100 Dead-rich / 50 large-nucleus-rich / 50 random) re-generated from GT masks; per-object
  rendering scored by an OUT-OF-FOLD detector (split-2 CellViT-UNI, fold 1 = its val fold):
  coverage >= .5 rendered / .2-.5 partial / < .2 dropped, by size stratum and class; phantom rate,
  Laplacian variance ratio (global + in-nuclei), OD colour distance, UNI2-h cosine. Baseline
  (pre-LoRA) run (`runs/pixcell/eval_base`); smoke numbers on 8 patches: Dead <100 px
  rendered .63 vs 1.0 for larger Dead, Laplacian ratio .68 in nuclei (texture deficit confirmed).
- Next when LoRA finishes: `pixcell_eval.py --lora runs/pixcell/lora_fold1/transformer_lora.pth`
  before/after comparison + montage review; then layout sampling at scale.

## 2026-09-26 (midday) — PixCell LoRA round-1 verdict: over-adapted; synthesis pipeline works, appearance not yet
- **Generator eval harness numbers** (`runs/pixcell/eval_base` vs `eval_lora16`, 200 stratified fold-1
  patches, paired context, same seeds; OOF split-2 detector): the r16/5000-step LoRA made generation
  WORSE — od_mean_l1 .098 -> .266, uni_cos .472 -> .327, in-nuclei Laplacian ratio unchanged .72,
  global Laplacian 0.95 -> 1.24 (speckle, not chromatin), Dead<100px render .650 -> .606, phantom
  px .0036 -> .0073. **Verdict: over-adapted (contrast/saturation blow-up).**
- **Synthesis smoke review** (32 images, LoRA + RANDOM context + bf16, sonnet subagent): geometry,
  density and label placement good (outlines on visible nuclei, phantom-labeled area 3.8%,
  teacher keep rate .886), but appearance failed: nucleus interiors crushed to near-black
  (median 42 vs real 121/255), per-patch bimodal colour (uniform violet or brick-red, saturation
  .40 vs .29), soft focus (Laplacian 3-10x low in 8/12 rows), milky fog / streak stroma, halo rims.
  10-20% of visible nuclei per patch left unlabelled (small, dark) — the known teacher-blind-spot
  caveat, confirmed visually.
- Diagnosis split: paired-context eval (above) isolates the LoRA as the main cause; random context
  amplifies it (off-manifold conditioning). Fixes launched:
  1. **LoRA round 2**: r8, lr 5e-5, 2500 steps, step-stamped snapshots every 250 (`lora_fold1_r8`) — checkpoint selection by image-space metrics (od_mean_l1 / uni_cos / in-nuclei
     Laplacian on a 60-patch eval), NOT by the noisy epsilon-val (its trajectory was flat/noisy).
  2. **Paired-context synthesis** (default in `scripts/synth_pannuke.py`; random ctx now opt-in).
  3. Batched sampler verified equal to the reference pipeline (mean|diff| 0.015/255, fp32).
- `scripts/synth_pannuke.py` end-to-end: layout perturbation (2.38 inserted/layout) -> batched
  generation -> OOF teacher labelling with coverage>=.4 + type prob>=.6 -> keep-frac .7 gating;
  866/978 objects kept, 1/32 images dropped on the smoke batch.
- Training-side synthetic mixing implemented (`--synth DIR --synth-frac r`; synthetic samples carry
  the base patch's tissue label and the same cell+tissue sampler semantics).

## 2026-09-26 (afternoon) — CP1 copy-paste verdict: NEGATIVE (hurts Dead typing)
Run `split1_cp1` (seed 19, split 1, cp-prob .5, paste-before-augs, all v2 fixes) vs the 3-seed
baseline, paired tissue-stratified image bootstrap (`runs/analysis/compare_split1_cp1{,_val,_tta}.json`):

| fold | d mPQ [95% CI] | d bPQ [95% CI] | d Dead PQ [95% CI] |
|---|---|---|---|
| val 2    | +.0048 [+.0002,+.0090] | +.0002 [-.0018,+.0022] | +.0235 [-.0079,+.0652] |
| test 3   | +.0009 [-.0024,+.0043] | -.0010 [-.0031,+.0009] | **-.0310 [-.0576,-.0100]** |
| test 3 TTA | -.0004 [-.0035,+.0027] | -.0005 [-.0022,+.0011] | **-.0315 [-.0570,-.0116]** |

- The test Dead drop is real (>4x seed std .0074, CI excludes 0, TTA-consistent) and acts mostly
  through TYPING, not detection: among matched GT-Dead, correct type .690 -> .609; matched-Dead
  precision .512 -> .484; matched Dead 497 -> 484 (confusion matrix, test fold 3). Confusion moves
  both ways (Dead->Neo/Infla up AND Conne->Dead up) — a noisier Dead boundary, not a recall gain.
- The VAL fold pointed the opposite way on Dead (+.0235, n.s.): single-seed Dead PQ swings ~.05
  between folds. Third independent confirmation of the multi-seed/multi-fold rule.
- Interpretation: transplanting real-appearance donors (colour-shifted into random stroma) teaches
  "small isolated nucleus in clean stroma" as a Dead-ish cue and blurs the type boundary instead of
  improving detection. => Layout-level intervention with REAL appearance does NOT fix the Dead
  deficit; pillar B stands or falls with full synthesis (appearance generated in-context).
- mPQ is neutral everywhere (+.0009 test) — the pastes are not toxic to overall segmentation, the
  damage is Dead-specific.

## 2026-09-26 — PixCell LoRA round 2: abandoned; Reinhard colour fix instead
- r8 / lr 5e-5 / 2500 steps with snapshots every 250 (`runs/pixcell/lora_fold1_r8`, 60-patch
  image-space eval): EVERY checkpoint is worse than the base model (od_mean_l1 .60-1.10 vs base
  .116; uni_cos .13-.25 vs .496; even step 250 gives od .80). Round 1 (r16/5000, od .098 -> .266)
  was not a tuning accident — any DiT LoRA adaptation pulls generation off the 20x prior. Route
  abandoned; documented in the `train_pixcell_lora.py` header.
- Post-hoc colour fix instead: Reinhard LAB transfer of each generated image to its base real patch
  (`nucseg.pixcell.reinhard_lab`): probe on 24 pairs, od_mean_l1 .0723 -> .0094; mean saturation
  70.0 -> 68.9 (real 69.7). Synthesis recipe locked: BASE model + PAIRED context + Reinhard
  (`scripts/synth_pannuke.py` defaults; batched sampler verified equal to the pipeline, 0.015/255).
- SYN-v2 smoke (32 images, fold 1, that recipe): teacher keep rate .901 (729/809), 0 images
  dropped, 2.69 insertions/layout; colour vs paired real: median RGB |diff| ~3/255, saturation
  68.5 vs 69.6; Laplacian ratio .70 (the base model's texture ceiling — accepted).

## 2026-09-26 (evening) — SYN-v2 review: appearance PASS; Dead labels structurally ZERO (diagnosed)
- Sonnet montage review (12 rows, real|gen|label): every v1 defect fixed — patch median V 165-231
  (v1: 42), per-patch saturation within 1-3 of real, no milkiness, halos gone (dark-rim like
  real), 0/12 failure rows; placement 12/12 correct, 0/222 mask objects under-rendered by an
  inside-vs-ring contrast test; unlabelled-but-visible nuclei ~4-5% overall (worst row 25%).
  Residuals (accepted): nuclei somewhat darker than real on intrinsically dark patches (row 2),
  per-nucleus hue dispersion ~0.5x real, sharpness 0.45-0.69x real on high-frequency rows.
- **Blocker found by the review: 0 of 729 kept objects typed Dead** (expected ~60: ~11% of
  base-layout GT objects are Dead + 38% of the 86 inserted donors were Dead-class).
- Diagnosis (`scripts/diag_synth_dead.py`, same split-2 OOF teacher, same labelling code path):
  - REAL fold-1 Dead-rich patches: GT-Dead objects with coverage >= .4 are argmax-typed Dead 77%
    with MEDIAN max-prob .99 (97% clear the .6 gate) -> the teacher and the threshold are fine.
  - SMOKE synthetics re-read: P(Dead) = .000-.005 for ALL 729 kept objects (max .005; not a
    runner-up/confusion pattern, which would sit at ~.2-.4) -> the generated small nuclei simply
    carry no Dead cue.
- Structural cause: the ControlNet condition is a BINARY mask (class-agnostic) and the context
  embedding carries no per-nucleus class information; the generator's 20x prior renders isolated
  small nuclei as lymphocyte-like objects regardless of the donor's class. Class-targeted
  appearance is not expressible in this generator without re-training it (the abandoned LoRA
  route). Forcing donor-class labels anyway would recreate the CP1 failure mode (labels that do
  not match appearance).
- Consequence for pillar B: synthesis can add DETECTION supervision for small isolated nuclei
  with SELF-CONSISTENT (teacher-read) types — the CP1 harm channel (40% of pastes carrying forced
  Dead labels) is structurally absent, and the label/appearance pairs stay natural. Dead PQ can
  then only move through its detection factor. => SYN1 launched as the completing arm (3000
  images, fold 1, class-agnostic labels, synth-frac .5 first); if it also fails, pillar B closes
  with "small-nucleus detection is not data-fixable by real-appearance pastes or in-context
  synthesis".

## 2026-09-26 (night) — SYN1 verdict: null overall, Dead +1.0 PQ at the seed-noise edge; pillar B closes
- SYN1 = exact split-1 base recipe (seed 19) + 1146 synth (synth-frac .5) from
  runs/pixcell/synth_fold1 (3000 img, keep-rate .886); trained to completion, output
  runs/cellvit_abl/split1_synth1. Test fold 3, last ckpt:
  - mPQ .4900 / bPQ .6604 (base .4951 / .6676); TTA .4956 / .6659 (base .5013 / .6728). mPQ sits
    at the BOTTOM of the 3-seed range (seed1 .4910, seed2 .4924, seed19 .4951) — null-to-slightly-
    negative overall.
  - Dead PQ .1521 vs .1424 (TTA .1570 vs .1461): above all 3 seed draws (max .1446) but Dead seed
    noise alone spans .131-.142, so ~+1 pt is at the edge of noise. Decomposition: Dead PQ+ flat
    (.2761 vs .2759), DQ+/SQ+ flat => the gain is correct-TYPE pairing (Dead-typed matches 367 vs
    352 of ~500 Dead GT), not better segmentation; Inflammatory PQ -1.0/-1.6 pts (possible
    small-nucleus synth side-effect).
  - Loss re-weighting dominates: npwce_small5 remains the best Dead arm (TTA Dead .1626, mPQ
    .4998) — better than SYN1 on BOTH axes; data-side levers (CP1 paste, SYN1 synthesis) add
    nothing on top.
- Verdict per the pre-registered rule: SYN1 fails to move the needle => pillar B closes: Dead /
  small-nucleus performance is not data-fixable via real-appearance pastes (CP1) or in-context
  synthesis with self-consistent labels (SYN1); the remaining lever is the loss (npwce_small5).

## 2026-09-26 (late) — Pillar C launched: cross-domain sets converted + reviewed; PanNuke-CF renderer validated
- **External datasets** (`scripts/prepare_external.py`, `data/external/`):
  - CoNIC2022 (HF MedOtter mirror, 4,981 patches): the 112 `source=pannuke` patches EXCLUDED
    (leakage guard); 20x -> 2x bilinear upsample (maps nearest) -> quad-split 256 -> **19,476
    tiles**; classes mapped Neut/Lymph/Plasma/Eos -> Inflammatory, Epithelial, Connective
    (Neoplastic & Dead absent). Single pseudo-tissue "Colon".
  - MoNuSAC (HF RationAI, official TEST split, 101 whole images 40x): RGBA->RGB (alpha verified
    opaque), Ambiguous instances dropped (2,403), tile 256 non-overlap, keep full tiles with >=1
    nucleus -> **443 tiles**; Epi->Epithelial, Lymph/Macro/Neut->Inflammatory; tissues
    Breast/Kidney/Lung/Prostate.
  - Montage review (sonnet subagent, GT-ellipse overlays): **no conversion bug** — annotations
    sub-pixel aligned (0.4-1.1 px), CoNIC-upsampled vs MoNuSAC-native nuclear scale within ~13%
    in the correct direction, upsampling interpolated not blocky. Noted: CoNIC has some
    mostly-acellular tiles + border-clipped nuclei chains (inherent to quad-split, documented).
- **Eval plumbing**: `evaluate()` gained an optional `tissue_names` (default PanNuke; externally
  readable per-tissue reports); `src/nucseg/data/external.py` mirrors PanNukeFold with a lazy
  per-image `gt_channels` (flat RAM); `predict_external.py` / `collect_external.py` for
  prediction and cross-split aggregation. Identity-prediction smoke = 1.0 PQ. Fixed a silent
  repo bug: `.gitignore`'s `data/` swallowed `src/nucseg/data/` — `pannuke.py`/`prepare.py`
  were NEVER tracked; now root-anchored `/data/` and committed.
- **Zero-shot predictions running**: CellViT-UNI splits 1-3 x {conic, monusac} x
  {plain, TTA}. HoVer-Net after.
- **PanNuke-CF** (`scripts/render_pannuke_cf.py`): fixed fold-3 GT labels, factorial appearance
  intervention — context {paired, swapped-tissue} x Reinhard target {self, donor} = arms
  control/stain/ctx/tissue (+ `real` reference arm on the same 500 patches; 3 generator-seed
  replicates of control for the noise floor). 8-patch smoke (split-2 model): real .4735 ->
  control .4860 (generator renders EASIER images: bPQ .71 -> .80, but strict mPQ drops .47 ->
  .40 — generated nuclei carry type-inconsistent appearance), tissue arm .2807 (strong stress
  signal, missed_bg .07 -> .17). Smoke montage review forced two fixes: (a) blank/no-label
  patches excluded (context embedding is a CONTENT channel — hallucinated a whole tissue on an
  empty patch), (b) context donors luminance-guarded (±35 mean-L; near-white donor bleached a
  patch to 88% white). Full render (500 patches/arm) running.

## 2026-09-27 — External zero-shot cross-domain (pillar C): CoNIC + MoNuSAC, 3-split mean
`scripts/predict_external.py` -> `scripts/collect_external.py` (mPQ/bPQ/F_d 3-split mean +- std;
reports `runs/analysis/ext_{cellvit,hovernet}.json`). HoVer-Net has no TTA row
(hardware budget; CellViT TTA answers the does-TTA-help question). In-domain reference from the
baselines table above: CellViT .4995 mPQ / .6653 bPQ, HoVer-Net .4564 / .6613.

| model | CoNIC mPQ / bPQ / F_d | MoNuSAC mPQ / bPQ / F_d |
|---|---|---|
| CellViT-UNI | .3351 +- .0037 / .5314 / .7816  [TTA .3398 / .5381 / .7858] | .2495 +- .0218 / .5616 / .7652  [TTA .2484 / .5650 / .7672] |
| HoVer-Net | .2610 +- .0061 / .5136 / .7669 | .0370 +- .0003 / .2756 +- .0701 / .3658 +- .0980 |

- **Domain drop (in-domain mPQ minus external)**: CellViT -.164 (CoNIC) / -.250 (MoNuSAC);
  HoVer-Net -.195 / -.419. Both degrade, but **HoVer-Net nearly collapses on MoNuSAC**
  (F_d .37, bPQ split std +- .070 vs +- .003 on CoNIC — per-split F_d ranges .26-.50; failure
  mode is missed_bg, i.e. detection, not typing).
- **TTA does not rescue cross-domain**: +.005 mPQ on CoNIC, -.001 on MoNuSAC (CellViT).
- **Per-class PQ (present classes only)**: CoNIC — CellViT Infla .425 / Conne .371 / Epith .246;
  HoVer-Net .339 / .255 / .193. MoNuSAC — CellViT Infla .309 / Epith .213; HoVer-Net .061 / .012.
  Epithelial is the weakest class on CoNIC for both models; Inflammatory transfers best.
- Classes absent in GT: CoNIC has no Neoplastic/Dead, MoNuSAC has no Neoplastic/Connective/Dead
  (mapping in the 2026-09-26 entry); strict mPQ on absent classes is excluded per official protocol.
- CF arms (real/control x3/stain/ctx/tissue on fold-3 500 patches) are predicted + evaluated for
  both models (2 splits each; `eval_ext_*` dirs) — paired bootstrap/rank-agreement analysis via
  `scripts/analyze_cf.py` is the next step.

## 2026-09-27 — PanNuke-CF analysis (pillar C): noise floor, paired deltas, rank agreement
`scripts/analyze_cf.py` on 4 evaluators (CellViT-UNI / HoVer-Net x splits 1/2; 494 images/arm,
fixed fold-3 GT; log `logs/analyze_cf.log`; matched PanNuke-test references).

- **Generator noise floor**: control-replicate mPQ std .0039-.0055 (range <= .0131) — an order of
  magnitude below the stress effects below, so arm deltas are attributable to the intervention.
- **real vs control** (same patches): CellViT +.042/+.051 mPQ (CI excl. 0) — but split by head:
  bPQ control .78 vs real .67 (generated nuclei are EASIER to segment), strict .36 vs .45 (their
  TYPE appearance is less consistent with GT labels). Both baselines show this; keep using mPQ +
  strict for CF arms.
- **Paired deltas vs control mean** (image-bootstrap 95% CI, all CIs exclude 0):
  - `stain` (Reinhard to donor stain): CellViT -.023/-.024, HoVer-Net -.061/-.071 — mild stress.
  - `ctx` (paired->swapped-tissue context): CellViT -.187/-.192, HoVer-Net -.135/-.144 — strong.
  - `tissue` (ctx + donor stain): CellViT -.193/-.189, HoVer-Net -.147/-.145 — strongest for CellViT.
  - Class structure: Epithelial collapses under ctx/tissue (CellViT -.41/-.42 PQ) while Dead loses
    -.04..-.10 — context swap mostly destroys type information, not detection (F_d only -.10).
- **Rank agreement (the pillar-C validation) FAILS at architecture level**: real domain drop
  (PanNuke test - mean ext mPQ) CellViT .21 vs HoVer-Net .30, but ctx CF drop CellViT .19 >
  HoVer-Net .14 -> Spearman **-1.00** across the 4 evaluator points (tissue -.60, stain +.60 ns).
  Interpretation: the CF stress axis measures **context/stain sensitivity**, which is NOT the axis
  that dominates real transfer — HoVer-Net's large real drop is its MoNuSAC *detection* collapse,
  and the less context-dependent (weaker) model is hurt less by context swap. Caveat: 4 evaluator
  points = 2 architectures x 2 near-identical splits, so the correlation is effectively the single
  architecture contrast; next step is a 3rd architecture (HoVer-NeXt-T per-fold weights) +
  PUMA (apoptotic->Dead) as an additional real external set before concluding.

## 2026-09-27 (eve) — HoVer-NeXt-T port decoded: flat validation decode + VAL-tuned thresholds (.458 mPQ)
- Port gap diagnosed with two ablation scripts (`hn_postproc_ablation.py`, `hn_thresh_sweep.py`,
  same raw maps scored under 3 decodings + a 5x5 flat fg/seed grid): the maps are FINE — the
  bottleneck was the decode. The vendored per-class inference-repo decode (their
  `pannuke_test_param_dict.json` thresholds) gives mPQ .365 on fold-3 400-img; their own
  **validation decode** (flat fg/seed + softmax-sum typing, `validation.py make_prediction`)
  gives .425 with their defaults (.7/.3); grid best .447. Their pp (hole fill + PANNUKE size
  filters) HURTS the flat decode (.425 -> .326) — dropped.
- The released 0.477 comes from `evaluate.py --tta 16`: 16 stochastic spatial+COLOR-aug views
  (inverse-transformed, averaged) + per-class fg/seed tuned ON THE TEST FOLD (`get_pp_params`).
  Not reproduced (stochastic color TTA at eval + test-tuned thresholds is their recipe, not a
  fair-comparison decode); we instead tune the flat pair on each split's VAL fold
  (`decode.json` per weight dir; .6/.7, .6/.7, .6/.6) — same protocol as our other baselines.
  (2026-09-28 audit: decode.json records sweep fold 2/1/2 for splits 1/2/3
  = the VAL folds; test folds 3/3/1 were never swept.)
- **Sanity, full test folds (3-split mean): mPQ .4579 / bPQ .6367 / Dead .1361**
  (paper .477/.656/.154; old port decode .3582/.5030/.080). Sits between HoVer-Net
  (.4564/.6613) and CellViT-UNI (.4995/.6653) on mPQ — a usable 3rd evaluator.
  Per-split: s1 .4423/.6247, s2 .4644/.6454, s3(test fold1) .4670/.6400.
- Notable behaviour (ERROR PROFILE, corrected 2026-09-28 audit — an earlier version of this line
  had the comparison inverted): HoVer-NeXt-T's flat decode merges ~2x MORE touching nuclei than
  the watershed decodes (merged GT rate .096-.101 vs CellViT-UNI .044 / HoVer-Net .049) and
  misses more background nuclei than CellViT-UNI (.122-.131 vs .102-.109; HoVer-Net .119-.125
  sits in between) — a different error profile, which is exactly what the rank-agreement
  analysis needs. (The "~.14" previously attributed to HoVer-Net was the merge rate of the
  HNXT per-class-decode ablations, .156-.177 in hn_pp_ablation.log, not of any in-domain
  evaluator.)

## 2026-09-27 (eve) — PUMA third external set (melanoma, apoptotic->Dead): converted, reviewed, predicted
- `prepare_puma.py` (Zenodo 15050523 ROI pack): 3,278 tiles, tissues Melanoma-primary/-metastatic
  (1648/1630); classes mapped nuclei_tumor->Neoplastic, lymphocyte/histiocyte/melanophage/plasma/
  neutrophil->Inflammatory, stroma/endothelium->Connective, **apoptosis->Dead (1,991 nuclei)**,
  epithelium->Epithelial (all 5 PanNuke classes present). Scale verified vs PanNuke fold-3 GT
  (median nucleus 390 vs 464 px; tif tags 0.226 um/px but effective scale behaves ~0.5 um/px).
- Sonnet montage review (GT outlines over 16 tiles): PASSED — 1-px contours boundary-exact,
  class colours coherent (black-stroke Dead confined to pyknotic/dumbbell nuclei, 1-3/tile;
  blue lymphocyte fields; orange spindle stroma), no conversion artifacts. Caveat inherited
  from PUMA (HoVer-Net-initialised, pathologist-corrected GT): ~5-15% of nuclei in the densest
  tiles carry no label -> absolute recall/mPQ on PUMA has a ceiling; model comparisons share
  the same GT so remain fair.
- Zero-shot (3-split mean, no TTA): **CellViT-UNI mPQ .4518 +- .0031 / bPQ .7177 / F_d .8831;
  HoVer-Net .3203 +- .0136 / .7134 / .8714**. Domain drop (in-domain mPQ - PUMA): CellViT
  -.048, HoVer-Net -.136 — ordering matches CoNIC/MoNuSAC (CellViT more robust), and unlike
  MoNuSAC there is NO detection collapse (F_d ~.87-.88): the transfer loss is in TYPING
  (CellViT strict .37 vs mPQ .45; Dead PQ per-arch mean CellViT .0024 / HoVer-Net .0327
  [.0211/.0209/.0559] / HoVer-NeXt-T .0046, Epithelial .09-.20). Apoptotic bodies
  remain undetectable-as-Dead cross-domain for both models — the Dead failure mode follows us
  out of domain (and PUMA's apoptotic GT is the only large external Dead supply we found).

## 2026-09-28 — Pillar C closes: 3-architecture CF verdict (inverted on mPQ, positive on the typing axis)
HoVer-NeXt-T predictions were added everywhere (external x9, CF arms x2 splits);
`analyze_cf.py` extended with an axis decomposition + axis-matched rank agreement (exact
permutation p; `logs/analyze_cf3.log`, `runs/analysis/ext_hovernext.json`).

- **HoVer-NeXt-T external zero-shot (3-split mean mPQ/bPQ)**: CoNIC .2786/.4811,
  MoNuSAC .1123/.4025, PUMA .3813/.6888. Real domain drop (per-split in-domain reference,
  2-split mean): .199 — ordering CellViT .154 < HoVer-NeXt-T .199 < HoVer-Net .246,
  consistent across all 3 sets.
- **Headline rank agreement now has 6 evaluator points (3 architectures x 2 splits)**:
  ctx rho **-0.89** (perm p .033), tissue -0.89 (.033), stain +0.43 (ns). Architecture means
  (honest n=3): ctx/tissue **-1.00** — perfect inversion. The 2-architecture caveat is closed:
  more architectures made the inversion stronger, not weaker.
- **Axis decomposition (why)** — real drops split into detection (F_d) vs typing (bPQ-mPQ gap):
  | evaluator | real mPQ | F_d | typing | ctx CF mPQ | F_d | typing |
  |---|---|---|---|---|---|---|
  | CellViT-UNI (s1/s2)  | .155/.152 | **.007/.007** | +.092/+.084 | .187/.192 | .101/.103 | +.081/+.082 |
  | HoVer-Net (s1/s2)    | .242/.250 | **.149/.176** | +.076/+.062 | .135/.144 | .082/.086 | +.033/+.035 |
  | HoVer-NeXt-T (s1/s2) | .190/.207 | **.046/.043** | +.078/+.092 | .161/.164 | .092/.089 | +.062/+.066 |
  Between-architecture real differences live almost entirely on the **detection** axis
  (CellViT loses nothing, HoVer-Net loses .15-.18 = MoNuSAC collapse); typing-gap increase is
  similar for all (.06-.09). But CF rendering gives a **homogeneous** detection stress
  (.082-.103 for every architecture) — with labels fixed and the generator re-rendering
  well-formed nuclei, the detection axis cannot be stressed by construction.
- **Axis-matched rank agreement**: typing axis ctx rho **+0.71** (all 6 pts, perm p .136) /
  **+1.00** (arch means); tissue typing +0.83/+1.00. Detection axis: -0.83/-1.00 (CF has no
  variance there, so the correlation is meaningless-inverted). Stain arm's largest signal is
  also on F_d (+0.83, p .058 — HoVer-Net biggest on both sides), same story at weak signal.
- **Verdict (pillar C terminal)**: PanNuke-CF label-fixed counterfactual rendering is NOT a
  valid proxy for real cross-domain transfer — at the mPQ level it *inverts* the architecture
  ranking (the model most sensitive to context swap in-domain is the most robust transferrer).
  Mechanistically, its blind spot is exactly the detection axis, which is where real transfer
  differences concentrate. Where CF CAN apply stress (typing under context/stain mismatch), it
  ranks architectures consistently with real typing transfer. ⇒ report CF as a **typing
  stress test**, never as a transfer surrogate. Honest caveats: n=3 architectures (arch-mean
  |rho|=1 has exact p=.33; the defensible evidence is perfect sign inversion across 3 archs x
  2 splits plus the axis mechanism), external typing-axis differences are small (real typing
  gap range .069-.088 vs detection range .007-.176).

## 2026-09-28 (evening) — Benchmark artifacts: border fragments, the Uterus Dead subpopulation, cross-architecture consensus
`scripts/analyze_artifacts.py` over the per-GT records of our three evaluators (CellViT-UNI /
HoVer-Net / HoVer-NeXt-T; evaluation records collected under a common input root). Consensus stats use splits 1+3 = test folds 3+1 (129,872 GT nuclei; rows verified
aligned across models: identical image/cls/area); per-model stats use all 3 splits. Frame
cached at `runs/analysis/artifacts_gt.csv.gz`. Survey of the same date (leaderboard / datasets
/ multimodal / generators): `docs/literature/survey_2026_09.md`.

**1. Most "missed small nuclei" outside Dead are patch-border fragments.**
- 89% of GT nuclei < 60 px touch the 256 px patch border (31% of all GT do); 62% of < 60 px GT
  are missed by ALL 3 models (interior < 60 px: 33%).
- Border-touching objects are 57% of CellViT-UNI false negatives (missed_bg + missed_shape);
  40% of FNs are border AND < 100 px.
- Interior-only missed_bg: Neo .037 / Inf .027 / Epi .028 / Con .095 / **Dead .310** (Dead with
  border .351). ⇒ the stage-1 "small nuclei are missed for every class" observation is a
  border-fragment effect except for Dead; M2's < 100 px up-weighting mostly up-weighted
  fragments (its null result is unsurprising in retrospect).

**2. The Dead deficit is class-specific and shared by all three DENSE decoders.**
- Size–missed_bg curves coincide across ViT-L/16 / full-resolution ResNet-50 / ConvNeXt:
  < 60 px .73/.75/.76, 60–100 .33/.37/.33, 100–256 .11/.14/.14, ≥ 256 .03/.04/.05.
- Dead interior missed_bg by size (CellViT): .67 / .30 / .19 / .17 across the same bins
  (all-3-consensus .48/.18/.12/.13) — the appearance effect survives every size bin.
- Dead F_c (HoVer-Net def., 3-split means): CellViT-UNI .359 (.314/.322/.441),
  HoVer-NeXt-T .349 (.303/.322/.422), HoVer-Net .230 (.202/.213/.274). Reference points:
  HoVer-NeXt paper .49 (test-fold-tuned per-class thresholds + 16 stochastic TTA views),
  detection-first KongNet .59 [paper-reported, unverified]. ⇒ "the deficit is architectural"
  is narrowed to "shared across the dense decoders tested"; the detection-first family is
  untested under our strict protocol (phase-2 line A).

**3. Official Dead PQ is an image-subset metric; the split-to-split Dead gap is test-fold composition.**
- Official per-class PQ averages per-image PQ over images CONTAINING the class: Dead is scored
  on 97 images (fold 3) / 65 (fold 1); the top-10 Dead images hold 43% / 51% of all Dead GT.
- Fold 3 contains 34 Uterus Dead images (fold 1: 2). Their Dead PQ is .01–.05 for every
  architecture; interior Dead there are intraluminal clusters of pale shed cells (visual
  audit), missed by all 3 models on 69% of nuclei, median area 156 px (NOT small).
- Excluding Uterus images, Dead PQ per split (official → excluded):
  CellViT-UNI .195/.227/.231 (.142/.162/.224), HoVer-Net .133/.147/.175 (.103/.108/.170),
  HoVer-NeXt-T .143/.171/.197 (.102/.116/.191). Published CellVTA Dead (.145/.169/.240) shows
  the same fold-3 dip. ⇒ cross-split Dead comparisons (and any "Dead improved" claim) must
  control for image composition.
- Dead GT concentration (folds 3+1, n=2,024): Lung 54% + Colon 20% + Uterus 9%. Lung is the
  worst tissue overall (CellViT 3-split mPQ .402; 23% of its nuclei < 100 px); Colon has the
  highest merge rate (.089) and lowest bPQ (.570).

**4. No local-suppression halo; clustering hurts Dead.**
- Interior non-Dead nuclei within 40 px of a consensus-missed Dead are all-3-missed at .045 vs
  .033 far (baseline .027): no meaningful co-missing of neighbours.
- Dead in same-class clusters (≥ 3 Dead within 40 px; 58% of Lung / 78% of Colon Dead) are
  consensus-missed .237 vs .159 for scattered Dead (interior) — dense Dead fields are harder,
  consistent with the karyorrhexis-cluster merge finding, but isolation does not protect Dead
  either (stage-1 mining: isolated Dead miss .39–.42).

**Implications (phase 2, agreed with the user 2026-09-28):** report Dead four ways (official /
Uterus-excluded / interior recall / per-nucleus F_c); pre-registered endpoints for any Dead
intervention = interior Dead recall + Uterus-excluded Dead PQ with bPQ non-regression, always
3-split × multi-seed; the resolution hypothesis needs a detection-first or 2×-resolution test,
not a third dense decoder. Stage report 1 (EN+ZH) revised accordingly this date (wording +
bibliography corrections; see git).

## 2026-09-28 (late evening) — Phase 2 launched: B1 (2x resolution) training; A1 assets staged
- **B1 run live**: `runs/cellvit_uni_x2/split1` (seed 19, `--upscale 2 --tta`): the exact
  CellViT-UNI split-1 recipe with the working resolution doubled to 512 (bilinear image upsample
  AFTER augmentation / nearest labels — single-variable change; commit d2deb93). Encoder gradient
  checkpointing on (measured bs16@512 unfrozen: 39.8 GB peak); epoch time 242 s -> ~9 h for 130
  epochs. Prediction post-processes at 512 and downsamples to 256 for the standard evaluator
  (per-instance type-prob table computed at model resolution — a resolution mismatch caught by
  the smoke test). Compare vs baseline seeds 19/1/2 with `scripts/compare_runs.py` + the
  artifact-aware Dead views (same rule as above: 3 seeds × 3 splits before any claim).
- **A1 assets staged** (10.3 GB): LKCell-L per-fold
  `model_best.pth` + config.yaml for all three splits (fold mapping verified from the run
  configs, see survey §4) and `KongNet_PanNuke_{1,2,3}.pth`; repos cloned
  (`hustvl/LKCell`, `Jiaqi-Lv/KongNet_Inference_Main`). KongNet is a 6-head CenterNet-style
  detector (per-class heatmap + regression channels, NMS post-proc, patch 256); its fold->ckpt
  mapping is NOT documented -> resolve empirically on validation folds before any test-fold run
  (never tune on test). Integration plan: vendor their decode, feed instance/type outputs into
  `nucseg.metrics.pannuke_eval` like HoVer-NeXt-T.

## 2026-09-29 (early hours) — post-review fixes to the phase-2 upscale plumbing
Code review of d2deb93/a70ac15 found 5 correctness bugs; all fixed with tests (32 passed).
- **inst_probs table vs downsampled map** (engine.py): the per-instance table was built from the
  512-px map while the saved inst map is nearest-downsampled to 256 — ids living only on odd
  rows/cols (1-2 px Dead fragments) stayed in the table but vanished from the map, crashing
  retype_conch (lut[ids]) or inflating npred. Table ids are now enumerated from the FINAL map
  (`instance_table_rows`), type probs still average at model resolution. The B1 run's own auto
  eval was never affected (train_cellvit uses inst_probs=False).
- **encoder grad checkpointing was a no-op**: timm 1.0.9 only checkpoints inside
  forward_features; UNIEncoder overrides forward. Now applied per block (use_reentrant=False,
  inert while frozen). The live B1 run trained the unfrozen phase WITHOUT checkpointing
  (48.7 GB, fits — only the code comment's claim was wrong); future bs increases are safe.
- **resolution now resolves from the run's config.json everywhere** (`run_upscale`):
  train_cellvit --skip-train, predict_external (cellvit branch) no longer silently evaluate a
  512-trained model at 256; predict_cellvit's ad-hoc logic replaced by the shared helper.
- **artifacts encode working resolution** (`res_tag`, "_x2"): pred_fold{k}[_x2][_tta].npz,
  eval_test_fold{t}[_x2][_tta], eval_ext_{data}[_x2][_tta], conch cache conch_fold{k}[_x2...].
  Native 256 runs keep the old names (no migration). NOTE: the in-flight B1 run (old code in
  memory) will write UNTAGGED pred_test_fold3[_tta].npz — its results are the x2 ones;
  re-running predict on the updated code regenerates them tagged.
- analyze_artifacts.py cleanup: reuses metrics.instance.centroids + gt_records tissue (output
  verified byte-identical), dead scaffolding removed, --out-gt documented as opt-in.

## 2026-09-29 (morning) — KongNet fold mapping resolved empirically: identity; class channels verified
`scripts/kongnet_foldmap.py` (200 imgs/fold linspace-sampled, vendored
`KongNet_Inference_Main` model + decode at repo defaults: ImageNet norm, fp16 autocast, peak
threshold .5 / min_distance 11). Pooled detection F1 (centroid pairing r = 12 px, per image as in
pannuke_eval), ckpt x fold:
| ckpt | fold1 | fold2 | fold3 | -> train fold (split: train/val/test) |
|---|---|---|---|---|
| 1 | **.9051** | .7807 | .7777 | 1 (1/2/3) |
| 2 | .7845 | **.9435** | .7795 | 2 (2/1/3) |
| 3 | .7690 | .7730 | **.9121** | 3 (3/2/1) |
- **`KongNet_PanNuke_k.pth` = official split k** (train fold k; diagonal train-fold F1 margin
  >= .12 over both held-out folds, min .124, clean permutation). Decision rule = argmax train-fold F1
  (memorisation), so the identified test folds' numbers never entered any decision; nothing tuned.
- **Channel map confirmed on the train folds only**: matched-pair confusion (pred x gt) is
  diagonally dominant, class agreement .970/.986/.976; Dead diagonal 71/26/36 with ~0 off-diagonal.
  So heads = [overall] + 5 classes, heatmap = 3rd channel of each head ([5,8,11,14,17] =
  neoplastic..epithelial, official order) exactly as the repo's `cell_channel_map` claims.
- Checkpoints are {epoch, model, optimizer} (~2.1 GB each, 176.9 M params: tf_efficientnetv2_l
  encoder + 6 SMP decoders; epochs 56/87/67). **The released inference pipeline outputs detection
  POINTS only** — no instance masks — so strict-protocol PQ needs the other 2 channels per head
  (their `prediction_utils` suggests seg + contour semantics, `multihead_seg_post_process`) or a
  point-detection evaluation; any decode thresholds must be chosen on VAL folds only.
- Artifact: `runs/analysis/kongnet_foldmap.json`; dependency `segmentation-models-pytorch`
  0.5.0. Probe fixes: type.npy classes are 1..5 (off-by-one vs channel indices); release each
  2.1 GB checkpoint before loading the next to bound memory. Pairing must be per image
  (fold-level Munkres is O(n^3) on 40k points).
- (2026-09-29 audit erratum for the 2026-09-28 evening entry, point 3: the two parentheticals in
  "official -> excluded" are swapped — Uterus Dead PQ is near 0, so excluding Uterus RAISES Dead
  PQ; split-1 seed-19 no-TTA is official .142 / Uterus-excluded .195. The table's per-split
  numbers are the same values, just mislabeled.)

## 2026-09-29 (midday) — B1 (2x resolution) split-1 result: every Dead endpoint positive, but a uniform bPQ tax
Run `runs/cellvit_uni_x2/split1` (seed 19, 130 epochs @ ~326 s = 12.6 h; eval + TTA
complete — the run's untagged `eval_test_fold3[_tta]` ARE the x2 results per the
in-flight-run note). Compare vs 3 baseline seeds (19/1/2) with `scripts/compare_runs.py`
(2000-draw paired image bootstrap; JSONs `runs/analysis/x2_split1_{tta,notta}_vs_base.json`;
Dead views `scripts/x2_dead_views.py`):

| metric (TTA) | base mean (seed std) | x2 s19 | delta [95% CI] |
|---|---|---|---|
| mPQ | .4979 (.0032) | .4800 | -.0180 [-.0238,-.0122] |
| bPQ | .6708 (.0018) | .6447 | **-.0261** [-.0308,-.0219] |
| Dead PQ (official) | .1429 (.0054) | .1614 | **+.0185** [+.0019,+.0364] |

No-TTA is the same story (dDead +.0202 [+.0029,+.0389], dbPQ -.0286).

- **All pre-registered Dead endpoints positive** (TTA; base = mean over seeds 19/1/2):
  Uterus-excluded Dead PQ .198 (.188-.204) -> **.219**; interior Dead missed_bg .352 (.343-.365)
  -> **.251** (-.101, ~10x the seed spread); interior Dead matched share .50 -> .56; Dead F_c
  .310 -> .356; pooled DQ+ .352 -> .371; even Uterus intraluminal clusters .041 -> .055.
- **bPQ non-regression FAILS**: -.0261 = 13x the seed std, negative in ALL 19 tissues (Thyroid
  -.000 .. Pancreatic -.048), and every non-Dead class regresses (Neo -.036, Epi -.021, Con -.020,
  Inf -.011). A pure Dead-for-everything-else trade.
- Interpretation: the -10 pt interior Dead miss is exactly where the 2026-09-28 artifact analysis
  placed the class-specific Dead deficit, so working resolution IS a Dead bottleneck (the
  resolution hypothesis survives on the Dead side). But the tax is uniform, not Dead-adjacent —
  consistent with a decode-side cost (512 -> 256 nearest-downsample of the instance map) rather
  than per-class model damage; one cheap follow-up before believing a model-level trade: re-decode
  variants (e.g. downsample probability maps, not label ids). Per the pre-registered combined
  endpoint (Dead + bPQ non-regression), **x2-as-is is not a win**; the grid continues for the
  science claim only.
- **Follow-up runs launched with fixed code** (tagged `eval_test_fold{t}_x2[_tta]`):
  split2/s19, split3/s19, split1/s1 and split1/s2; outputs
  `runs/cellvit_uni_x2/{split2,split3,split1_seed1,split1_seed2}`. Full 3x3 still needs
  split2/3 seeds 1+2. Baselines exist for all splits (seed 19; extra seeds only on split1).

## 2026-09-29 (evening) — x2 deep-dive: decode-downsample hypothesis REFUTED; bPQ tax = fixed-px decode constants at 512 (large-nucleus over-segmentation + tiny fragments)
Re-analysis of the same four TTA runs from per-nucleus `gt_records`/`pred_records` (join key
image+cumcount; base = 3-seed mean). The midday interpretation ("uniform bPQ tax ~ decode-side
512->256 nearest downsample of the instance map") is **wrong in mechanism**; montages
`runs/analysis/x2_montage_{damage,gain,fragments}.png` (script `scripts/x2_montage.py`,
x2 predictions `pred_test_fold3_tta.npz`).

- **The tax is not colocated with the Dead gain**: per image (97 Dead-bearing), corr(dbPQ, dDead)
  = +.23; median dbPQ = +.003 among dDead>0 images vs **-.027** among dDead<=0. Images that
  gained Dead are bPQ-neutral; the losing images are the ones without Dead.
- **Size, not class, is the axis** (nonDead GT-area bins, match-rate / matched-IoU, x2-base):
  0-200 px **+.104 / +.094** (n=15802); 200-400 -.010/+.001; 400-800 -.020/-.010;
  800-1600 -.029/-.023; 1600-3200 **-.070/-.048**; 3200-6400 -.115/-.070. Dead 0-200 +.091/+.082.
  x2 helps small nuclei everywhere (incl. Dead) and hurts large ones monotonically.
- **Boundary quality of matched nuclei is unchanged**: matched IoU by n_touch 0..4:
  -.004/-.003/+.000/+.005/-.003. The nearest-downsample of label maps costs ~nothing measurable
  (also matched median pred areas ~unchanged) — refutes the decode-downsample story for bPQ.
- **Error composition**: split rate up in every class (Neo .009->.017, Con .013->.025, Dead
  .010->.026); among large (>=1600 px) nonDead: split .072->.104, merged .003->.009, missed_bg
  .024->.054. Pred side: fp_split ~doubles (Neo .018->.037, Con .026->.049, Dead .015->.047),
  fp_bg +40-80%. **#preds/image 24->27; <80 px pred share .034->.109 (3.2x)** — over-segmentation
  plus a small-fragment flood.
- **Mechanism (code-verified)**: `third_party/hover_net/models/hovernet/post_proc.py::process`
  hardcodes pixel-unit constants applied to the 512-res maps: `remove_small_objects(min_size=10)`
  (AREA, skimage semantics -> 2.5 native-px^2 equiv at u=2) on both the np blob and the marker
  map, and `Sobel(ksize=21)` (LENGTH -> 10.5 native-px equiv) on the HV maps. At x2 every
  threshold relaxes in native units -> spurious markers survive and fragment -> large nuclei
  split/merge; tiny blobs pass -> fp_bg/fp_split. Small-nucleus detection gain is real model
  resolution benefit (missed_bg halves; matched Dead IoU +.08).
- **Pre-registered re-decode follow-up (decode-only, no retraining)**: scale px-unit constants
  by u — min_size 10 -> 10*u^2 (=40), ksize 21 -> nearest odd *u (=41) — re-run
  `predict_fold` + eval on the existing x2 checkpoints (split1 ckpt is `final.pth`).
  Prediction: recovers most of bPQ (-.026 -> ~0) while keeping the Dead gains, since boundary
  quality is already at parity and the Dead gain lives in detection, not decode. If it does,
  x2 + scaled decode should be re-judged against the combined endpoint before writing B1 off.
- **Visual verdict (sonnet vision subagent on the three montages, tile-level + re-measured)**:
  confirms all four numeric claims with refinements. (i) Damage visible in 11/12 sampled large
  nuclei, dominant mode = **interior fragmentation** (img 477: one nucleus -> 7-piece mosaic
  with axis-aligned stair-step dividers; outer contour stays on the GT in 12/12, GT coverage
  .866->.871) — marker/watershed seed plateaus on chromatin texture, NOT boundary wander;
  2/12 true merges, 0 misses, 1/12 an IoU-threshold artifact (no visible difference). If
  anything x2 boundaries adhere slightly better (img 275: base's inset/shrinkage disappears).
  (ii) Small-Dead gains 12/12, mean IoU .70, centroid offset <= 1.6 px. (iii) Tiny-FP surplus
  mostly genuine (8/12 x2-only specks on stroma/lumen/chromatin gaps) BUT ~1/3 of sampled
  fragments are **border-clipping slivers** of nuclei both models detect (3/4 of those at
  y=0-2 or x=0 of the 256-px patch) and 1-2 sit on unannotated pyknotic dots (plausible real
  biology) — the 3.2x fragment figure overstates new hallucination somewhat. (iv) No erosion,
  halo, or double-contour artifacts anywhere.
- **Decode-fix menu updated by the visuals** (all decode-only, pre-registered): (1) scale the
  px constants (min_size*u^2, ksize*u) — attacks fragmentation + fragments; (2) post-hoc merge
  of same-class adjacent instances — img 477-style mosaics have clean outer boundaries, so
  merging may convert the damage into a net win; (3) drop edge-touching components (border
  slivers). Also noted: base itself under-covers 25-42% of the GT in 4/12 damage cases —
  "base matched" is a low bar for large nuclei.

## 2026-09-30 (night) — pre-registered decode-fix re-decode CONFIRMED: scaled px constants own ~2/3 of the x2 bPQ tax and ~3/4-9/10 of the mPQ+ loss; Dead gain kept and slightly enlarged
`--decode-u` (predict_cellvit.py) scales `min_size 10 -> 10*u^2` and the Sobel aperture `21 -> odd(21*u)`
(=41). cv2.Sobel caps at 31, so aperture >31 uses the official 21-tap operator's impulse-response kernels
linearly resampled (identity at 21, tested; u=1 path bit-exact, tests/test_decode_scale.py). Artifacts
`eval_fold{f}_x2_du2{,_tta}/`; split1 built-in x2 evals keep the pre-res_tag names
`eval_test_fold3{,_tta}`.
3-split test means (last ckpt), base / x2 / x2+du2:
- TTA:   mPQ .5047/.4848/.4926  bPQ .6706/.6488/.6620  mPQ+ .5227/.5070/.5189  Dead .1768/.1786/.1810
- noTTA: mPQ .4995/.4791/.4894  bPQ .6653/.6413/.6583  mPQ+ .5178/.5012/.5158  Dead .1760/.1816/.1831
Verdict vs the prediction ("recovers most of bPQ -.026 -> ~0, keeps Dead"): bPQ tax recovered 61% (TTA
-.0218 -> -.0086) and 70% (noTTA -.0240 -> -.0071) — most, but not to ~0; mPQ+ nearly closes (-.0158 ->
-.0038 TTA, 76%; -.0166 -> -.0020 noTTA, 88%); mPQ recovers only ~40-50%. Dead gain is kept and slightly
larger after du2 (TTA +.0019 -> +.0042; noTTA +.0056 -> +.0071). Per-split Dead: the x2 gain is fold3-driven
(s1 noTTA +.0171, s2 +.0091) while s3 (fold1) shows a small x2 Dead loss (-.0094) that du2 partly recovers
(-.0032) — consistent with the known fold3/fold1 Dead asymmetry. Combined endpoint: x2+du2 still nets
slightly below base (bPQ -.007..-.009, mPQ -.010..-.012, mPQ+ -.002..-.004), so x2 remains unjustified for
overall metrics; residual suspects = the unscaled 5x5 marker-open kernel and genuine fp_split at 512 —
next decode-menu levers: post-hoc merge of same-class adjacent instances, drop edge-touching slivers.

## 2026-09-30 (morning) — Line A verdict (KongNet under the strict protocol): NEGATIVE — detection-first does not fix Dead; paper Dead F_c .59 does not replicate
Phase-2 line A executed end-to-end in ~35 min (sweeps + 3 full-fold test
predictions on L4 ~5 min; separate CPU strict evals ~12 min). Released per-split checkpoints (`KongNet_PanNuke_k.pth`
= official split k, train-fold F1 margin >= .12 — legal for test, unlike the all-data releases);
6-head decode = per-class seg-minus-contour masks + argmax conflict resolve + per-class CC
(`src/nucseg/kongnet/decode.py`, tests/test_kongnet_decode.py). Protocol: decode thresholds tuned on
VAL folds ONLY (800-img linspace subsets, `light` scoring), grid extended twice until the optimum was
bracketed on both axes — contour hit its 0.8 edge with the gradient unflattened (+.008/step), extended
to 1.0; seg then hit its 0.2 edge, extended to 0.1. Test folds ran once with the frozen constants.
- **Val sweep verdict: the contour channel is NET-HARMFUL at 40x — every checkpoint's optimum disables
  the cut (contour_thr 1.0)**; final constants c1 seg .1 / c2 seg .2 / c3 seg .1, all contour 1.0.
  Val mPQ c1 .3237->.3445 and bPQ .4545->.4731 from the extensions alone. The repo's contour-cut
  recipe (thr .3) over-splits nuclei at this scale.
- **Test 3-split means: mPQ .3460 / bPQ .4706 / Dead PQ .053 / Dead F_c .344** (per-split mPQ/bPQ/DeadPQ/
  F_c: s1 .3480/.4706/.049/.344, s2 .3372/.4640/.044/.277, s3 .3527/.4772/.067/.411). vs dense decoders:
  mPQ 11 pts below the WORST (HoVer-Net .4564, HNXT-T .4579, CellViT-UNI .4995); bPQ 17-19 pts below.
- **Pre-registered Dead endpoints all fail**: official Dead PQ .053 (dense .127-.176); Uterus-excluded
  Dead PQ .070 (dense .152-.218); interior-Dead miss .496 on the shared splits-1+3 frame (CellViT-UNI
  .354, HoVer-Net .413). Only per-nucleus Dead F_c is mid-pack (.344 vs .359/.349/.230).
- **Error profile**: same-class touching nuclei FUSE — merged-GT rate .121-.175 (CellViT-UNI .044,
  HoVer-Net .049); missed_bg .11-.15. With the contour cut disabled the decode is plain per-class CC,
  which cannot separate touching instances; the contour channel was doing (imperfect) separation work
  — cutting hurts individual nuclei more than it helps separation at this scale.
- Interpretation: the 2026-09-28 conclusion strengthens from "shared across the dense decoders tested"
  to "shared across decoder FAMILIES tested" — the detection-first architecture, given its best val
  decode, is WORSE on interior Dead recall and Dead PQ, not better. KongNet's paper Dead F_c .59
  (their decode, their eval) does not transfer to the strict protocol. Line A is closed as a negative
  result; the x2-resolution line (B) remains the live lever, and the merged-heavy KongNet profile is
  consistent with Dead's karyorrhexis-cluster merge failure being an instance-separation problem, not
  a detection-density problem. Artifacts: `runs/kongnet/{sweep_c*.json, pred_*.npz,
  split{k}/eval_test_fold{3,3,1}}`; portable entry point `scripts/kongnet_eval.py`.

## 2026-09-30 late night — batch A/B + lever M/S selection (verdict pending)

Batch-invariance record: `runs/_ab_verdict.md`.

- **Batch-size invariance checked**: preallocation refactor bit-identical (0.00e+00);
  batch 2 vs 32 differs <= 2.82e-05 on all rate metrics (4/~66k instances, cudnn algorithm
  noise). Inference uses batch 2; predictions and CPU evaluation run in separate processes
  to bound peak memory, without changing the evaluation protocol.
- **Lever M/S chosen VAL-only: (frac 0.25, a_min 40) for all splits** (du2 VAL sweeps: bPQ
  rises on all three, mPQ +.0014/+.0012/+.0018; rule bPQ >= baseline-.002, max mPQ). The same
  pair wins the first extended sweep (split1 du2mk2 VAL: .4845/.6486 vs baseline .4836/.6455).
- du2lev TEST (eval_test_fold{3,3,1}_du2lev): mPQ mean .4910, bPQ mean .6608,
  Dead PQ .162/.169/.219, Uterus-excluded Dead PQ mean .2266, interior-Dead miss pooled
  .3041. vs x1 CellViT-UNI (mPQ .4995, bPQ .6653, Utex DeadPQ .2175, interior-Dead miss
  .3738): the levered x2+du2 trades ~.008 mPQ/bPQ for +.009 Uterus-excluded Dead PQ and
  −.070 interior-Dead miss. Pre-registered PASS/FAIL vs un-levered du2 lands when the du2
  test evals finish (not yet computed at this entry).
- Seed grid started: split1 test fold3 seed1 .4825/.6568/DeadPQ .1392, seed2 .4854/.6565/
  .1515 (seed19 = the du2 eval, pending). split1_seed1/seed2 complete; remaining split2/3
  seed runs still pending at this entry.

## 2026-10-01 (early) — Lever VERDICT: PASS, strict test improvement; B1 final standing (3-split, seed 19)

The pending du2 TEST evals (eval_test_fold{3,3,1}_du2) are complete. Verdict script
`scripts/lever_verdict_views.py` (+ `compare_runs.py` paired bootstraps,
`runs/analysis/lever_s{1,2,3}_du2_vs_du2lev.json`).

- **PASS: the VAL-only-selected lever (frac 0.25, a_min 40) is a strict improvement on TEST.**
  du2lev - du2 per split: mPQ +.0020/+ .0016/+.0014, bPQ +.0030/+.0025/+.0021 — all six
  95% paired image-bootstrap CIs exclude 0 on the positive side. Dead PQ unchanged
  (+.0015/+.0005/-.0018, CIs straddle 0); every Dead view within ±.002 (interior-Dead miss
  .3023 -> .3023, Uterus-excluded .2262 -> .2266). npred 27.1 -> 26.6/image — the lever removes
  ~0.5 fragments+slivers per image and nothing else measurable. Selection generalised with zero
  test-fold tuning; all FOUR extended VAL sweeps (plain x2 x3 splits + du2mk2) independently
  pick the same (0.25, 40) — the choice is robust across decode variants.
- **B1 final standing, 3-split means (noTTA, last ckpt, `lever_verdict_views.py` table)**:

| arm | mPQ | bPQ | mPQ+ | DeadPQ | DeadPQ -Uterus | Dead F_c | Dead DQ+ | int-Dead miss | npred |
|---|---|---|---|---|---|---|---|---|---|
| base (x1) | .4995 | .6653 | .5178 | .1760 | .2175 | .359 | .396 | .372 | 25.5 |
| x2+du2 | .4894 | .6583 | .5157 | .1831 | .2262 | .395 | .424 | .302 | 27.1 |
| x2+du2+lever | .4910 | .6608 | .5171 | .1832 | .2266 | .393 | .422 | .302 | 26.6 |

(int-Dead miss = missed_bg+missed_shape among non-border Dead GT, mean of per-split rates;
the 2026-09-30 entry's .3041/.3738 are pooled-count versions of the same quantities. mPQ+ is
at parity; mPQ/bPQ are not, so per the pre-registered combined endpoint **x2 is not a default
win** — it is a priced trade: -.0085 mPQ / -.0045 bPQ buys -.070 interior-Dead miss, +.034
Dead F_c, +.026 Dead DQ+, +.009 Uterus-excluded Dead PQ, +.071 interior-Dead matched share.
[2026-10-01 correction: F_c/DQ+ were first written as the du2-arm deltas +.036/+.028; all
figures in this sentence are now the levered arm, matching the table row above.])
- **The x2 cost is now fully accounted**: decode constants (du2) ~2/3 of the bPQ tax, lever
  M/S ~1/8 more (.6583 -> .6608), residual -.0045 vs base. du2mk2 (marker-open kernel scaled
  too) evaluated on split1 VAL: best (0.25,40) .4845/.6486 vs du2's .4850/.6478 — worse mPQ,
  NOT adopted (the marker-open kernel is not the residual).
- Bookkeeping: the watcher's built-in `eval_fold{TF}_x2_du2` dirs are bit-identical to the
  central `eval_test_fold{TF}_du2` ones (verified on split1: mPQ/bPQ/Dead to 17 digits) — two
  names, one eval; prefer the `eval_test_*` name when citing.
- Grid at 01:12: 5/9 du2 test evals in hand (split1 x3, split2 s19, split3 s19); remaining
  runs pending before the pre-registered 3x3 seed-noise verdict for B1.
- **Grid extended to the levered arm + split1 3x3 complete** (02:00): lever applied with the
  frozen constants to split1_seed1/2 and split3_seed1 du2 preds; central evals written under
  canonical names `eval_test_fold{T}_du2{,lev}` in each seed dir (`scripts/x2_grid_sum.py`
  sums whatever exists). split1 3-seed table (mean±std; base / du2 / du2lev): mPQ
  .4928±.0021 / .4846±.0019 / .4863±.0021; bPQ .6661±.0015 / .6565±.0003 / .6590±.0002;
  Dead PQ .1393±.0074 / .1504±.0107 / .1508±.0112; interior-Dead miss .400±.015 / .348±.008 /
  .348±.008. Paired per-seed du2lev-vs-base: Dead PQ + 3/3 seeds (+.020/+.009/+.006),
  interior-Dead miss - 3/3 (-.045/-.039/-.074), Dead F_c + 3/3; mPQ - 3/3 (-.007/-.007/-.006),
  bPQ - 3/3 (-.008/-.006/-.007). **The Dead-detection gain is seed-robust (3/3, effect ~1.5x
  the seed std); so is the price (~4x std).** split3_seed1 du2 (.4987/.6598/Dead .2415) beats
  its seed19 (.4964/.6620/.2204) — split3 seed spread is the larger one. Lever on the new
  seed runs (split1 s1/s2, split3 s1): dmPQ +.0014/+.0015/+.0021, dbPQ +.0022/+.0023/+.0030,
  Dead +.0005/-.0007/+.0004 — the PASS verdict replicates across seeds (6/6 runs so far
  positive on both mPQ and bPQ over the 3 splits' available seeds).

## 2026-10-01 (morning) — LKCell-L strict-protocol re-eval: paper gap is uniform ~-.015; better segmenter, worse typer; no Dead gain

First entry of the A1 external pool (survey §A): released LKCell-L per-fold checkpoints
(HF `xiazhi/LKCell-L`; UniRepLKNet-S + RepLK decoder + CellViT np/hv/tp heads; `model_best`
= val-selected, like HoVer-NeXt), predicted through the shared pipeline — LKCell norm
(mean/std .5), official decode at 256, no TTA — and evaluated with
`nucseg.metrics.pannuke_eval` on all 3 folds (`runs/lkcell/split{1,2,3}/eval_fold{3,3,1}`).

| arm | mPQ | bPQ | mPQ+ | DeadPQ | DeadPQ -Uterus | Dead F_c | Dead DQ+ | int-Dead miss |
|---|---|---|---|---|---|---|---|---|
| LKCell-L strict (ours) | .4923 | .6729 | .5235 | .1542 | .2010 | .382 | .418 | .347 |
| CellViT-UNI base (seed19) | .4995 | .6653 | .5178 | .1760 | .2175 | .359 | .396 | .372 |
| x2+du2+lever (seed19) | .4910 | .6608 | .5171 | .1832 | .2266 | .393 | .422 | .302 |
| LKCell-L paper | .508 | .685 | — | .172 | — | — | — | — |

Per-split strict: mPQ .4857/.4855/.5057, Dead PQ .1150/.1415/.2062 (splits 1/2/3).

- **Uniform ~-.015 vs the paper on every headline metric** (mPQ -.016, bPQ -.012, Dead -.018)
  — a protocol translation, not a broken pipeline (a fold/norm error craters metrics, it does
  not shift them all by the same amount). No TTA applied; checkpoint = their released
  `model_best`. Cannot be resolved further from the paper alone; recorded as-is for the
  architecture table.
- **Under matched conditions LKCell-L does not beat our base where it counts**: mPQ -.0072,
  Dead PQ -.0218 vs CellViT-UNI, despite bPQ +.0076 and mPQ+ +.0057. Decomposition: LKCell is
  the better *segmenter* (mPQ+/bPQ up; Dead F_c .382 vs .359 — it FINDS Dead nuclei better)
  but a worse *typer* — its detected Dead nuclei still fail class-PQ pairing (type/segmentation
  of matched pairs), so better detection does not become better Dead PQ. Same direction as the
  KongNet verdict (line A): general architecture strength does not transfer into Dead or
  typing gains under a matched protocol. Even our costliest arm (x2+du2+lever) is at mPQ parity
  with strict LKCell (.4910 vs .4923) while winning every Dead endpoint.
- Pillar-C rank pool: LKCell joins with full Dead endpoints (n=4 architectures).
- Sanity: predicted density tracks GT block-wise (fold3 opens with a sparse tissue: GT 12.9
  vs pred 13 instances/img on the first 40 patches); per-split Dead ordering (3 >> 2 > 1)
  matches the base's.
- Reproduction note: vendored `CellViT()` constructs blocks on GPU (`.to(device)` in
  `__init__`), so allow device memory for model construction as well as inference.
  This evaluation used `--batch-size 3` (~2 min for 2722 patches on L4).
- Grid 7/9 (04:55): split2_seed1 du2+lever collected (lever dmPQ +.0012, dbPQ +.0022,
  Dead +.0007 — the PASS verdict is now 7/7 runs); `x2_grid_sum.py` ARMS globbed so remaining
  seed evals are picked up automatically. split2_seed2 and split3_seed2
  still pending for the pre-registered 3x3 seed-noise verdict.

## 2026-10-01 (midday) — grid 8/9: split2 complete; lever PASS replicates 8/8; docs/artifact audit clean

- split2_seed2 predictions and x2/x2-TTA evaluations collected;
  canonical `eval_test_fold3_du2` recomputed independently —
  **bit-identical to the watcher's `eval_fold3_x2_du2`** (first verified on split1 seed19,
  now on a second chain). Lever applied with the frozen constants (0.25, 40) ->
  `eval_test_fold3_du2lev`.
- split2_seed2 lever: dmPQ +.0011, dbPQ +.0021, Dead +.0004 — the PASS verdict is now
  **8/8 runs** (original 3 splits + every subsequent seed chain). split2 3-seed means
  (du2 / du2lev): mPQ .4847±.0009 / .4860±.0011, bPQ .6568±.0005 / .6591±.0005,
  Dead PQ .1606±.0068 / .1611±.0068, int-Dead miss .3395 / .3391 — split2's seed spread is
  the small one, like split1's bPQ; Dead stays the noisiest endpoint everywhere.
- 8/9 3-split means barely move vs the seed19-only table (du2lev vs base: mPQ -.0082,
  bPQ -.0043, Dead +.0057, int-Dead miss -.070) — the B1 trade prices are stable under
  seed pooling. Remaining: split3_seed2 training,
  after which the pre-registered 3x3 seed-noise verdict is computed from `x2_grid_sum.py`.
- Audit (this session): `pytest` 49/49 in the nuclei env; the 2026-10-01 B1 standing table,
  split1 3-seed table, lever PASS CIs (all six mPQ/bPQ CIs > 0, Dead straddling 0), the
  7/7-run lever replication numbers, and the LKCell table (incl. exact 3-fold bPQ mean
  .672947 -> .6729, DeadPQ-Uterus .2010, int-Dead miss .347) all re-derived from artifacts
  and matched docs to 4 digits.

## 2026-10-01 (evening) — grid 9/9: 3x3 seed grid complete; pre-registered seed-noise verdict

- split3_seed2 predictions and evaluations collected; canonical
  `eval_test_fold1_du2` recomputed independently — **bit-identical to the watcher's
  `eval_fold1_x2_du2`** (summary.json equal; per_image arrays equal under `equal_nan`; third
  chain verified after split1 s19 and split2_seed2). split3_seed2 du2: .4973/.6596,
  Dead PQ .2241 F_c .5025 — between its siblings (s19 .4964/.6620/.2204, s1 .4987/.6598/.2415).
- split3_seed2 lever (frozen 0.25/40): dmPQ +.0017, dbPQ +.0024, Dead +.0000 — **lever PASS
  9/9 runs** (every chain that landed).
- Full 3x3 du2 per-split seed means (mean±std over seeds {19,1,2}, from `scripts/x2_grid_sum.py`):
  split1 mPQ .4846±.0019 bPQ .6565±.0003 Dead .1504±.0107; split2 .4847±.0009 .6568±.0005
  .1606±.0068; split3 .4975±.0012 .6605±.0013 .2287±.0112 (s1 .2415 is the grid's Dead high).
  3-split means: du2 mPQ .4889 bPQ .6579 Dead .1799 F_c .3938 DQ+ .4220 int-Dead miss .3068;
  du2lev .4905/.6603/.1801 miss .3070. vs the seed19-only table the 3-split means barely move
  (du2lev-vs-base: mPQ -.0082 -> -.0083, bPQ -.0043 -> -.0045, Dead +.0057 -> +.0051,
  int-Dead miss -.070 -> -.069).
- **Seed-noise verdict (pre-registered)**: the trade prices are seed-robust — dmPQ -.0098 and
  dbPQ -.0069 (du2 vs x1 base) are 4.7x/5.3x the largest per-split seed std (.0021/.0013),
  same sign on every run. The Dead gain is the fragile endpoint: +.0049 on 3-split means is
  only 0.4-0.7x the per-seed Dead std (.0068-.0112); paired seeds 4/5 favor du2 (split3 s19
  flips, -.0032) and split seed-means go +.0111 (split1, 3/3 seeds) / -.0015 (split2, flip)
  / +.0051 (split3, 2/3 seeds). x1 stays ahead on mPQ/bPQ at ~5x seed noise; x2's Dead
  advantage is real on split1, present-but-noisy on split3, a wash on split2.

## 2026-10-01 (night) — Stage report II built (EN+ZH+digest), all numbers re-derived first

- docs/report/stage_report_2/: stage_report_2.tex (EN, 12pp, 7 tables 5 figs),
  stage_report_2_zh.tex (10pp), summary_plain_zh.tex (4pp digest). Compiled with
  TeXLive 2026 (pdflatex EN / xelatex ZH, two passes each, 0 errors 0 overfull after
  fixing two overfull boxes: an unbreakable \texttt path in ZH, the digest footer line);
  PDFs included. figs/ self-contained (6 charts EN+ZH + 3 montages copied).
- Every table number was re-derived from artifacts BEFORE writing
  (scripts/report2_numbers.py -> runs/analysis/report2_numbers.txt): architecture table
  (KongNet .3460/.4706/Dead .0534/F_c .344; HoVer-Net row now with full Dead endpoints
  .1268/.1517/.230/.275/imiss .404; LKCell-L .4923/.6729/.1542/.2010/.382/.418/.347),
  du2 recovery (bPQ 61% TTA / 71% noTTA, mPQ+ 76/87%, mPQ 39/50%), lever CIs (6/6
  mPQ/bPQ positive, Dead straddle), 3x3 grid per-split means+-std, B1-initial Dead
  endpoint battery (Uterus Dead PQ .041->.055 on the 34 images; interior missed_bg
  .352->.251), size-binned mechanism (+.104 ... -.115; Dead 0-200 +.091), error
  composition, npred 25.5->28.4 (TTA, split1), <80px share .034->.109, KongNet
  pooled interior-Dead miss .496 (shared s1+s3 frame), foldmap diagonal .905/.944/.912.
  Two findings-entry errata found by the re-derivation and corrected IN THE REPORT
  (docs entries stand): npred "24->27" (09-29 evening) should be 25.5->28.4 TTA
  split-1; findings' size-bin "matched-IoU" column was actually mean best-IoU over ALL
  bin nuclei (report uses both, labeled). Also note base TTA Dead 3-split mean is
  .1767 (findings wrote .1768 once).
- Required eval directories collected into the runs tree for the tables:
  split2/eval_test_fold3_x2{,_tta}, split3/eval_test_fold1_x2{,_tta} (the x2 pre-du2
  test evals; split1's stay untagged), and the du2-TTA evals under their watcher names
  eval_fold{3,3,1}_x2_du2_tta. Bibliography verified against arXiv metadata
  (KongNet = Lv et al., arXiv:2510.23559; LKCell = Cui et al., arXiv:2407.18054).

## 2026-10-03 — Typing-confidence audit (P0-P3 §7 follow-up): confidence gates typing, not existence

- Descriptive audit of predicted-class confidence vs typing consistency on the cached P0-P3 v2
  predictions; no thresholds selected, no training. New code: `src/nucseg/postproc/typing_audit.py`,
  `scripts/analyze_typing_confidence.py`, `scripts/typing_audit_numbers.py`,
  `tests/test_typing_confidence.py` (9 tests; full CPU suite 137 passed / 3 CUDA-skipped).
  Artifacts + exact commands: `runs/analysis/typing_audit_20261003/` (per-split JSONs with input
  hashes, `summary.json/md`, `commands.md`).
- The three x1 **test** folds were re-predicted on ugradx (L4, batch 2, no TTA, ~8 GPU-min total)
  to obtain per-instance probability tables (val folds already had tables). Sanity vs the cached
  maps: 0.05-0.09% of instances per side remain unpaired after greedy matching at IoU>0.
  Among those greedy pairs, 95.8-96.0% exceed IoU 0.5; type agreement on IoU>0.9 pairs is
  99.96-99.98%. These checks do not establish distributional equivalence or bit identity; all headline P0-P3 numbers remain
  bound to the cached artifacts.
- Replay fidelity: re-applying the frozen P2 rule reproduces the val selections exactly
  (split2 added 1713, split3 2604 = the recorded `added` counts).
- **Typing accuracy among matched** (3-split mean): x1 .8640 vs x2 .8617 (test). **Dead:
  x1 .5539 vs x2 .5788 (test), .5735 vs .6144 (val)** — du2 types matched Dead BETTER than x1.
  The P1 type-swap advantage of x1 is therefore not located in Dead typing nor in the stable-pair
  disagreement population: agree rate 93.2%, and on disagreements base is right 1101 vs x2 right
  1038 (test, three-split means of counts, not pooled counts; both wrong 729) — near-symmetric. x2 confidence is slightly higher when x2 is
  right (.765) than when base is right (.735): a weak arbitration signal, not deployable alone.
- **Map-majority vs mean-prob argmax disagreement** (test three-split means: 1.23% x1 / 1.55% x2 of instances): typing
  joint match-and-correct-type rate is .16-.20 (vs .66-.69 when consistent; denominator includes unmatched predictions) on both scales — a free wrongness
  marker (abstention/flag candidate for the matched-seed round).
- **P2 added candidates (test)**: match rate .39-.43, typing acc given matched .86-.88; added
  Dead acc .821 (split3, p>=.7) but **.667 on split2 (p>=.9) vs .850 on its own val** — the
  split2 Dead typing quality did not transfer val->test. Rejected near-threshold candidates:
  match .32-.34, acc .75-.77, Dead .4375-.4444 (test only; .60 was split3 validation). Confidence orders typing risk (sharpest for Dead),
  but the dominant cost is **existence**: 57-61% of additions match no GT instance (pure FPs).
- Conventions: the anchor uses raw-du2 candidates vs raw x1 (P2's convention; P1's candidate
  table counted M/S'd-x2 candidates — different population by design, 3879 vs 3304 on split1 val).
- **Matched-seed round candidates (recorded, not tested here)**: an existence-side filter for
  additions (the frontier is detection precision, not type precision); map-vs-row consistency
  abstention; per-class Dead thresholds. Per the frozen-v2 rule, no new menus were run on test.

## 2026-10-05 — Matched-seed verdict (pre-registered rule): P2 is NOT a robust improvement; the Dead gain is real and seed-robust, the bPQ price is not free

- Round `runs/analysis/matched_seed_20261004/`: 18/18 val selections frozen before any test-fold
  read; 9 (split, seed) matched pairs trained/evaluated to the frozen protocol (5 new runs from
  2026-10-03/04, incl. LM2 cross-machine placements). P0 M/S uses the same v2 menu, selected separately per pair; choices vary by seed
  (e.g. base split2_seed1 selects .5/40, vs .25/40 for split2_seed19); P2 val: 8/9 GO, only split1_seed19 NO_GO (identity) — same as v2's split1.
- Incident + fix (commit da3b934): `validate_run_context` hardcoded seed19 and aborted every
  non-seed19 `p2_val` on entry (lanes died 2026-10-04 22:15, watcher timed out 00:01). The check
  now binds both arms to the declared pair seed (`run_scale_fusion --expect-seed`); the
  split1_seed19 P2 redo reproduced the archived v1 selection **bit-exactly** (same sweep rows,
  NO_GO/identity), proving the fix is validation-only.
- **Pre-registered decision (scripts/matched_seed_stats.py, stats/stats.json):**
  - ΔDead PQ 3-split seed-mean **+.00742** ≥ 1× largest per-split seed std (.00312) — PASS
  - Dead sign count **9/9 pairs ≥ 0** (≥ 7/9 required) — PASS
  - ΔbPQ **−.00087** < −1× its largest per-split seed std (.00064) — **FAIL** → verdict:
    **robust_improvement = false**; per the frozen rule P2 stays a conditional candidate
    (no new menus, no test-driven re-selection, no endpoint substitution).
- Per-split deltas (P2 − x1+M/S baseline, seed-mean±std; tissue-stratified paired-image
  bootstrap 2000, seed 20261004): split1 Dead **+.0039±.0030**, CI [+0.0005,+0.0078] (excl. 0);
  split2 Dead +.0020±.0016, CI [−0.0009,+0.0053] (straddles); split3 Dead **+.0163±.0031**,
  CI [+0.0062,+0.0283] (excl. 0) with bPQ −.0018, CI [−0.0031,−0.0006] (excl. 0). mPQ CIs all
  straddle 0. The Dead gain concentrates in split3 and its price is split3's bPQ.
- Interpretation vs the 2026-10-01 3×3 grid: raw du2's Dead endpoint was seed-fragile (4/5
  pairs, 0.4–0.7× seed std); the gated P2 additions turn it seed-robust (9/9 pairs, Δ ≈ 2.38×
  max within-split SD of paired deltas) but do not remove the bPQ tax — the nine split/seed-pair evidence prices P2 as a
  **Dead-for-bPQ trade**, not a free win. M/S secondary (same pairing): ΔmPQ +.00028,
  ΔbPQ +.00030, ΔDead +.00046 — small positive overall means, with 7/9 nonnegative
  pairs for each of these endpoints, not universal per-pair or per-split improvements.
- Bootstrap bug found & fixed during verification: the first implementation
  resampled *positions within a tissue* instead of absolute image indices, shifting CIs off the
  point estimates (caught because CIs excluded their own points); regression test uses unequal
  tissue sizes/levels (constants cannot catch index bugs). Buggy outputs archived in
  `stats_v1_indexbug/`; decision-rule inputs (summary-level) were unaffected.
- CPU suite after both fixes: **142 passed / 3 CUDA-skipped**.

## 2026-10-05 (later) — EF-P2 round: the existence filter clears the bPQ guard; P2 becomes a robust improvement under the pre-registered rule

Round `runs/analysis/existence_ef_20261005/` (plan `docs/superpowers/plans/2026-10-05-existence-filter-p2.md`,
menu frozen BEFORE any EF sweep; CPU only — no new training or inference, all inputs are the
matched-seed round's frozen artifacts).

- **Val-side existence audit first** (`scripts/audit_existence_candidates.py`,
  `runs/analysis/existence_audit_20261005/`): 17,359 frozen-P2 val additions over the 9 pairs,
  41.50% (7204/17359) match a GT instance. Area identifies a low-precision small-object slice (<30 px: 20% matched, n=2294;
  30–60: 44%; 60–100: 49%); finite distance to the nearest base instance has similar match rates (39–44% per bin),
  which does not support using proximity alone as a filter or exclude all fragmentation mechanisms.
  Circularity/extent also show marginal separation, without establishing independent predictive value; additions never re-detect a base-matched GT (0/17359). Audit correctness: reproduced the
  frozen stage-1 added counts exactly (split2_seed19 1713, split3_seed19 2604) after fixing an
  empty-base-image skip (110 zero-base val images contribute candidates).
- **Frozen menu**: {identity, off (=frozen P2 unchanged), a30 (area ≥30 px), a30d (Dead exempt),
  a60}; P2 gate + selection rule and the matched-seed decision rule verbatim. The runner
  (`scripts/run_existence_ef.py`) hard-asserts its `off` row reproduces the frozen stage-1 row —
  held bit-exact on all 9 pairs.
- **Val selections (9/9 frozen before any test read)**: 7× a30, 1× a30d (split1_seed1), 1×
  identity (split1_seed19 — its P2 was already identity).
- **Pre-registered verdict (`stats/stats.json`): robust_improvement = TRUE** — all arms pass:
  ΔDead PQ +.00699 ≥ 1× max per-split seed std .00337 (2.1×) with 9/9 pairs ≥ 0; ΔmPQ +.00068
  and ΔbPQ −.0000091 both ≥ −1× their stds (.00073/.00028). The mean bPQ loss is much smaller than P2
  (−.00087), not proven absent;
  94% of the Dead gain survives (+.00742 → +.00699). strict mPQ −.00104 → −.00030; mPQ+
  +.0031 → +.0038.
- Per-split seed means (EF−baseline): split1 Dead +.00426 (CI [+0.00096,+0.00796] excl. 0),
  split2 +.00150 (straddles), split3 +.01520 (CI [+0.00572,+0.02625] excl. 0); split3 bPQ now
  −.00040 (CI straddles 0; was −.0018 excl. 0 under plain P2). mPQ CIs straddle everywhere;
  sign counts mPQ 9/9, bPQ 5/9, Dead 9/9.
- **Attribution (EF − plain P2, per pair)**: bPQ improves on all 8 non-identity pairs
  (+.0005…+.0019), mPQ likewise (+.0003…+.0016); Dead cost is small and mixed (−.0023…+.0010,
  seed-mean ≈ −.0004) — the audit-predicted trade (sub-30 px additions are 80% FP but hold 43
  Dead-typed val TPs).
- Verification: raw-summary deltas match recorded stats to <1e-12 on 3 spot-checked pairs; the
  decision rule re-derived independently; the identity pair's EF eval is byte-identical to its
  P0 baseline eval; CPU suite **150 passed / 3 CUDA-skipped**.
- Interpretation: the validation-selected existence filter passes the frozen decision rule,
  with a Dead-centred improvement and approximately unchanged mean bPQ. This is not a
  statistical demonstration of non-inferiority, nor a computationally free improvement.
  Honest caveats: (i) the margin is thin (mPQ/bPQ CIs straddle 0; the win is Dead-centred);
  (ii) this is the THIRD pre-registered read of these test folds (v2, matched-seed, EF) — within-
  round selection is val-only, but cross-round test reuse is a multiplicity caveat on any single
  round's significance; (iii) EF-P2 is a post-processing arm on top of x1+M/S+P2, not a new
  model. Both scales require inference; deployment cost has not been measured.
  Nonnegative counts include the identity pair (eight positive Dead deltas and one zero);
  the SD uses ddof=0 over three within-split paired deltas, not baseline-run SD.
  Bootstrap intervals hold the three trained seeds fixed and do not include training uncertainty.
  Per the frozen plan, EF-P2 replaces P2 as the reported candidate arm.


## 2026-10-05 — Follow-up code and documentation corrections

- Fixed EF test application ignoring an EF `identity` choice when stage 1 was nonidentity.
  A regression first reproduced one unwanted addition instead of zero. None of the nine
  archived EF pairs exercises that combination: the only EF identity pair already had
  stage-1 identity, so its archived scores remain valid. The fix changes source fingerprints;
  do not edit old selection manifests or bypass provenance guards to reuse them with new code.
- Fixed the existence audit's border table: it previously selected only area [0,1), making
  both border rows zero for nonempty masks. Future outputs include all areas and retain the
  per-candidate `border` flag in records.npz. The archived audit's `added_by_border` table is
  invalid and must not be used; its records lack border flags. Corrected counts have now
  been regenerated from the frozen validation predictions in `existence_audit_20261005_v2` (below). Other archived bins and the EF menu do not use
  that table, so no published EF scores change. A serialization regression covers both
  border groups, including one-pixel and larger candidates.
- Rechecked all 54 endpoint deltas against raw eval summaries in each of the matched-seed
  and EF rounds (108 total), and all nine EF off rows against frozen P2 rows (tolerance 1e-12).
  The identity pair's summary.json is byte-identical to its baseline summary.
- Corrected the val addition match rate to 7204/17359 = 41.50%; the sub-30px slice contains
  454/7204 = 6.3% of all matches and 43/699 correctly typed Dead matches. Also corrected the
  class table (Inflammatory, not Dead, has the highest match rate; Neo+Con supply 49.5% of matches).
- Typing audit: re-prediction sanity uses greedy IoU>0 pairing, not IoU>0.5; argmax-disagreement
  success rates include unmatched predictions in the denominator. Test-only rejected Dead
  typing accuracy is .4375-.4444. Count averages are labelled as averages, not pooled counts.
- M/S choices vary by seed and its three headline deltas each have 7/9 nonnegative pairs.
  EF-P2 still passes the original rule, but that is not proof of non-inferiority or zero cost.
- Verification after these fixes: **153 passed, 3 CUDA-skipped, 62 dependency warnings**
  (`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q`,
  pinned nuclei environment). No GPU tests, training or new threshold sweeps were run.


### Regenerated border bins (validation only; post-execution descriptive audit)

Source: `runs/analysis/existence_audit_20261005_v2/summary.json`, table
`added_by_border`; the old directory remains unchanged. Border means the candidate
mask touches at least one edge of its native 256-pixel prediction map. Counts pool
the nine split/seed pairs and are not counts of independent patients or images.

| Candidate | Added | IoU>0.5 matched | Match rate | Matched and correctly typed | Correctly typed Dead | GT already matched by base |
|---|---:|---:|---:|---:|---:|---:|
| Border-touching | 9975 | 5273 | 52.86% | 4553 | 85 | 0 |
| Interior | 7384 | 1931 | 26.15% | 1649 | 614 | 0 |
| Total | 17359 | 7204 | 41.50% | 6202 | 699 | 0 |

The regenerated table does **not** support treating border contact alone as a reason
to discard additions: the marginal match rate is higher at the border, while most
correctly typed Dead additions are interior. Class, area and fold composition differ
between these groups; this is descriptive, not a causal effect or a newly validated
filter. No EF menu, selection or test evaluation was changed on this evidence.

Verification (`verification.json` in the new directory):
- All 37,915 candidates retain exactly the same values in every legacy records column
  (including NaNs); all non-border summary tables are unchanged.
- All nine per-pair addition counts equal the frozen P2 selections; hashes of both
  input predictions per pair match the frozen selection records.
- Independently read each candidate ID against the four edges of the saved prediction
  map, without using `candidate_geometry`: all 37,915 border flags agree.
- Recomputed both table rows from records and reconciled every count with area-bin
  totals: 17,359 additions, 7,204 matches, 6,202 correctly typed matches, 699 correctly
  typed Dead matches, zero already-base-matched GT.
- Full CPU suite rerun: **153 passed, 3 CUDA-skipped, 62 dependency warnings**.

Regeneration command (output must not already exist):
```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/audit_existence_candidates.py \
  --round runs/analysis/matched_seed_20261004 \
  --out runs/analysis/existence_audit_20261005_v2 --workers 4
```

## 2026-10-07 — E2a HV-threshold controls complete: the support cutoff matters only at x2; scale keeps most of the Dead gain and all of the bPQ tax

Round `runs/hv_threshold_controls_20261006/` (design and contrasts frozen before launch in
[the plan](superpowers/plans/2026-10-06-hv-threshold-controls.md); full tables, diagnostics and
limitations: [HV controls results](results/HV_CONTROLS_RESULTS_2026-10-07.md)). Split1, train fold1 /
validation fold2 only; no fold3 inference at any point.

- **8/8 runs complete** (x1_hv30 / x1_hv8 / x2_hv30 / x2_hv120 × seeds 19, 1; 130 epochs each,
  no duplicate epoch records, source frozen at `c6e7e92`). Every run has a hash-bound audit
  re-inference (final checkpoint, no TTA, `decode-u=upscale`, `marker-u1`, batch8); the
  audit−original difference is exactly 0 on all 8×6 endpoint cells, so nothing was cherry-picked
  between the two evaluations. Total training-loop time in the logs: 41.76 h on shared A100s —
  not a controlled throughput benchmark.
- **Arm means** (2-seed, validation fold2; PQ units):

  | arm | mPQ | bPQ | Dead PQ | strict Dead PQ |
  |---|---:|---:|---:|---:|
  | x1_hv30 | .482109 | .651972 | .162139 | .126295 |
  | x1_hv8 | .480794 | .650671 | .163580 | .130169 |
  | x2_hv30 | .481307 | .644702 | .177168 | .123403 |
  | x2_hv120 | .479802 | .644481 | .167352 | .118987 |

- **Pre-registered contrasts** (mean; brackets = conditional tissue-stratified paired-image
  bootstrap 95% CI, 2000 draws, seed 20261006 — image sampling only, not seed/patient
  significance):

  | contrast | ΔDead PQ | ΔbPQ |
  |---|---:|---:|
  | x1_hv8 − x1_hv30 | +.001441 [−.010275,+.013596] | −.001301 [−.002950,+.000392] |
  | x2_hv30 − x2_hv120 | +.009817 [−.001653,+.023967] | +.000221 [−.001108,+.001681] |
  | x2_hv120 − x1_hv30 | +.005212 [−.012984,+.026086] | −.007492 [−.010332,−.004394] |
  | x2_hv30 − x1_hv8 | +.013588 [−.010212,+.042515] | −.005969 [−.008959,−.002893] |

- **Verdicts.**
  1. *Lowering the cutoff at x1 is not a usable substitute for x2*: ΔDead flips sign across
     seeds (−.001378/+.004260) and bPQ drops in both seeds.
  2. *The cutoff has a real effect at x2*: hv30 beats hv120 on official Dead PQ in both seeds
     (+.010830/+.008803), but the CI crosses zero and strict Dead is mixed — a candidate
     signal, not a robust improvement claim.
  3. *Matched-native-support scale contrasts keep the bPQ tax*: both x2−x1 contrasts have
     bPQ CIs entirely below zero and both seeds negative; official Dead stays positive on
     average while **strict Dead is negative in all four x2−x1 seed pairs**. Official and
     strict endpoints must be reported together.
  4. *Object level*: interior-Dead matched rate rises 61.3%→71.0% (border 40.2%→48.3%) with
     scale at matched support, while lowering the cutoff at x2 moves the interior matched
     count from 526 to 526.5 — the +.0098 support effect on Dead PQ is **not** explained by
     recovering many more interior objects (mask/FP structure instead, not decomposed here).
- **DSB Gate A arithmetic (pre-frozen rule):** ΔDead(x1_hv8−x1_hv30) = +.00144 is **9.59%**
  of ΔDead(x2_hv30−x1_hv30) = +.01503 — below the 50% switch line and the 90%
  full-explanation demotion line. **DSB/C0/C1 base stays x1_hv30; the architecture line is
  not demoted.** This is point-estimates plugged into the pre-registered rule, not a causal
  mediation estimate, and Gates B/C/D remain open.
- **Decisions:** default `hv_min_size=30` unchanged; x2_hv30 kept as a frozen mechanism
  reference; only the HV branch is published in this round — the in-development DSB branch
  is not merged and no DSB GPU work is started.
- **Verification:** collector re-run on the frozen analysis source (`39110dd`) reproduced
  `verified_numbers.json` byte-identically; an independent re-derivation of all 4 arm means
  and 4 contrast means from the audit `summary.json` files matches bit-exactly; HV-branch CPU
  suite **210 passed / 3 CUDA-dependent skipped**.
## 2026-10-08 — DSB pre-training gates: A/C/D land (proceed/proceed/pass); gate B first run VOID on probe sampling, not a real fail

- Context: dead-specialist branch (DSB) CPU implementation complete on `research/dead-specialist-branch`
  (Tasks 1–11 TDD commits; full local suite 253 passed on 2026-10-08, CUDA cases included). Gates ledger:
  `runs/analysis/dsb_gates_20261007/` (local, outside Git). No GPU training arm started; test fold not read.
- **Gate A (rule arithmetic on frozen E2a numbers): proceed, base stays x1_hv30.** r/R = 9.59% < 50%
  switch line, < 90% downgrade line, bPQ tax −.0013. Independently re-derived bit-exactly from
  `hv_threshold_controls_20261007_verified.json` (all six eval summaries SHA-matched).
- **Gate C (decision decoupling, fold 2): proceed.** tau_dead menu {.5,.4,.35,.3} peaks at dDead
  +.0021 (tau .4), d_bPQ +1.1e-5 — below the +.005 close-line, so architecture decoupling is not
  obviated. Baseline Dead PQ .16386 bit-exact vs `eval_fold2`.
- **Gate B (gradient conflict): 00:58 run VOID, not fail.** Sequential first-256 fold-1 images are all
  Breast with exactly ONE Dead-positive image (index 178); 63/64 batches contribute structural
  zero-gradients counted as non-conflict, capping conflict_fraction at 1/64 ≈ .016 < .30 before any
  gradient geometry is measured. The single informative batch shows positive cosines (decoder
  composite +.27). Probe mechanics verified correct (disjoint masks, one forward, retain_graph;
  39d4c66 did not change the parameter set — concat-equivalent list cosine); the flaw is the
  pre-frozen plan's "fixed order" sampling plus zero-batch accounting. Spec §10 stop rule NOT fired.
  Amendment recorded in the ledger (Dead-stratified sampling with fixed seed; conflict_fraction over
  L_dead-active batches with n_active recorded; optional per-layer significance clause); rerun
  ~1 GPU·h — pending user decision.
- **Gate D (oracle ceiling, fold 2, seed 20261007): pass.** Inserting all 267 unmatched internal Dead
  GT instances at IoU .7: Dead PQ .1639 → .3199 (**ΔDead +.156**; exact-mask variant +.245), bPQ
  +.0015, mPQ +.0023 — far above the +.010 bar. The ceiling is high: detection, not type quality, is
  the binding constraint, consistent with the detection-first diagnosis.
- State after this round: gates A/C/D favorable; gate B awaits an amended rerun (user checkpoint
  before any GPU arm per plan Task 13).
- Verification: all gate numbers re-derived from gate JSONs and eval artifacts bit-exactly; full
  suite 253 passed; `git diff --check` clean.

## 2026-10-08 (later) — Gate B amended rerun: a real measurement, and it fails — DSB closes under the pre-registered stop rule

- User go-ahead received 2026-10-08 for the amended gate B rerun (~minutes of GPU, not the
  estimated hour). Amendment pre-registered in the gates ledger BEFORE the run (commit 9009a21;
  4 new TDD tests watched RED then GREEN; full CPU suite 254 passed / 3 CUDA-skipped):
  Dead-stratified seeded sampling over the 65 Dead-positive fold-1 images (all covered each
  pass), conflict fractions over L_dead-active batches with n_active recorded (VOID < 16),
  and the spec §6 B per-layer clause implemented (one-sided one-sample t-tests, Holm across the
  six module groups). Per-batch cosines serialized in the output JSON.
- **Result (gate_b_v2.json, 64/64 Dead-active batches, all 65 images used):** decoder-composite
  gradient cosine mean **+0.749**, per-batch min +0.594 — **0/64 batches negative** (conflict
  fraction 0.000 vs the 0.30 bar; P(0/64 | true rate ≥ .30) ≈ 1e-10). Every module group's mean
  cosine is positive: encoder +0.406, skips +0.608, np_branch +0.917, hv_branch +0.760,
  tp_branch +0.196 (tp is the only group with any negative batches: 7/64, t = +12.3). Per-layer
  clause: all t > 0 → p = 1, none significant. **Verdict: fail** — the VOID run's lone
  informative batch (+0.27 decoder) was not an outlier.
- **Interpretation:** H1 (capacity-level Dead-vs-common gradient conflict at a trained
  same-recipe checkpoint) is not supported — Dead-pixel and common-class gradients are strongly
  aligned, i.e. improving Dead pixels moves shared weights in roughly the same direction as
  improving common-class pixels. This coheres with the training-side negative results (M1 pixel
  weighting ✗, C1 balanced sampling ✗ — nothing to re-balance) and with the detection-first
  diagnosis (gates C/D: decision decoupling buys +.002; the oracle ceiling +.156 is detection
  headroom, not type headroom). Standing caveats: measured at the trained optimum (early-
  training conflicts could have resolved during training; the probe cannot exclude this);
  fallback same-recipe seed-19 sibling checkpoint; equal-weight per-group losses (directions,
  scale-free).
- **Decision per spec §10 (pre-registered):** any gate failure ⇒ the line closes without GPU
  escalation. Gate B failed ⇒ **no smoke run, no dev round, no confirmatory round**; DSB ends
  here with its mechanism evidence retained: A proceed/base x1_hv30 (support cutoff explains
  9.59% of the x2 Dead gain), B no gradient conflict (this entry), C decision decoupling
  insufficient (+.0021 < +.005), D oracle ceiling high (+.156). The four gates together say:
  the Dead deficit is a detection-recall problem that neither re-weighting, nor threshold
  decoupling, nor architectural specialist decoupling at the trained optimum addresses.
- Verification: verdict re-derived from the JSON (fraction 0/64 < .30, no significant group,
  n_active 64 ≥ 16 → fail); the 00:58 VOID artifact retained unchanged alongside gate_b_v2.json.

## 2026-10-08 (cost round) — Reduced E0: the EF-P2 two-model system costs ~6.1x single-model x1 inference; not a deployment win

Round `runs/analysis/inference_cost_20261008/` (script `scripts/inference_cost.py`, TDD; full
commands, attempt log and caveats in the round's commands.md). Fixed protocol: one shared
A100-SXM4-80GB, bf16 autocast forward (2026-10-08 audit: cost.json's protocol label says "fp32" —
the forward actually runs under predict_fold's cuda autocast bfloat16; all arms share the path so
the ratios are unaffected; the script label is fixed after the round), batch 32, decode workers 16;
wall-clock of the UNMODIFIED `predict_fold`
in 256-image chunks over fold 2 (2523 imgs — the timed split2_seed1 pair's training fold; cost
is checkpoint-independent, no test fold read, no endpoint evaluated). GPU arms: x1 15.50 ms/img
(64.5 img/s, peak 4.8 GiB), x2-du2 52.88 (18.9, 12.9 GiB), x1-TTA 72.33 (13.8, 4.9 GiB). CPU
fusion: the frozen split2_seed19 EF-P2 deployment call replayed per image over its cached
validation fold — 25.97 ms/img mean (p95 36.9), wall≈process-CPU; **the replay reproduced the
frozen selection's 1542 additions exactly** (deployment-path check).

- **Headline ratios: two-model system (x1 + x2-du2 + fusion) = 6.09x single-model x1; x1-TTA
  alone = 4.67x.** Across the four launch windows (co-tenant load 19–46) absolute ms varied
  (x1 mean 10.7–15.5) but the ratios held: x2/x1 = 3.1–3.4, two-model/x1 = 6.1–6.3. TTA/x1 was
  the most load-sensitive (3.3–4.7; most CPU-side work).
- **E0 decision rule (RESEARCH_NEXT, pre-registered): no performance–cost advantage ⇒ EF-P2 is
  mechanism evidence / a teacher candidate, NOT a deployment gain.** The reported EF-P2 result
  (Dead-centred robust improvement, findings 2026-10-05) stands as a mechanism finding; any
  deployment framing must carry the ~6x inference cost (≈1.3x the cost of plain x1-TTA, which
  buys ~+0.5 pt mPQ but cannot add Dead detections).
- Incident + fix during the round: the fusion timer originally accessed `xz["inst"][j]` per
  image — NpzFile member access re-decompresses the whole array every time, silently timing npz
  decompression (~1–27 s/img) instead of fusion (~26 ms/img). Fixed by materializing the stacks
  once; the canonical run is post-fix and its added-count check binds it to the frozen path.
- Caveats: single shared GPU (wall-clock contamination quantified above); M/S lever not
  separately timed (CPU relabel, same order as fuse); fusion measured single-threaded
  (trivially parallel across images); model loading excluded; one machine, one batch size.

## 2026-10-08 (audit) — full code/docs/experiments consistency audit; in-place corrections + two artifact regenerations

Three-sonnet-agent read-only audit (docs↔code, docs-numbers↔artifacts, code↔recorded-runs + full
pytest), every flagged item re-derived by hand before editing. Headline results all reproduce:
gate A/B/C/D verdict values bit-consistent with `runs/analysis/dsb_gates_20261007/`, E0 6.09×
re-derived from cost.json, all stage-report/HV-controls/P0-P3 tables exact, protocol rules
(pannuke_eval, last-checkpoint, 3-split mean, no released-checkpoint test results) hold everywhere.
pytest after the audit's code hardening: 271 passed.

Corrections applied in place (all re-derived from artifacts this session):
- CellViT-UNI 3-split bPQ .6654 -> **.6653** (mean-of-rounded artifact; exact .66535), here,
  RESEARCH_PLAN, and two later mentions; recovery table means thr/beta/dead .4997 -> .4996.
- Recovery negative-result wording: "every change within ±.0004 test mPQ" -> within -.0012/+.0004
  (the -.0012 is split-3 "all", visible in the table); RESEARCH_PLAN phrasing aligned.
- Dead PQ val-grid range at k=4 ".181-.186" -> **.181-.184** (the .186 endpoint was the orphans
  variant); SYN1 "PQ+ FLAT" clarified to "Dead PQ+ flat".
- CF tissue paired deltas: CellViT -.189/-.193 -> -.193/-.189, HoVer-Net -.145/-.147 -> -.147/-.145
  (s1/s2 order; the other arms were already correct).
- PUMA HoVer-Net ±.0139 -> ±.0136, bPQ .7136 -> .7134, Dead mean .0326 -> .0327.
- KongNet train-fold F1 margin ">= .13" -> ">= .12 (min .124)" (ckpt1 .1244/.1274).
- split2_seed1 lever trio corrected to +.0012/+.0022/+.0007 (exact .001165/.002229/.000677).
- RESEARCH_PLAN real-domain drop CellViT .152 -> .154 (2-split mean .1551/.1519).
- Stale refs fixed: README M/S lever attributed to scripts/lever_postproc.py (not recovery.py);
  P0-P3 commands.json path; RESEARCH_NEXT line pointers. Two LM2-resident logs restored to
  `logs/` (analyze_cf.log, hn_pp_ablation.log) so their findings references resolve again.
- inference_cost.py: protocol label fp32 -> bf16-autocast (all arms share the autocast path, so
  the 6.09× is unaffected; frozen cost.json keeps its original "fp32" label — see the E0 entry
  note above), docstring command completed with the required --base-pred/--x2-pred.
- Hardening: new `scripts/fold_guard.py` — dsb_dev_eval / dsb_gate_c / dsb_oracle_ceiling /
  inference_cost now refuse their run split's TEST fold (and refuse unverifiable splits);
  dsb_dev_eval freezes its menu with input sha256 binding and refuses overwrite.
  Tests: tests/test_fold_guards.py + test_dead_merge updates.

Artifact regenerations (both reproduce the documented numbers exactly):
- `runs/analysis/failures_fold3.csv.gz` was lost on both machines; regenerated from
  `runs/cellvit_uni/split1/pred_fold3.npz` (deterministic) — fold-3 mining numbers verify
  exactly (isolated Dead missed_bg .421 n=882; merged Dead 94, touching Dead 77).
- split2_seed2's du2/du2lev test evals (grid 9/9, 2026-10-01) were lost on both machines
  (pred npz on disk is the 2026-10-04 watcher regeneration). Rebuilt from that npz with the
  pre-e1aaeee lever code (verified bit-exact against split2_seed1's recorded du2lev) + frozen
  (0.25, 40): du2 .4868/.6584/.1526, lever +.0016/+.0024/+.0007 — lever PASS and every
  grid verdict unchanged; 9-run means shift by <=.0004 (du2 .4889->.4893, du2lev
  .4905->.4909). The 2026-10-01 quoted values (+.0011/+.0021/+.0004) belonged to the lost
  original prediction and stand as recorded. Sidecar `pred_fold3_x2_du2_lev.npz.json` binds
  the regenerated lever npz's provenance.

Previously-unsourced numbers verified on LM2 this session (all match docs): HoVer-Net split2
TTA .4633/.6696/.105; SYN1 (mPQ/bPQ/Dead/TTA/Dead-PQ+/Dead-typed 367 vs 352/Infla -1.0/-1.6);
HoVer-NeXt-T per-split + means + error rates + Uterus-excluded Dead (.143/.171/.197);
PUMA per-arch components; LoRA-r8 sweep (od .60-1.10 vs .116, uni .13-.25, step-250 .80);
du2 VAL lever sweeps (+.0014/+.0012/+.0018, baseline .4836/.6455 -> best .4850/.6478);
external-table F_d and per-class columns. CONCH pillar-A probe outputs were never cached
anywhere — the only remaining unverifiable entries (gt embeddings exist, probes would need
CONCH re-runs).

## 2026-10-21 P6 closure and P7 submission

> **2026-10-21 P6 closed / P7 submitted:** All four P6 130-epoch runs completed; original bandwidth admission failed in both seeds (Dead +.02069 each, absolute .48966 and precision .66035/.67345). Independent P7 training-only RGB gains, C/J × seeds19/1, preserve the original cohort, targets, model, decode and gate. V100 smoke 55175430 passed; formal array 55176183 was submitted once: four tasks, concurrency2, maximum48 GPU-hours. Submission is not completion. No Phase B, masks or fold2/fold3 model evaluation. [P6 results](results/PROPOSAL_P6_RESULTS_2026-10-21.md), [P7 submission](results/PROPOSAL_P7_SUBMISSION_2026-10-21.md).


## 2026-10-22 — P7 verified partial result; independently frozen P8 submission

# P8 image-normalized center focal loss: formal submission

2026-10-22 America/New_York. **Array 55191961 was submitted once at 2026-10-22T12:21:30.982366Z UTC. Submission is not training completion.** Four fresh130 epoch runs C19/I19/C1/I1, concurrency2,1V100/4 CPU/32 GiB/12h each, gpu-shared/gpu-shared-normal/jhu159,Lustre,no-requeue; maximum48 GPU-hours. No automatic expansion.

P8 was frozen before reading P7 endpoint scores. [P7 partial review](results/PROPOSAL_P7_PROGRESS_2026-10-22.md) reports completed seed19 and explicitly leaves seed1 pending. P3-P6 closures and the original admission remain unchanged.

## Frozen question

Original focal numerator N_i and true instance count n_i. C retains sum_i N_i/max(1,sum_i n_i), preserving microbatch scalar reduction order. I uses mean_i N_i/max(1,n_i), including full empty-image negatives, actual effective tail B and counts from annotations rather than Gaussian peak counts. This changes the objective weighting and gradient scale, not sampling or class/tissue balancing. No inferred causal density mechanism. Same original D4/brightness/contrast; no P7 RGB gains.

Original model, sigma1/radius3 targets, exact fold1 fit 2134/audit 522, UNI weights, encoder blocks13-24, optimizer/schedule, effective16/micro4,BF16,last checkpoint, decoder and admission are fixed. No fold2/fold3 model evaluation, threshold/checkpoint selection, extra seeds/epochs, masks or automatic PhaseB. Original admission F1=C/T1=I: both seeds Dead>=.60,delta>=.05,original1000draw tissue-stratified paired CI lower>0,precision>=.75 and>=C-.02,overall recall>=C-.02,nondeg>=.95,support>=10 images/100 objects. Failure/inconclusive closes this exact recipe at its budget.

Fit-only count summary: 2134 images,50968 objects,93empty images; count percentiles[0,25,50,75,100]=[0.0, 12.0, 20.0, 31.0, 276.0]. Descriptive only, no changed sampling or tuning. Repeated audit 11Dead-positive images/145 objects is exploratory development, not independent validation/test/PQ or official three-split performance.

## Validation evidence

LM1 baseline57passed13.66s;18new tests first failed with missing-trainer assertions; P3-P8 focused75passed12.00s. Node75passed29.57s. Analytical loss/gradient tests cover counts[1,3,0],colliding centers,all-empty,unequal counts and actual tail3,exact C loss/gradient and full/micro numerical agreement. Bash/embeddedPython syntax and diff checks passed. Independent design/code review had no blocking issue; a minor pre-copy project capacity gap was fixed before sealing.

Real V100 smoke 55191831 COMPLETED/0:0,stderr empty. Both seeds' initial model states agree. C versus original P3 over two actual effective optimizer steps has bitwise equal all trainable gradients,adapted parameters and complete AdamW states. Both C/I train first256fit images with finite nonzero head/encoder gradients and updates, real BF16 internal operations/FP32 terminal bilinear, equal last/final reload predictions. Reload reconstruction/verification is timed separately, outside training. Preliminary measurements MEASURED; final PASS and strict ready bind all identities and script hashes.

| Arm | Original1.5x safety estimate for130 epochs(hours) | Training peak(GiB) | Reload reconstruction/verification(seconds) |
|---|---:|---:|---:|
|C|6.995|4.059|5.526|
|I|7.031|4.059|5.594|

Each estimate is below12 hours under the unchanged formula; it is not a completion guarantee. Resource checks use actual GPU free memory>=12 GiB,scratch>=80 GiB,atomic checkpoint20 GiB+1.1tensorbytes; project filesystem/group quota rechecked before copying large checkpoints. Existing P7/native13 jobs and connections were preserved.

This change does not touch metrics/HoVer-Net and uses focused verification per user AGENTS. No new full-suite pass is claimed. Historical P4 full regression: 588 passed / 3 skipped / 5 dense_votes failures remains disclosed: `tests/test_dense_votes.py::test_dense_targets_include_every_foreground_pixel_and_singletons`, `tests/test_dense_votes.py::test_foreground_offsets_vote_back_to_predicted_common_point`, `tests/test_dense_votes.py::test_single_pixel_and_empty_foreground_are_supported_without_area_filter`, `tests/test_dense_votes.py::test_dense_vote_decoder_has_no_gt_and_returns_classed_mask`, `tests/test_dense_votes.py::test_empty_gt_losses_and_dense_head_gradients_are_finite`.

## Provenance and recovery

Training commit 0a62bf8d4cc38bbe44e9ce2e992fff8fb20e6dc9; source manifest 795fb49c4ce4d947f5e19c9481ae4d79901c3a4def92a468737812a3574c8864; branch research/proposal-p8. Cohort 55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22. Original data/UNI/pinned p3 runtime reused, no dependency changes or large transfers. Source hashes and strict ready were rechecked before the single formal sbatch; create-only submission.started,job_id.txt and durable receipt exist. Do not rerun once-only scripts or submit another copy.

Operational receipts,review,validation logs and collected evidence are in ignored LM1 ops/p8_development_20261022. Project/scratch paths are in its HANDOFF. Query exact job IDs; retain failed/unfinished outputs, inspect unknown outcomes without replay. Checkpoints/results are preserved on project with SHA256 manifests and scratch logs; Lustre is not archival backup.

Current [SDSC Expanse guide](https://www.sdsc.edu/systems/expanse/user_guide.html) was rechecked this round; allocation/partition/QoS/quota were verified live. LM1/LM2 did not receive new GPU workloads. Main checkout/teammate changes remain intact; no shared merge or push.


## 2026-10-23 evening — P10 final closeout

All four 130-epoch runs completed with SLURM status 0:0. The 40 small files and 36 payload-preservation hashes were verified. S−C Dead recall changed by +.027586/−.013793, with 95% CIs [−.010000,.047297]/[−.055556,.051020]; the original gate was false for both seeds. S overall recall fell by .040408/.040980, and precision was .704591/.716963, below .75. Close the last-4 recipe. Fit recall also fell; this is not evidence of improved generalization. See [P10 results](results/PROPOSAL_P10_RESULTS_2026-10-23.md). These are reused fold-1 development results, not PQ or test results. The independent P11 spatial positive-mass redistribution recipe was frozen under the user's autonomous authorization.

## 2026-10-23 evening — P11 submitted

Formal array **55244166** was submitted once at **2026-10-23T20:07:03.418066-04:00 (America/New_York)**. Submission does not mean training completed. Task mapping: 0=C19, 1=G19, 2=C1, 3=G1. The array contains four fresh 130-epoch runs, with concurrency 2. Each requests 1 V100, 4 CPU, 32 GiB, and 12 hours on gpu-shared/gpu-shared-normal/jhu159 using Lustre and no requeue; the formal ceiling is 48 GPU-hours. Existing unrelated jobs were preserved.

Real V100 smoke **55243478 COMPLETED/0:0**, with stderr 0 bytes. Node checks reported 139 passed in 143.82 s (0:02:23). The two seeds' full initial states matched. C/P3's two effective updates had bitwise-equal trainable gradients, adapted parameters, and AdamW states. The independent G GPU loss and analytic gradients agreed; negative gradients outside support were unchanged; augmented positive shoulders and mass conservation were verified. Finite updates, AMP, frozen tensors, optimizer membership, last/final reload, cross-arm rejection, and strict readiness all passed. Safety-adjusted 130-epoch estimates were {'C': 7.197, 'G': 7.021} hours; training peaks were {'C': 4.072, 'G': 4.072} GiB. Estimates do not guarantee completion. The focused LM1 CPU suite reported 139 passed in 12.21 s. No metrics/HoVer-Net edits or current full-suite claim; the historical 5 dense_votes failures remain disclosed.

See [P11 submission](results/PROPOSAL_P11_SUBMISSION_2026-10-23.md). No formal completion or performance claim was made.

## 2026-10-24 — P12 closeout

All four 130-epoch runs completed. Original admission failed for both seeds (Dead recall R−C −0.034483 / −0.027586), so this fixed decoder-dropout recipe is closed. See [P12 results](results/PROPOSAL_P12_RESULTS_2026-10-24.md). These are development point metrics only; no test claim is made.

## 2026-10-24 evening — P12 independent closeout recheck; P13 submitted

The P12 original gate was reproduced exactly after checking 40 small files, 36 payload hashes, and the 130-epoch/cohort/configuration identities. Both seeds failed; Dead recall R−C was −.034483/−.027586. The fixed dropout recipe remains closed.

The independent P13 intervention replaced 25% of epoch positions with positions drawn from the 52 fit-only images containing any Dead. C/S × seeds 19/1; four fresh 130-epoch runs; original model, loss, augmentation, last checkpoint, decoder, and gate. At epoch 0, S yielded 578/573 Dead-image exposures and 1608/1613 unique images, with total exposures unchanged at 2134; duplicate augmentations were unchanged. Sampling changes image/background weighting and coverage, so this is not an isolated causal test.

The CPU suite reported 195 passed in 14.98 s. Real V100 smoke 55262502 completed/0:0. Strict two-seed C/P3 gradient/parameter/AdamW/RNG parity and S order, AMP, frozen parameters, reload, and timing checks passed. Formal array 55262695 was submitted once at 2026-10-24T18:41:22.310662-04:00, mapping 0–3%2, with each task requesting 1 GPU, 4 CPU, 32 GiB, and 12 h; maximum 48 GPU-hours. Completion was not claimed. See [P13 submission protocol and provenance](results/PROPOSAL_P13_SUBMISSION_2026-10-24.md). No fold2/fold3 evaluation, Phase B, or masks. Reused fold-1 development data, not PQ/test; no full-suite claim.

## 2026-10-25 — P14 closeout and P15 frozen translation hypothesis

All 4 tasks in P14 array 55270432 completed with exit status 0:0. The 41 small files (3,025,326 bytes) and 36 preserved-payload SHA/byte records were verified. Original configurations, cohort, data, UNI, and finite 130-epoch histories were verified; original admission was false for both seeds. Internal-Dead recall for C/L was .510345/.441379 (seed 19; delta −.068966, CI [−.114583,+.021517]) and .475862/.386207 (seed 1; delta −.089655, CI [−.154545,−.049451]). L precision was .776569/.786264, but overall recall was .495102/.475673 versus C at .583429/.595755. Candidate counts changed 10372→7810/10468→7411; fit recall fell about .978→.663/.631. No cap saturation. Descriptive Colon counts: 23/56→10/56 and 17/56→12/56; Lung counts: 39/60→39/60 and 39/60→33/60. These results are not causal and are not PQ/test. The fixed 1e-6 recipe is closed; no LR menu or extended epochs. Large checkpoints were not independently rehashed, and the Lustre archive remains outstanding.

P15's agent-chosen hypothesis was that missing translation under original D4/color augmentation may leave grid-position sensitivity; this is not an established mechanism. Original C versus J adds integer dx/dy in [-16,16] with white fill before the original augmentations; shifted centers are retained in-canvas, the Gaussian is rebuilt, no halo objects are dropped, duplicates count as objects, and the full effective-batch denominator is retained. Original model/LR1e-5/head3e-4/last12/130 epochs/fullpermutation/cohort/lastcheckpoint/decode/admission remain unchanged. Fresh C/J × 19/1, four fresh 130-epoch runs. RED missing-module tests became GREEN with 227 passed in 54.97 s; fresh design/code review found no blockers. Smoke 55278688 was submitted once and was pending validation at that snapshot; the same PENDING job was moved from debug to shared, with 30 min/resources/source unchanged. No formal result was claimed.

See [P14 results](results/PROPOSAL_P14_RESULTS_2026-10-25.md) and [P15 design](superpowers/specs/2026-10-25-proposal-p15-design.md). Reused fold-1 audit support is 11 positive images and 145 internal Dead objects, with conditional image bootstrap only; historical controls were not pooled. No Phase B/masks/fold2/fold3 model evaluation.

P15 smoke 55278688 completed/0:0 at the formal endpoint timestamp 2026-10-25T18:10:57.270105Z. The smoke passed 227 checks in 222.79 s (0:03:42), and strict readiness passed. Formal array 55278753 was submitted once with mapping 0-3%2 and a maximum of 48 GPU-hours. C/J safety estimates were {'C': 7.0267, 'J': 7.0429} hours; peak memory was {'C': 4.0709, 'J': 4.0709} GiB. Source commit cf9777be1914884f48d2d8cb142a9b09d7408fc6 and manifest 5654147fa8ba972d012e0bee07c1ff3dfc6d1013c4dfcefe17f27dfcd06995a5 identify the submission. See [P15 submission](results/PROPOSAL_P15_SUBMISSION_2026-10-25.md). The smoke and submission record did not establish training completion.

## 2026-10-26 — P15 closeout; P16 objective-control submission

All four P15 arm/seed runs completed for 130 epochs with exit status 0:0. Small-file preservation, input/cohort/config/source identities, and all 520 translation records were verified. Original admission was false for both seeds: Dead J−C was −.193103 [−.292453,−.124138] / −.089655 [−.200000,−.021739]. Precision was .8654/.8629, but overall recall was .3884/.3802 and fit all-class recall was about .470. Close the fixed translation recipe; this negative result does not rule out all spatial augmentation. See [P15 results](results/PROPOSAL_P15_RESULTS_2026-10-26.md) and [synthesis](results/PROPOSAL_PROGRESS_2026-10-26.md).

P16 is an independent fixed-Gaussian MSE-objective control, C/R × 19/1, preserving the original architecture/data/optimizer/130 epochs/decoding/admission, with no P15 translation. The focused LM1 suite reported 258 passed after RED/GREEN and a review-gap fix. V100 smoke 55287339 completed/0:0; 258 node checks passed. Checks covered real both-seed all-state C/P3 parity, the independent R loss/derivative and measured parameter updates, frozen parameters, AMP, reload, timing, and strict readiness. Formal array 55287551 was submitted once at 2026-10-26T05:29:54.529735Z, mapping 0-3%2, with a maximum of 48 GPU-hours. The submitted array had no completion result in the timestamped evidence used here; current completion has not been revalidated. See [P16 protocol, validation, and submission](results/PROPOSAL_P16_SUBMISSION_2026-10-26.md). Reused fold-1 development point metrics are not PQ/test; no automatic Phase B/masks/fold2/fold3/menus. Original recipes remain closed; external Lustre archive pending.
