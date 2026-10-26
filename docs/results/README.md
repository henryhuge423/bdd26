> **2026-10-26 evening — P17 closed; P18 submitted:** Four P17 130-epoch runs completed/0:0; original admission failed in both seeds. The fixed scratch CNN has low fit/audit coverage despite high precision. Independently frozen RGB residual P18 passed V100 smoke 55310632 and formal array 55311007 was submitted once (C/H x19/1,concurrency2,max48GPUh). Submission is not completion. Reused fold1 point metrics are not PQ/test evidence; no automatic masks/PhaseB/menu. [P17 results](PROPOSAL_P17_RESULTS_2026-10-26.md), [P18 protocol](PROPOSAL_P18_SUBMISSION_2026-10-26.md), [中文分析](RESEARCH_REVIEW_2026-10-26_EVENING_ZH.md), [中文导读](RESEARCH_READER_2026-10-26_EVENING_ZH.md).

Latest closeout (2026-10-26, 11:29 EDT): [P16 completed, original admission false for both seeds](PROPOSAL_P16_RESULTS_2026-10-26.md); P17 production training remains in progress.

# Experiment results

## Latest evidence: October 26, 2026

[Stage report6: Chinese report and plain-language digest](../report/stage_report_6/README.md). Current09:16EDT P16/P17 status is recorded in the new partial/submission documents; the older pending snapshot below is historical.

The P-series studies use a repeatedly reused **fold1 development point-detection audit**.
These are not PQ scores, held-out test results, independent validation or official
three-split performance. Both original seed-specific gates must pass; an isolated gain
does not override a failed gate. The audit includes 522 images, with 11 positive images
and 145 internal Dead objects. See the [P3–P15 synthesis](PROPOSAL_PROGRESS_2026-10-26.md).

| Study | Tested change | Recorded outcome | Evidence |
|---|---|---|---|
| P3 | Encoder adaptation with augmentation | Original admission failed | [Result](PROPOSAL_P3_PHASE_A_2026-10-20.md) |
| P4 | Foreground BCE | Both seed gates failed; recipe closed | [Result](PROPOSAL_P4_PHASE_A_2026-10-20.md) |
| P5 | Native-grid refinement | Both seed gates failed; recipe closed | [Result](PROPOSAL_P5_RESULTS_2026-10-21.md) |
| P6 | Target bandwidth | Both seed gates failed; recipe closed | [Result](PROPOSAL_P6_RESULTS_2026-10-21.md) |
| P7 / P8 | RGB gains / image-level loss normalization | Both recipes failed their original two-seed gates | [Results](PROPOSAL_P7_P8_RESULTS_2026-10-22.md) |
| P9 | Dead-center positive weight | Both seed gates failed; recipe closed | [Result](PROPOSAL_P9_RESULTS_2026-10-23.md) |
| P10 | Last-four-block encoder adaptation | Both seed gates failed; recipe closed | [Result](PROPOSAL_P10_RESULTS_2026-10-23.md) |
| P11 | Positive focal-mass redistribution | Both seed gates failed; recipe closed | [Result](PROPOSAL_P11_RESULTS_2026-10-24.md) |
| P12 | Decoder channel dropout | Both seed gates failed; recipe closed | [Result](PROPOSAL_P12_RESULTS_2026-10-24.md) |
| P13 | Rare-positive replacement sampling | Both seed gates failed; recipe closed | [Result](PROPOSAL_P13_RESULTS_2026-10-25.md) |
| P14 | Lower encoder learning rate | Both seed gates failed; recipe closed | [Result](PROPOSAL_P14_RESULTS_2026-10-25.md) |
| P15 | Training translation | Both seed gates failed; recipe closed | [Result](PROPOSAL_P15_RESULTS_2026-10-26.md) |
| P16 | Direct Gaussian regression | Seed19 complete/original condition failed; seed1 running | [Partial results](PROPOSAL_P16_PROGRESS_2026-10-26.md) |
| P17 | From-scratch convolutional representation | Formal55297061 submitted09:08EDT after V100 validation; completion pending | [Submission](PROPOSAL_P17_SUBMISSION_2026-10-26.md) |

P16 array 55287551 was submitted once at `2026-10-26T05:29:54.529735Z`.
The stored `2026-10-26T05:30:48.240388+00:00` snapshot shows four pending tasks;
it is historical evidence, not a live scheduler query. Do not resubmit from this page.
The [documentation audit](../audits/DOCUMENTATION_RESULTS_GIT_AUDIT_2026-10-26.md)
records the evidence scope and branch map.

## Earlier segmentation and mechanism results


## 2026-10-18 high-resolution adapter dev (E19)

- [E19 result: dev negative](HIRES_ADAPTER_E19_DEV_2026-10-18.md) -- a high-resolution CNN adapter on the finest decoder skip of the CellViT-UNI baseline does not improve internal-Dead recall; `dev_admission=false` (clauses 1/2/3 fail, cost clause passes).

## 2026-10-17 query/point representation closeout

- [E7–E18 synthesis and decisions](QUERY_POINT_CLOSEOUT_E7_E18_2026-10-17.md)
- [E17 sparse predicted-point detector](PREDICTED_POINTS_E17_DIAGNOSTIC_2026-10-17.md)
- [E18 dense foreground-to-point voting](DENSE_VOTES_E18_DIAGNOSTIC_2026-10-17.md)


> **2026-10-17 E8a/E8b:** The frozen classifier-only probe fails allocation: fit foreground accuracy 95.17%/97.58%, audit foreground changes −7.69/−6.87 pp and final geometric recall gains +2.20/+2.47 pp. Stop this balanced-head recipe. Separate original-model audit of all 65 Dead-positive fold1 images / 967 objects finds raw-mask matches 35.57%/36.40%, final typed matches 2.79%/12.20%. These are train-only diagnostics, not validation/test performance. [Full mechanism review](QUERY_MECHANISM_2026-10-17.md).


> **2026-10-17 E7 review:** All four 130-epoch runs and final-checkpoint fold2 evaluations are complete. Slower decay does not rescue Q (two-seed mPQ .087958 vs matched B .472939). A new fixed-train-batch audit localizes substantial loss between assigned raw masks and query retention/overlap resolution; this is diagnostic, not method performance. The independent clean-GPU closeout is complete (exit 0): all quality gates fail, interior-Dead matching change -.528340 [95% CI  -.589426,-.468664]; cost passes (Q/B .738576/.764427). Preserve the failed recipe; no expansion. [Detailed E7 review](SET_MASK_E7_DEV_2026-10-17.md).


## 2026-10-15 point/mask representation feasibility

[Point/mask training-label audit](POINT_MASK_FEASIBILITY_2026-10-15.md): full fold1 CPU geometry, neighbourhood suppression risk, exact hard-block mask IoU oracle, interior/border sensitivity and HV30 semantic correction. Independent arithmetic and sparse-graph verification complete. No fit, new inference, fold3 or changed stop rules. [Next minimal comparison draft](../proposals/RESEARCH_NEXT_2026-10-15-instance-representation.md).


- [Fusion gating dev closeout](FUSION_GATING_DEV_2026-10-15.md): primary gate fails; PQ gains and quality floors do not override the stop rule.
- [PM-Net point-grid feasibility](PMNET_POINT_GRID_AUDIT_2026-10-15.md): train-only legal targets and stride collision counts.

- [E4 visibility probe](E4_VISIBILITY_PROBE_2026-10-13.md): corrected 2026-10-14 run, frozen gates and limits.
- [HV threshold controls](HV_CONTROLS_RESULTS_2026-10-07.md): four arms, two seeds and audit-bound results.
- [P0–P3](P0_P3_RESULTS_2026-10-02.md): audited historical results and corrections; later matched-seed evidence is in [stage III](../report/stage_report_3/README.md).
- `hv_threshold_controls_20261007_verified.json` and `inference_cost_audit_20261009.json`: portable verification snapshots.

Keep dated results and verification evidence together. Later corrections must identify the
superseded version and preserve the original artifacts. [Documentation index](../README.md).

- [E6 final development results, trainability probe and E7 allocation (2026-10-16)](SET_MASK_E6_DEV_2026-10-16.md).

- [E9 local-query mechanism closeout,2026-10-17](LOCAL_QUERY_E9_DEV_2026-10-17.md): both train-only allocation gates fail; no expansion.

- [E10 masked-query mechanism closeout,2026-10-17](MASKED_QUERY_E10_DEV_2026-10-17.md): exact control reuse; both original gates and A0 guards fail; no expansion.

- [E11 auxiliary-query development probe: gate failed](AUX_QUERY_E11_DEV_2026-10-17.md)

- [E12 fixed-feature GT fitting diagnostic](FREE_MASK_E12_DIAGNOSTIC_2026-10-17.md)
- [E13 stable projection: allocation failed](STABLE_MASK_E13_DEV_2026-10-17.md)


P12 closeout (2026-10-24): all four 130-epoch runs completed; original admission failed for both seeds (Dead recall R−C −0.034483 / −0.027586). This fixed decoder-dropout recipe is closed. See `PROPOSAL_P12_RESULTS_2026-10-24.md`. Development point metrics only; no test claim.
