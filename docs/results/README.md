# Experiment results

## 2026-10-17 query/point representation closeout

- [E7–E18 synthesis and decisions](QUERY_POINT_CLOSEOUT_E7_E18_2026-10-17.md)
- [E17 sparse predicted-point detector](PREDICTED_POINTS_E17_DIAGNOSTIC_2026-10-17.md)
- [E18 dense foreground-to-point voting](DENSE_VOTES_E18_DIAGNOSTIC_2026-10-17.md)


> **2026-10-17 E8a/E8b:** The frozen classifier-only probe fails allocation: fit foreground accuracy95.17%/97.58%, audit foreground changes-7.69/-6.87pp and final geometric recall gains+2.20/+2.47pp. Stop this balanced-head recipe. Separate original-model audit of all65Dead-positive fold1 images/967objects finds raw-mask matches35.57%/36.40%, final typed matches2.79%/12.20%. These are train-only diagnostics, not validation/test performance. [Full mechanism review](QUERY_MECHANISM_2026-10-17.md).


> **2026-10-17 E7 review:** All four 130-epoch runs and final-checkpoint fold2 evaluations are complete. Slower decay does not rescue Q (two-seed mPQ .087958 vs matched B .472939). A new fixed-train-batch audit localizes substantial loss between assigned raw masks and query retention/overlap resolution; this is diagnostic, not method performance. The independent clean-GPU closeout is complete (exit0): all quality gates fail, interior-Dead matching change -.528340 [95%CI -.589426,-.468664]; cost passes (Q/B .738576/.764427). Preserve the failed recipe; no expansion. [Detailed E7 review](SET_MASK_E7_DEV_2026-10-17.md).


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
