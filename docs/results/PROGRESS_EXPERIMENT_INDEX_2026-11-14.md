# Complete progress experiment coverage — 2026-11-14

The English PDF and editable PPTX use the same authored content. Each PPTX slide contains English speaking notes and Chinese explanation. PDF is compiled from LaTeX/TikZ.

The complete sequence covers published historical results and execution states through November 14. P29 was queued at 16:08 EST. Historical result numbers are traced to committed reports; this update does not rerun their training or independently rehash every checkpoint.

| Slide | Experiment / topic | Source |
|---:|---|---|
| 1 | Nuclei instance segmentation on PanNuke | README.md; docs/RESEARCH_PLAN.md; experiment records through 2026-11-14 |
| 2 | A reader’s route through the experiments | This deck’s experiment index and per-slide source notes |
| 3 | What must be correct for one nucleus to count? | docs/RESEARCH_PLAN.md; src/nucseg/metrics/pannuke_eval.py |
| 4 | Read instance quality and center detection separately | docs/RESEARCH_PLAN.md; docs/results/p28_p29_2026-11-14/RESULTS_EN.md |
| 5 | Three evidence scopes in the project | CLAUDE.md; docs/RESEARCH_PLAN.md; docs/results/p28_p29_2026-11-14/RESULTS_EN.md |
| 6 | The reproducible starting point | CLAUDE.md; README.md; docs/audits/RESEARCH_AUDIT_2026-10-09.md |
| 7 | Baseline and released-weight reproductions | README.md: Reproductions and released-weight re-evaluations |
| 8 | A-CONCH — Language-guided nucleus classification | docs/RESEARCH_PLAN.md: 2026-09-25; docs/report/stage_report_1/ |
| 9 | A-Retyping — Fuse text-informed predictions into CellViT | docs/RESEARCH_PLAN.md: 2026-09-25; docs/report/stage_report_1/ |
| 10 | A-Recovery — Training-free missed-nucleus recovery | docs/RESEARCH_PLAN.md: 2026-09-25 evening; docs/report/stage_report_1/ |
| 11 | Loss weighting and sampling: the early controls | docs/RESEARCH_PLAN.md: 2026-09-25/26; docs/findings.md |
| 12 | Why Dead nuclei became the research target | docs/RESEARCH_PLAN.md: 2026-09-25/26; docs/report/stage_report_1/ |
| 13 | Baseline — the missed nuclei in the actual image | Previous progress slide6; docs/report/stage_report_1/ |
| 14 | CP1 — Copy real rare nuclei into background | docs/RESEARCH_PLAN.md: 2026-09-26; docs/report/stage_report_1/ |
| 15 | PixCell adaptation: appearance fidelity came first | docs/RESEARCH_PLAN.md: 2026-09-26; docs/report/stage_report_1/ |
| 16 | SYN1 — Train with contextual synthetic tissue images | docs/RESEARCH_PLAN.md: 2026-09-26 night; docs/report/stage_report_1/ |
| 17 | Real external-domain evaluation | docs/RESEARCH_PLAN.md: 2026-09-26–28; docs/report/stage_report_1/ |
| 18 | Counterfactual rendering: what did it actually stress? | docs/RESEARCH_PLAN.md: 2026-09-26–28; docs/report/stage_report_1/ |
| 19 | B1 — Double the working resolution | docs/results/P0_P3_RESULTS_2026-10-02.md; docs/report/stage_report_2/ |
| 20 | The audited historical P0–P3 sequence | docs/results/P0_P3_RESULTS_2026-10-02.md; docs/audits/RESEARCH_AUDIT_2026-10-09.md |
| 21 | Matched seeds and the area-filtered candidate | README.md: Status 2026-10-05; docs/report/stage_report_3/ |
| 22 | Candidate existence and class confidence | docs/RESEARCH_PLAN.md: 2026-10-03/05; docs/report/stage_report_3/ |
| 23 | E2a — Separate image scale from HV target support | docs/results/HV_CONTROLS_RESULTS_2026-10-07.md |
| 24 | Dead specialist branch: all four mechanism checks | docs/RESEARCH_PLAN.md: 2026-10-07/08; docs/report/stage_report_4/ |
| 25 | E0 — Measure the actual two-model deployment cost | docs/results/inference_cost_audit_20261009.json; docs/report/stage_report_4/ |
| 26 | Typing headroom and the T0–T3 controls | docs/report/stage_report_5/; docs/RESEARCH_PLAN.md: 2026-10-10 |
| 27 | T0–T3 — what each fixed-geometry arm changed | docs/report/stage_report_5/stage_report_5.tex: fixed-geometry probe |
| 28 | E1 — Label review and detector agreement | docs/report/stage_report_5/; docs/RESEARCH_PLAN.md: 2026-10-10–12 |
| 29 | E1-T — Which label does the detector jury support? | docs/report/stage_report_5/; docs/RESEARCH_PLAN.md: 2026-10-12 |
| 30 | E3 — Learn candidate usefulness | docs/report/stage_report_5/; docs/RESEARCH_PLAN.md: 2026-10-12 |
| 31 | E3 — all five candidate-addition replay arms | docs/report/stage_report_5/stage_report_5.tex: candidate-quality replay |
| 32 | M-labels — Merge labels to examine the metric response | docs/RESEARCH_PLAN.md: 2026-10-13/14; docs/findings.md |
| 33 | E4 — Audit visible fragments near image boundaries | docs/results/E4_VISIBILITY_PROBE_2026-10-13.md |
| 34 | Visible-fragment supervision: four arms, two seeds | docs/results/VISIBILITY_SUPERVISION_DEV_2026-10-15.md |
| 35 | Visible-fragment supervision: the final score table | docs/results/visibility_supervision_original_image_snapshot_2026-10-15.json |
| 36 | Visibility-C2 — Did selective HV reduce fragment misses? | docs/results/VISIBILITY_SUPERVISION_MECHANISM_2026-10-15.md |
| 37 | Geometry feasibility before a new representation | docs/results/PMNET_TRAIN_GEOMETRY_AUDIT_2026-10-15.md; docs/results/PMNET_POINT_GRID_AUDIT_2026-10-15.md; docs/results/POINT_MASK_FEASIBILITY_2026-10-15.md |
| 38 | E5 — Fuse features with a learned gate | docs/results/FUSION_GATING_DEV_2026-10-15.md |
| 39 | E6 — Replace dense masks with learned object queries | docs/results/SET_MASK_E6_DEV_2026-10-16.md |
| 40 | E7 — Test slower decay for the query model | docs/results/SET_MASK_E7_DEV_2026-10-17.md |
| 41 | E8a — Foreground classifier repair | docs/results/QUERY_MECHANISM_2026-10-17.md |
| 42 | E8b — Trace raw masks through the final decoder | docs/results/QUERY_MECHANISM_2026-10-17.md |
| 43 | E9 — Initialize queries from local image content | docs/results/LOCAL_QUERY_E9_DEV_2026-10-17.md |
| 44 | E10 — Restrict attention with predicted masks | docs/results/MASKED_QUERY_E10_DEV_2026-10-17.md |
| 45 | E11 — Supervise intermediate query stages | docs/results/AUX_QUERY_E11_DEV_2026-10-17.md |
| 46 | E12 — Can frozen features fit masks with true assignments? | docs/results/FREE_MASK_E12_DIAGNOSTIC_2026-10-17.md |
| 47 | E13 — Freeze the shared mask projection | docs/results/STABLE_MASK_E13_DEV_2026-10-17.md |
| 48 | E14 — Compute matching at native image resolution | docs/results/NATIVE_MATCH_E14_DEV_2026-10-17.md |
| 49 | E15 — Read a mask at a given true nucleus point | docs/results/POINT_READOUT_E15_DIAGNOSTIC_2026-10-17.md |
| 50 | E16 — Give the point readout more feature context | docs/results/CONTEXT_READOUT_E16_DIAGNOSTIC_2026-10-17.md |
| 51 | E17 — Predict sparse centers before mask readout | docs/results/PREDICTED_POINTS_E17_DIAGNOSTIC_2026-10-17.md |
| 52 | E18 — Let dense pixels vote for nucleus centers | docs/results/DENSE_VOTES_E18_DIAGNOSTIC_2026-10-17.md |
| 53 | E19 — Add high-resolution features at the finest mask skip | docs/results/HIRES_ADAPTER_E19_DEV_2026-10-18.md |
| 54 | Point-P1-v3 — Frozen features with a short center-head budget | docs/results/P1_FROZEN_POINT_PROBE_RESULT_2026-10-18.md |
| 55 | Point-P1-R1 — Correct convergence, then measure transfer | docs/results/P1R1_RESULT_2026-10-18.md |
| 56 | Point-mask P2 remained a proposal | docs/results/P1R1_RESULT_2026-10-18.md; docs/results/PROPOSAL_P3_PHASE_A_2026-10-20.md |
| 57 | The common P3–P29 center-detection protocol | docs/results/p28_p29_2026-11-14/RESULTS_EN.md; docs/results/PROPOSAL_PROGRESS_2026-10-26.md |
| 58 | The original development acceptance conditions | scripts/proposal_p3_utils.py; docs/results/p28_p29_2026-11-14/RESULTS_EN.md |
| 59 | P3 — Encoder adaptation and augmentation | docs/results/PROPOSAL_P3_PHASE_A_2026-10-20.md |
| 60 | P3 — the complete eight-run factorial comparison | docs/results/PROPOSAL_P3_PHASE_A_2026-10-20.md |
| 61 | P4 — Foreground auxiliary loss | docs/results/PROPOSAL_P4_PHASE_A_2026-10-20.md |
| 62 | P5 — Refine at the native center grid | docs/results/PROPOSAL_P5_RESULTS_2026-10-21.md |
| 63 | P6 — Widen the Gaussian suppression neighborhood | docs/results/PROPOSAL_P6_RESULTS_2026-10-21.md |
| 64 | P7 — Apply fixed RGB gains during training | docs/results/PROPOSAL_P7_P8_RESULTS_2026-10-22.md |
| 65 | P8 — Normalize each image by its object count | docs/results/PROPOSAL_P7_P8_RESULTS_2026-10-22.md |
| 66 | P9 — Increase Dead-positive center weight | docs/results/PROPOSAL_P9_RESULTS_2026-10-23.md |
| 67 | P10 — Update only the last four encoder blocks | docs/results/PROPOSAL_P10_RESULTS_2026-10-23.md |
| 68 | P11 — Redistribute exact-center positive mass | docs/results/PROPOSAL_P11_RESULTS_2026-10-24.md |
| 69 | P12 — Regularize the decoder with channel dropout | docs/results/PROPOSAL_P12_RESULTS_2026-10-24.md |
| 70 | P13 — Show more rare-positive training images | docs/results/PROPOSAL_P13_RESULTS_2026-10-25.md |
| 71 | P14 — Lower the encoder learning rate | docs/results/PROPOSAL_P14_RESULTS_2026-10-25.md |
| 72 | P15 — Translate training images and center targets | docs/results/PROPOSAL_P15_RESULTS_2026-10-26.md |
| 73 | P16 — Regress the Gaussian center map | docs/results/PROPOSAL_P16_RESULTS_2026-10-26.md |
| 74 | P17 — Train a small convolutional detector from scratch | docs/results/PROPOSAL_P17_RESULTS_2026-10-26.md |
| 75 | P18 — Add an RGB residual to point logits | docs/results/p18_p19_2026-10-27/P18_RESULTS_EN.md |
| 76 | P19 — Increase effective batch size | docs/results/p19_p20_2026-10-27/P19_NUMBERS.json |
| 77 | P3–P19: two distinct reasons for weak coverage | docs/results/PROPOSAL_PROGRESS_2026-10-26.md; docs/results/PROPOSAL_P17_RESULTS_2026-10-26.md |
| 78 | P20 — Enlarge input to 512 pixels | docs/results/p20_p21_2026-10-30/P20_NUMBERS.json |
| 79 | P21 — Add class-specific center supervision | docs/results/p21_p22_2026-11-01/P21_NUMBERS.json |
| 80 | P22 — Update all 24 encoder blocks | docs/results/p22_p23_2026-11-04/P22_NUMBERS.json |
| 81 | P22 — lung1037: compare the final matched objects | docs/results/p22_p23_2026-11-04/; fixed final-checkpoint matching panels |
| 82 | P23 — Average parameters along the training trajectory | docs/results/p23_p24_2026-11-06/RESULTS_EN.md |
| 83 | P24 — Increase the last-block learning rate | docs/results/p24_p25_2026-11-09/RESULTS_EN.md |
| 84 | P24 — training coverage and development coverage | docs/results/p24_p25_2026-11-09/RESULTS_EN.md |
| 85 | P25 — Use layerwise rates across all blocks | docs/results/p25_p26_2026-11-10/RESULTS_EN.md |
| 86 | P26 — Constrain weights toward the pretrained encoder | docs/results/p26_p27_2026-11-12/RESULTS_EN.md |
| 87 | P26 — the weight anchor weakened center fitting | docs/results/p26_p27_2026-11-12/RESULTS_EN.md |
| 88 | P27 — Move the twelve trainable blocks to the start | docs/results/p27_p28_2026-11-12/RESULTS_EN.md |
| 89 | P28 — Distribute updates through alternating blocks | docs/results/p28_p29_2026-11-14/RESULTS_EN.md |
| 90 | P20–P23 — the paired count record | docs/results/p28_p29_2026-11-14/stage_pairs.csv |
| 91 | P24–P28 — the paired count record | docs/results/p28_p29_2026-11-14/stage_pairs.csv |
| 92 | D24 — Diagnose why individual Dead nuclei were missed | docs/results/p23_p24_2026-11-06/RESULTS_EN.md |
| 93 | Nine paired interventions: three different outcomes | docs/results/p28_p29_2026-11-14/stage_pairs.csv |
| 94 | P28 — which evaluation images gained or lost nuclei? | docs/results/p28_p29_2026-11-14/object_summary.json |
| 95 | P28 — lung1037, seed 19: the actual object exchanges | docs/results/p28_p29_2026-11-14/fixed_objects.csv; figures/p28_image_1037_seed_19.png |
| 96 | P28 — trace the same lung nuclei through both outputs | docs/results/p28_p29_2026-11-14/fixed_objects.csv; RESULTS_EN.md |
| 97 | P28 — colon776: unchanged counts conceal substitution | docs/results/p28_p29_2026-11-14/fixed_objects.csv |
| 98 | P28 — support concentration and fixed-example coverage | docs/results/p28_p29_2026-11-14/object_summary.json; RESULTS_EN.md |
| 99 | P29 — Adapt the first six and last six blocks | docs/results/p28_p29_2026-11-14/P29_SUBMISSION.md; operational status2026-11-14T21:08:38Z |
| 100 | P29 — what is fixed, and what is still to be measured? | docs/results/p28_p29_2026-11-14/P29_SUBMISSION.md |
| 101 | Infrastructure interruptions remain in the record | P20_SUBMISSION.md; P21_RECOVERY/STORE_RECOVERY; p24_submission_2026-11-07/; p28_account_recheck_20261113 |
| 102 | What the complete experiment sequence establishes | Complete experiment coverage and source index accompanying this deck |
| 103 | Preservation and reproducible handoff | docs/results/p28_p29_2026-11-14/collection_result.json; experiment index; build validation receipt |
