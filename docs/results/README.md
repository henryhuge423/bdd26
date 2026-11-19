> **2026-11-19 — P34 and P35 results; P36 submitted.** P34 (offset readout on frozen P22 detectors; three of four tasks completed, the fourth stopped at its first-epoch time guard): moving each predicted point gains 158–172 matches and loses 158–182 on the 522 development images of each detector; the readout places the center within three pixels for 99.4%–99.8% of nuclei on training images and for 71%–81% on development images. The frozen detectors match 96.0%–98.2% of nuclei on training images and 56.1%–64.6% on the same images moved by (5, 3) pixels. P35 (four frozen P22 detectors at sixteen alignments to the 16-pixel patch grid, array 55867961, all COMPLETED): nuclei at least 24 pixels from every border of the training images are matched at 97.4%–99.2% at the trained alignment and at 67.1%–72.6% at the fifteen others; on development images both figures are 61%–68%. Of 7,014 development nuclei, 11%–14% are matched at no alignment, 26%–38% at all sixteen and 51%–61% at some. Averaging the sixteen score maps at the threshold 0.5 keeps 9%–15% of nuclei; the primary criteria of P34 and P35 are not met. P36 records eight frozen detectors (P22 and P15, including the translation-trained P15 detector) at sixteen alignments with two fills, selects a score threshold and a consensus rule on shifted training images before any development image is read, and applies them unchanged: array **55892877**, eight tasks, each 32 cores/64 GiB/8 hours on the CPU allocation, submitted once at 2026-11-19T21:51:49Z. The project GPU allocation stood at 1,821 of 1,900 SU. [Bilingual analysis, figures, frozen design and receipt](alignment_p34_p36_2026-11-19/README.md).

> **2026-11-18 — Structure analysis of misses; center-offset readout P34 submitted and running.** An exclusive partition of the D32 records assigns each interior Dead nucleus to one of five outcomes. On the development images, 49.7% of 1,740 observations are matched, 11.0% respond above the threshold inside the nucleus with the peak beyond the three-pixel matching radius, 11.5% respond weakly and 27.8% do not respond; on the selected training images 96.1% of 7,872 observations are matched and the displaced pattern occurs 3 times. Matched points lie on the annotated center in training images (median 0.0 px) and 1.41 px away in development images. In every model, about 60–66% of all unmatched points lie inside an annotated nucleus that holds no matched point. A calculation on the implemented loss shows that the loss-optimal peak score falls below 0.5 once the center position is uncertain by more than about 0.95 px. P34 keeps four completed P22 detectors frozen and trains a 73,986-parameter offset readout on their last feature map, with and without random translation; the original, moved and moved-and-merged decoders are compared on identical retained points. The project GPU allocation stood at 1,818 of 1,900 SU with up to 83 SU of queued jobs from other members, so P34 runs on the CPU allocation in single precision: array **55855798**, four tasks, each 32 cores/64 GiB/10 hours, submitted once at 2026-11-18T19:31:05Z and running since 19:31:54Z. All tasks passed 44 node checks and the zero-offset identity check on the real detectors. [Bilingual analysis, theory, frozen design, figures and receipt](center_offset_p34_2026-11-18/README.md).

> **2026-11-18 — D32 recovered and D33 completed.** Twelve-model center-map diagnostics and the fixed interpolation-order comparison. [Bilingual analysis, object records and figures](center_diagnostics_recovery_2026-11-18/README.md).

> **2026-11-17 — P31 completed; D32 diagnostic designed.** [Analysis, bilingual plans and figures](center_diagnostics_2026-11-17/README.md).

> **2026-11-16 — P30 complete; P31 submitted and running.** Four P30 models completed 130 epochs, last-checkpoint evaluation and preservation. Middle-twelve adaptation adds 7/15 interior Dead matches and 648/516 all-nucleus matches, removing 215/254 unmatched points; both original decisions are false. We verified 117 small artifacts including 48 final PNGs. P31 adds eight fixed GroupNorm modules to the original point decoder while retaining last-twelve encoder updates. LM1 and V100 each passed 131 focused checks; actual hardware acceptance 55792375 completed in 369 seconds. Formal array **55792463** was submitted once at 2026-11-17T03:05:58Z: C19/G19/C1/G1, four independent 130-epoch runs, concurrency two, each 1 V100/4 CPU/32 GiB/12 hours. C19/G19 were running and C1/G1 waiting at 03:44 UTC. [P20–P30 detailed bilingual LaTeX reports, Chinese reader, large real-image comparisons and P31 receipt](p30_p31_2026-11-16/README.md).

> **2026-11-15–16 — P29 reviewed and preserved; P30 submitted.** Four P29 models completed 130 epochs and development evaluation. First-six plus last-six updates add 16/13 interior Dead matches and 283/165 total matches, removing 65/64 unmatched predictions; both original decisions are false. CPU recovery 55745860 preserved all 23 experimental seed-1 files, including independently verified checkpoints. P30 middle-twelve versus last-twelve passed CPU and V100 acceptance (121 focused checks each plus all hardware gates). Formal array **55746187** was submitted once at 2026-11-16T03:54:47Z: C19/M19/C1/M1, concurrency two, each 1 V100/4 CPU/32 GiB/11:52:00, 130 epochs. All four were PENDING/Priority at 04:22:29 UTC. The combined GPU envelope is 48.049722 hours within 48.5 hours. [Full-stage bilingual LaTeX reports, 61 result/analysis PNGs, four acceptance previews and actual receipt](p29_p30_2026-11-15/README.md).

> **2026-11-14 — P28 reviewed; P29 submitted.** Four P28 runs completed 130 epochs; 500 source files and 98 small payloads, including 48 final PNGs, were verified. Alternating encoder blocks added 528/273 total matches and 2/−1 interior Dead matches; both original decisions failed. P29 compares the first six plus last six blocks with the last twelve at equal capacity. LM1 and V100 acceptance passed 114 focused tests. Formal array **55735733** was submitted once at 14:31 EST: four 12-hour V100 tasks, concurrency two, maximum 48 GPU hours; all were PENDING/Priority at 14:36 EST. [Analysis, figures and receipt](p28_p29_2026-11-14/README.md).

> **2026-11-09 — P24 closed; P25 submitted:** P24 four130epoch runs completed; original two-seed gate failed despite consistent precision gain.126smallfiles/48finalPNG/479source verified. P25 fixed full24 layerwise adaptation passedCPU1086/3 andV10055635543(422s);formal **55635726** submitted once03:59EST,four12h tasks,concurrency2,max48GPUh. Tasks0/1RUNNING,2/3PENDING at firstcheck; submission is not completion. [EN/ZH/reader,rawresults and receipt](p24_p25_2026-11-09/README.md). Reusedfold1 point development,notPQ/test.

> **2026-11-04 — P22 closed;P23 submitted:** P22 all24 improves Dead recall by+.124138/+.137931 (positive paired intervals),but finalDead.586207/.565517 andprecision.728448/.733636 fail the original gates. Exact recipe closed. P23 tests fixed.999EMA on the original last12 shared trajectories,seeds19/1. CPU1029passed/3skipped plus17 storage checks;V10055536183PASS;formal **55538967**,0-1%2,1V100/4CPU/32GiB/12h each,24GPUh. Submission is not completion;reusedfold1 development,notPQ/test. [Analysis, figures, validation and receipt](p22_p23_2026-11-04/README.md).

> **2026-11-01 — P21 closed; P22 submitted:** P21 four 130-epoch runs completed (COMPLETED/0:0) after the Lustre recovery; 70 small payloads and 438 sealed-source hashes verified. Original two-seed gate failed: Dead A−C +.006897/−.034483, absolute Dead .448276/.448276, precision .699284/.696585; close the fixed five-class auxiliary recipe. P22 tests full-depth UNI adaptation (arm M: all 24 blocks trainable at the original rates) against the original last-12 control C, C/M × seeds 19/1, original cohort/loss/decoder/last checkpoint/gate. Full CPU 1002 passed, 3 skipped, 19073 warnings in 412.12s (0:06:52); V100 validation 55468892 COMPLETED/0:0 in 991 s with C/P3 parity, M depth witnesses, AMP, reload and strict readiness; 1.5x safety 7.0662/9.6372 h. Formal array **55469245** submitted once at 2026-11-01T14:19:48.884174Z: 0=C19/1=M19/2=C1/3=M1, concurrency 2, 1 V100/4 CPU/32 GiB/16 h each, 64 GPUh hard envelope (C self-limits at 12 h). Submission is not completion. [EN/ZH/reader](p21_p22_2026-11-01/README.md), [validation/receipt](p21_p22_2026-11-01/P22_SUBMISSION.md). Reused fold1 point development, not PQ/test; no masks/PhaseB/menu.

> **2026-10-30:** [P20 closeout and bilingual reader](p20_p21_2026-10-30/README.md); [P21 validation and formal submission](p20_p21_2026-10-30/P21_SUBMISSION.md). Submission is not completion.

> **2026-10-28:** [P19 closeout and bilingual reader](p19_p20_2026-10-27/README.md); [P20 amended validation and two formal arrays](p19_p20_2026-10-27/P20_SUBMISSION.md). Submission is not completion.

> **2026-10-27 — P18 closed; larger-batch P19 submitted:** All four P18 130-epoch runs completed; both original gates failed (Dead H−C −.006897/+.020690). Fitting coverage improved without qualified rare-class transfer. V100 validation55326654 passed; fitting-only microbatch16 improved measured throughput12.27% over4. Formal effective16/32 ×seeds19/1 array55326699 submitted once, concurrency2, max48GPUh; submission is not completion. [Detailed EN/ZH/Chinese reader](p18_p19_2026-10-27/README.md), [P19 protocol](p18_p19_2026-10-27/P19_SUBMISSION.md). Reused fold1 point metrics, not PQ/test; no automatic PhaseB/masks/menu.

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
