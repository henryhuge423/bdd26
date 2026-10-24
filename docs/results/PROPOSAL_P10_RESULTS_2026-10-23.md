# P10 restricted encoder adaptation: final development results

All four final-checkpoint runs completed 130 epochs with SLURM COMPLETED/0:0 and empty stderr. Forty small files (2,976,102 bytes) were returned; all 36 payloads match saved preservation-manifest SHA256 and byte counts. Configuration, exact cohort, input and UNI identities were verified. Training commit4163992bcd7387a96468015effd8f132d0a9f0d0; source manifest53965cfd9c5b7b95e9c351a37c63ca9740bf6f81eb3599ac55e0a6b4b733fa16; array55227144.

C adapts UNI last12 blocks; S last4. Original target, model/head, data, augmentation, optimizer, precision, 130epochs, final checkpoint, decoding and admission are frozen. Earlier feature taps remain frozen in S, so this compares whole adaptation-scope recipes, not parameter count in isolation.

| Seed | C audit Dead recall | S audit Dead recall | S−C | Paired 95% CI | C/S overall recall | C/S precision | Original admission |
|---|---:|---:|---:|---|---|---|---|
| 19 | 0.441379 | 0.468966 | +0.027586 | [-0.010000, +0.047297] | 0.590449/0.550041 | 0.693214/0.704591 | false |
| 1 | 0.475862 | 0.462069 | -0.013793 | [-0.055556, +0.051020] | 0.599918/0.558939 | 0.693891/0.716963 | false |

The two-seed mean Dead change is +0.006897; directions disagree and both CIs span zero. S absolute Dead .468966/.462069 is below .60, gains below .05, precision below .75, and overall recall drops .040408/.040980 beyond the .02 floor. The exact last4 recipe closes at the frozen budget. Lower memory/runtime does not rescue admission. No additional block-count menu, seeds, epochs, masks or PhaseB.

| Run | Fit overall recall | Fit Dead recall | Fit precision | Audit small-Dead hit/total | Candidates | Training h | Peak GiB |
|---|---:|---:|---:|---|---:|---:|---:|
| A_C_s19 | 0.972885 | 0.945122 | 0.998530 | 8/32 | 10434 | 4.001 | 4.047 |
| A_S_s19 | 0.841901 | 0.777439 | 0.981608 | 9/32 | 9563 | 3.189 | 2.255 |
| A_C_s1 | 0.978614 | 0.964939 | 0.998519 | 8/32 | 10591 | 4.032 | 4.047 |
| A_S_s1 | 0.851750 | 0.820122 | 0.984087 | 11/32 | 9550 | 2.913 | 2.255 |

No image reached the cap512. Fit overall recall falls from C .973/.979 to S .842/.852, and fit Dead from .945/.965 to .777/.820. Thus the smaller audit gap partly reflects worse fit, not evidence of improved generalization. Small-Dead hit counts improve descriptively from8 to9/11 of32; this selected subgroup cannot overturn the primary stop rule.

## Research interpretation across recent proposal rounds

| Round | Fixed treatment | Internal-Dead change, seed19 / seed1 | Original two-seed admission |
|---|---|---|---|
| P4 | Dense foreground BCE auxiliary | −.027586 / +.006897 | false |
| P5 | Matched-capacity native-grid refinement | .000000 / −.006897 | false |
| P6 | Wider negative-suppression Gaussian | +.020690 / +.020690 | false |
| P7 | RGB gain augmentation | +.034483 / −.013793 | false |
| P8 | Per-image nucleus normalization | −.034483 / −.048276 | false |
| P9 | Dead-center positive focal weight4 | +.006897 / +.048276 | false |
| P10 | Last4 rather than last12 adaptation | +.027586 / −.013793 | false |

The repeated pattern is high fit performance without sufficient audit precision/Dead recall. This motivates studying the supervision objective, but does not identify a causal bottleneck or exclude other members of these method families. Different rounds have fresh controls and are not pooled as paired replicates. No successful proposal method has been admitted.

P11 independently freezes per-image positive focal mass redistribution over the unchanged Gaussian support; it preserves total positive coefficient mass and original negatives. P6 only altered negative suppression, so it did not test this intervention. P11 is a new recipe choice, with the same cohort and original admission. No claim that it will improve performance.

## Limits and provenance

These are point matching rates on a repeatedly reused fold1 image development audit (522 images; only11 Dead-positive images/145internalDead), not PQ, patient-independent validation, test performance or a three-split result. Original tissue-stratified1000draw bootstrap RNG20261018 and all gates unchanged. No fold2/fold3 model evaluation.

Evidence: LM1 ignored ops/review_20261023_next/{p10_collection_manifest.json,p10_gate_01.json,p10_analysis_01.json,p10_live_01.json,p10_collected_01/}. Expanse project/scratch bdd26/p10_20261023_01 retains checkpoints; project/results has preservation manifests. Lustre external archive remains outstanding.


## P10 tissue decomposition (descriptive only)

The audit contains145internalDead across11images. Lung60objects/3images and Colon56objects/2images account for80% of these objects. Several remaining tissue strata have only1image; the fixed bootstrap cannot represent all uncertainty in such sparse strata. Tissue summaries are exploratory, not evidence for tissue-specific treatment choices. No new model evaluation or tuning.

| Tissue | Dead-positive images | InternalDead | S−C hits,seed19 | S−C hits,seed1 |
|---|---:|---:|---:|---:|
| Breast | 1 | 1 | +0 | +0 |
| Cervix | 1 | 15 | +4 | +3 |
| Colon | 2 | 56 | +0 | -7 |
| Esophagus | 1 | 8 | +1 | -2 |
| Lung | 3 | 60 | -2 | +4 |
| Stomach | 2 | 4 | +1 | +0 |
| Uterus | 1 | 1 | +0 | +0 |

Lung and Colon changes do not have a stable positive sign across both seeds. The small overall Dead gains/losses reflect only a few objects. This helps explain why subgroup positives cannot support promotion after the original admission failed. S−C total matched changes across all nuclei are −495/−502; each audit contains12,250GT nuclei.
