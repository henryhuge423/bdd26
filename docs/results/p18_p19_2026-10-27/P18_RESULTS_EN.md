# P18 closeout and P19 research decision — 27 October 2026

P18 completed all four fresh C/H × seeds 19/1 runs for 130 epochs. Every SLURM array task 55311007_0–3 reports COMPLETED/0:0. The RGB residual treatment failed the original admission rule in both seeds; close this fixed recipe. It slightly improves overall point coverage while leaving the rare-class generalization barrier unresolved. P19 is a separately frozen effective-batch experiment; its subsequent measured validation and submission are attached below.

## Question and fixed comparison

C retains adapted UNI last 12 blocks and the original point head. H adds a normalized-RGB 32-channel convolutional path to final logits, with zero-initialized output and RNG-preserving construction. It supplements rather than replaces the pretrained representation. Original augmentation, fold 1 fit 2134/audit 522, Gaussian targets sigma 1/radius 3, exact-center focal loss normalized by whole-effective-batch object count, effective 16/micro 4, head LR 3e-4/backbone 1e-5, AdamW, clipping,130 epochs, last checkpoint and threshold 0.5/radius 3 unique point matching remain fixed. Added parameters participate in global gradient clipping, so this tests input-path/capacity/optimization interactions together.

## Per-seed results and original admission

| Seed / arm | Audit Dead hits /145 | Audit overall recall | Audit precision | Fit overall recall | Fit Dead recall /656 |
|---|---:|---:|---:|---:|---:|
| 19 / C | 71 = 0.489655 | 0.585469 | 0.690678 | 0.974278 | 0.949695 |
| 19 / H | 70 = 0.482759 | 0.600898 | 0.694368 | 0.985167 | 0.958841 |
| 1 / C | 64 = 0.441379 | 0.593633 | 0.696152 | 0.976456 | 0.946646 |
| 1 / H | 67 = 0.462069 | 0.596653 | 0.696228 | 0.985442 | 0.966463 |

| Seed | H−C Dead recall | Original 95% interval | Net Dead hits | Pass |
|---|---:|---|---:|---|
| 19 | -0.006897 | [-0.029703, +0.022727] | -1 | false |
| 1 | +0.020690 | [-0.021517, +0.041667] | +3 | false |

For BOTH seeds, the treatment must have Dead recall≥0.60, gain≥0.05, original bootstrap lower bound>0, unique precision≥0.75 and≥C−0.02, overall recall≥C−0.02, nondegeneracy≥0.95 and support≥10 images/100 objects. Both seeds pass support, nondegeneracy, relative precision and overall-recall preservation; both fail absolute Dead recall, minimum gain, positive lower bound and absolute precision. A positive mean must not replace those clauses. The descriptive two-seed mean gain is+0.006897, or+0.69 percentage points; it is not an estimate of a training-seed population effect.

The bootstrap was independently implemented without importing admission or aggregate. For each seed, reset NumPy default_rng to 20261018; group only Dead-positive images by sorted tissue; sample the original number of images with replacement inside every tissue for 1000 draws; divide summed paired hit differences by the sampled Dead denominator; take 2.5/97.5 percentiles. Both deltas, intervals and final gates exactly match the supplied verified JSON.

## Where the small differences occur

| Tissue | Images / Dead | C19→H19 | Net 19 | C1→H1 | Net 1 |
|---|---:|---:|---:|---:|---:|
| Breast | 1 / 1 | 0→0 | +0 | 0→0 | +0 |
| Cervix | 1 / 15 | 8→7 | -1 | 8→6 | -2 |
| Colon | 2 / 56 | 20→15 | -5 | 15→17 | +2 |
| Esophagus | 1 / 8 | 4→6 | +2 | 5→4 | -1 |
| Lung | 3 / 60 | 36→40 | +4 | 35→37 | +2 |
| Stomach | 2 / 4 | 2→1 | -1 | 0→2 | +2 |
| Uterus | 1 / 1 | 1→1 | +0 | 1→1 | +0 |

| Image | Tissue | Dead | C19→H19 (net) | C1→H1 (net) |
|---|---|---:|---:|---:|
| 178 | Breast | 1 | 0→0 (+0) | 0→0 (+0) |
| 776 | Colon | 7 | 4→2 (-2) | 4→3 (-1) |
| 1025 | Lung | 25 | 18→18 (+0) | 19→20 (+1) |
| 1037 | Lung | 20 | 10→12 (+2) | 8→9 (+1) |
| 1038 | Lung | 15 | 8→10 (+2) | 8→8 (+0) |
| 1482 | Colon | 49 | 16→13 (-3) | 11→14 (+3) |
| 1756 | Cervix | 15 | 8→7 (-1) | 8→6 (-2) |
| 2076 | Esophagus | 8 | 4→6 (+2) | 5→4 (-1) |
| 2482 | Stomach | 3 | 1→1 (+0) | 0→1 (+1) |
| 2485 | Stomach | 1 | 1→0 (-1) | 0→1 (+1) |
| 2587 | Uterus | 1 | 1→1 (+0) | 1→1 (+0) |

Lung and Colon contribute 116/145 interior Dead objects (80%) from 5/11 positive images. Seed 19 gains 4 in Lung but loses 5 in Colon; seed 1 gains 2 in both tissues but loses 2 in Cervix and 1 in Esophagus. Seed 19 has 3 positive-net,4 negative-net and 4 unchanged images; seed 1 has 5/3/3. Image 1482 alone contributes 49/145 Dead objects. These are paired image-level count differences, not identities of newly rescued versus newly lost objects: raw per-image summaries cannot reconstruct object-level crossover. No tissue-specific threshold or training rescue follows from these descriptive rows.

## Fitting, coverage and candidate behavior

| Seed / arm | Matched /12250 | Candidates | Empty images /522 | Median / max candidates | Small Dead hits /32 |
|---|---:|---:|---:|---:|---:|
| 19 / C | 7172 | 10384 | 22 | 17 / 125 | 8 |
| 19 / H | 7361 | 10601 | 24 | 17 / 123 | 9 |
| 1 / C | 7272 | 10446 | 22 | 17 / 117 | 5 |
| 1 / H | 7309 | 10498 | 23 | 17 / 120 | 8 |

Overall audit matching increases 7172→7361 (+189) and 7272→7309 (+37), while candidate counts rise 217 and 52. Precision changes only+0.003690/+0.000076. Fitting overall recall rises 97.43%→98.52% and 97.65%→98.54%; fitting Dead recall rises 94.97%→95.88% and 94.66%→96.65%. The fitting/audit gap persists. All 11 Dead-positive audit images retain candidates in every run. The 512 cap is inactive everywhere in the saved fit/audit summaries; audit maxima are 117–125. Small interior Dead (<67 native pixels, the inherited bin) improve 8→9 and 5→8 of 32, a descriptive change insufficient to qualify the recipe.

P17's from-scratch CNN instead had fitting overall recall 14.41%/11.92% and similarly weak audit coverage. P18 therefore avoids that fitting collapse, but does not demonstrate qualified rare-class transfer. P13's near-perfect fitting Dead recall did not generalize; P15 reduced fitting coverage; P16 smooth regression yielded modest audit gains without passing precision/absolute recall. These different failure patterns motivate diagnosis rather than treating every negative study as the same phenomenon. Cross-round treatment ranking is inappropriate: each round is paired against its own fresh C, and audit images are repeatedly reused.

## Time and storage contribution

| Seed / arm | Training total s | Mean training epoch s | Epoch checkpoint total s | Complete elapsed s | Checkpoint / elapsed |
|---|---:|---:|---:|---:|---:|
| 19 / C | 14526.31 | 111.74 | 711.34 | 15549.54 | 4.57% |
| 19 / H | 15992.66 | 123.02 | 279.08 | 16420.79 | 1.70% |
| 1 / C | 14502.63 | 111.56 | 449.89 | 15232.56 | 2.95% |
| 1 / H | 16006.17 | 123.12 | 279.20 | 16440.75 | 1.70% |

Training seconds include runner target preparation, transfers, forward/backward and optimizer work, not just kernels. Checkpoint totals sum the 130 per-epoch save durations; other time includes initialization, final checkpoint, final fit/audit evaluation and miscellaneous overhead, so it should not all be labeled I/O. Complete elapsed comes from complete.json, whereas the earlier summary's elapsed field was the final epoch timestamp and excludes subsequent evaluation. H training takes about 123s/epoch versus C about 111.6s, but C checkpoint totals vary 711/450s versus H 279/279s. C ran on exp-1-59 and H on exp-9-58. Node/storage differences prevent attributing wall-time differences purely to architecture. They also show why the requested batch test should separately measure training throughput and end-to-end runtime.

## P19: separate hardware batching from optimization batching

The frozen P19 comparison uses original UNI C effective 16 versus B effective 32, fresh seeds 19/1 and 130 epochs, original LR with no scaling, unchanged cohort/loss/augmentation/decoder/admission. This gives 134 versus 67 optimizer updates per full 2134-image epoch, with tails 6 versus 22. Gradient noise, clipping histories, Adam moments and cumulative AdamW decay change; it is a fixed optimization recipe, not a pure acceleration equivalence experiment.

Choose shared physical microbatch 4/8/16 using fitting-only 256-image benchmark trials with effective 16, identical initialization/input ordering, warmup/synchronization and three timed trials in counterbalanced orders. A candidate must satisfy the frozen finite-gradient/numeric thresholds and memory ceiling; choose the fastest median only if at least 5% faster than 4. No audit prediction may influence selection. BF16 changes kernel/reduction behavior, so selected batching is numerically checked rather than promised bitwise equivalent over 130 epochs.

Independent review requested candidate-local OOM/numeric rejection with durable evidence and fallback, production-equivalent timing without per-parameter synchronization checks, true parameter-update error rather than final-weight-normalized error, both-seed numerical witnesses, and actual runner C/B smoke with 278 fitting images to exercise 6/22 tails, correct whole-effective object denominators, clipping/update counts, metadata and strict last/final reload. Final validation receipts, selected batch, actual speeds, memory measurements and formal job identity must be attached separately when validation completes. No P19 pass or speedup is inferred from this design.

## Scope, provenance and next action

Point matching measures localization at a fixed working point; it does not establish mask contours, IoU, PQ or three-split/test performance. The 11 positive images are reused development data, with concentrated tissue support and unknown independent patient structure. Image bootstrap quantifies conditional image resampling uncertainty; it does not correct repeated development selection or establish independent-patient or training-seed population inference. Null Dead results in tissues without labels are unavailable support, not perfect performance.

The collection SHA256 is `7e32ac10b08af90511d39568b4c677780fbaa6aae7d389b796b37dcff2a28bf2`. The sealed source manifest SHA256 is `314fb59bcd9968965361ec0d5f567f54c495e2f79adacc33f7ed24e82e7cc47e`. Independently rehashed all 36 small payloads against collection metadata, checked seven per-run payloads against their preservation manifests, verified complete epochs 1–130, saved cohort ordering and common UNI/input/cohort identity, and reproduced all four audit aggregates. The collection records 413 source files verified upstream; independent checks cover the small payloads and statistical analysis. Large checkpoints were not independently transferred/rehashed. The durable record is `p18_independent_analysis.json`; primary inputs are `p18_collection_final.json` and `p18_verified_final.json`.

Close P18 without width/depth/LR/threshold/epoch/seed rescue. Proceed with P19 only after its frozen compute-node readiness and resource checks pass. Failure or inconclusiveness closes the fixed P19 recipe; success supports designing independent validation, not automatic mask/PhaseB expansion.


Reproduce the quantitative checks with `python analyze_p18_independent.py --root <directory-containing-the-two-input-JSON-files>`. The script reads raw per-image payload strings directly, verifies the 36 payload digests and per-run preservation records, independently computes aggregates and the frozen bootstrap, and checks its results against `p18_verified_final.json` and the durable independent analysis. Add `--out <new-json-path>` to save the reproduced machine-readable record. The original rule was inspected in `source/pmnet_p3_2026-10-18/proposal_p3_utils.py` (the archived reference to project `scripts/proposal_p3_utils.py`); its admission function is not imported by the independent implementation.


## Measured P19 validation and submission update

The attached [P19 protocol](P19_SUBMISSION.md) records completed V100 validation55326654 and formal array**55326699**, submitted08:50EDT. Shared physicalbatch16 was selected: median256-image times4/8/16 are13.273/12.543/11.823s, yielding12.27% greater throughput over4. Actual C/B smoke peaks6.004/6.580GiB. The formal array is submitted, not completed. Full quantitative evidence is in[P19_VALIDATION.json](P19_VALIDATION.json). FinalCPU: 919 passed, 3 skipped, 19073 warnings in 458.12s (0:07:38).
