> **Resource amendment/status update:** Original per-task24h readiness failed in smoke55343640 (U safety25.76h). Before another allocation, resource-only commit04d187e changed C12/U36h, retaining96GPUh and all scientific conditions. Third validation55357273 passed after55356965 TIMEOUT and the separately frozen1GPUhour validation allowance; C/U arrays55357457,55357476 submitted. Historical24h readiness text below describes the initial freeze and is superseded for resources by [the full validation/receipt](P20_SUBMISSION.md). Submission is not completion.

# P19 closeout and the independently frozen P20 study

27 October 2026, America/New_York. This is an experiment record; the earlier Stage6 PDFs retain their dated snapshots.

## Decision and evidence
Close the exact effective-batch32 recipe. All four P19 array55326699 runs completed130epochs with SLURM COMPLETED/0:0. The original admission conjunction fails in both seeds; an independent reconstruction reproduces the original endpoint and bootstrap arithmetic exactly. The two-seed mean Dead change is −.03103448. This is a negative development result, not a statement that every large-batch optimizer fails.

We recovered42 small files (3,545,876-byte collection), verified their transfer bytes/SHA256 and all collected entries against each run's in-job preservation manifest. All418 sealed source files were checked on Expanse; source-manifest SHA256 is49217e27e937b7ae373d6abd5c43da162c3109ca99a7c08784c3751fa2777cd1. Four130epoch histories, finite losses, original configurations/UNI/input/tissue/cohort identity, disjoint fit/audit image order and saved summaries agree. The histories record134/67 updates per epoch for C/B:17420/8710 updates over the run. The expected native tails are6/22; logs do not independently preserve every microbatch's tensor. No large checkpoint was independently transferred to LM1 or reread by the collector. In-job scratch→project copies and preservation hashes remain on Lustre.

## Matched fresh controls
C is effective16, B effective32; both share the fit-selected physical16, fixed learning rates,130epochs,2134fit/522audit images, original targets/augmentation/decoding and final checkpoint. Within-P19 C/B is the intended matched comparison. Earlier rounds often used physical4, so the P19 C score should not be treated as a bitwise cross-round reproduction; the benchmark established numerical tolerance for physical batching, not final-training trajectory identity.

| Seed / arm | Audit Dead hit/145 | Audit Dead recall | Audit recall | Unique precision | Fit recall | Fit Dead recall |
|---|---:|---:|---:|---:|---:|---:|
| 19 / C | 66/145 | 0.455172 | 0.584408 | 0.695319 | 0.973689 | 0.946646 |
| 19 / B | 57/145 | 0.393103 | 0.582122 | 0.688653 | 0.947398 | 0.916159 |
| 1 / C | 63/145 | 0.434483 | 0.592245 | 0.693860 | 0.979144 | 0.958841 |
| 1 / B | 63/145 | 0.434483 | 0.575673 | 0.694847 | 0.940963 | 0.920732 |

| Seed | Dead B−C | Original paired 95% CI | Overall recall B−C | Precision B−C | Original gate |
|---|---:|---|---:|---:|---|
| 19 | -0.062069 | [-0.125000, -0.022727] | -0.002286 | -0.006666 | fail |
| 1 | +0.000000 | [-0.027273, +0.016949] | -0.016571 | +0.000987 | fail |

The original gate requires Dead recall≥.60, gain≥.05, positive CI lower bound, unique precision≥.75 and no more than.02 precision/overall-recall loss, support≥10positive images/100interiorDead objects and nondegeneracy≥.95, in both seeds. Support, relative precision/recall and nondegeneracy pass. Both seeds fail the absolute Dead/gain/positive-CI/absolute-precision clauses. Seed19 loses9netDead matches; seed1's identical63/145 totals conceal gains and losses across images. The records lack cross-arm object identity, so net counts cannot be called individual rescue/retention rates. No arm hits the512proposal cap.

## What the fitting and tissue results imply
Batch32 lowers overall fit recall from.973689→.947398 and.979144→.940963. InteriorDead fit recall falls from621/656→601/656 and629/656→604/656. Audit overall matched counts change−28/−203, while candidate counts change+59/−307. This is not a clean precision-versus-recall improvement and does not support better transfer. The controlled intervention changes update count, gradient noise, AdamW moment history and accumulated decay. The weaker fitting results are compatible with less optimization progress at the fixed epoch budget. They do not isolate which optimizer mechanism caused it. Do not add epochs, scale LR, add seeds or retune thresholds after seeing these results.

| Tissue | Dead objects | Seed19 net hits B−C | Seed1 net hits B−C |
|---|---:|---:|---:|
| Breast | 1 | +0 | +0 |
| Cervix | 15 | -2 | +1 |
| Colon | 56 | -1 | +5 |
| Esophagus | 8 | -3 | -4 |
| Lung | 60 | -2 | -2 |
| Stomach | 4 | -1 | +0 |
| Uterus | 1 | +0 | +0 |

Lung/Colon contain116/145interiorDead objects across5images. In seed1, Colon gains5hits, but Esophagus loses4 and Lung loses2; the aggregate is unchanged. Seed19 losses are spread across Cervix,Colon,Esophagus,Lung andStomach. Per-tissue rows are descriptive: tiny subgroup denominators and the shared images cannot support tissue-generalization claims or a new subgroup-selected recipe.

## Runtime and throughput
| Seed / arm | Training h | 130 epoch-save h | Run elapsed h | Peak allocated GiB |
|---|---:|---:|---:|---:|
| C_19 | 3.6228 | 0.7388 | 4.4336 | 5.981 |
| B_19 | 3.5556 | 0.9102 | 4.5334 | 6.557 |
| C_1 | 3.6137 | 0.1977 | 3.9279 | 5.981 |
| B_1 | 3.5582 | 0.1914 | 3.8316 | 6.557 |

Training-only time falls just1.86%/1.53% for B versus C. Halving optimizer calls leaves nearly the same image-forward/backward work; effective32 also still uses physical16. Epoch-save costs vary substantially by seed/node/time (seed19 roughly44–55minutes versus11–12minutes forseed1), and make the end-to-end comparison reverse between seeds. These are shared-cluster observations on different nodes, not a controlled speed benchmark. Earlier short physical4/8/16 fitting-only measurements found12.27% higher throughput for16; that does not imply batch32 halves formal runtime. Keep physical-batch throughput and effective-batch optimization conclusions separate.

## P3–P19 interpretation and next experiment
Earlier frozen records close tested foreground auxiliary supervision, decoder-grid refinement, target bandwidth, stain gains, normalization, rare-class weights, adaptation depth, redistribution, dropout, resampling, low encoder LR, translation, Gaussian regression, scratch CNN and RGB residual recipes. This round independently closes P19. The evidence supports a persistent fit/audit transfer gap in the fixed point pipeline; it does not establish a universal impossibility or identify a single cause. The increasingly reused audit cohort makes another broad hyperparameter menu scientifically weak.

P20 was chosen and committed as4b141ce before opening P19 scientific scores. It tests a structurally distinct working-scale recipe: augmented nativeRGB256 is either encoded directly(C) or bilinearly enlarged512 before UNI normalization/encoding(U). Four token lattices become16²/32²; the same head returns256² logits and all GT/peak/matching coordinates remain native. It adds no optical detail. Positional interpolation, attention, tokenization and decoder internal size all change, so this is a full recipe comparison. P5 varied output-grid refinement with fixed encoder input; historical segmentation x2 results do not answer this point-model question. P20 restores effective16/physical4 in both arms and retains original LR/loss/augmentation/cohort/last-checkpoint/dual-seed gate.

Implementation uses direct original C forward, actual U token-shape assertions, and explicit pre-normalization bilinear interpolation with align_corners=False. Readiness requires both-seed full original-control state/gradient/AdamW/RNG equivalence over two effective steps, real U512input and1025token taps, nonzero second-step updates beyond pure weight decay, all frozen tensors unchanged, BF16 and strict last/final reload,134fit-image timing with tail6 and no audit prediction. Smoke is bounded by.5GPUh. Formal four-run concurrency2 has24h/item (96GPUh hard maximum), intentionally larger than prior12h because of the4×token lattice; measured1.5safety extrapolation must fit24h and training peak≤24GiB. No batching or science fallback if it fails.

## Reproduction and interpretation limits
Use the pinned project Python/NumPy and `scripts/analyze_proposal_p19.py --collection P19_COLLECTION.json --out NEW_ANALYSIS.json`. Supply the original small collection from ignored handoff/local delivery; an existing output is deliberately refused. `P19_NUMBERS.json` records scientific fields and checksums; the handoff retains operational accounting, source and collection. An initial independent-analysis invocation passed the original admission API a flat map; it returned missing_seed and comparison stopped. Correcting only the nested data schema reproduced the unchanged independent numbers and gates. The failed log remains preserved.

Bootstrap is1000draws with seed20261018, sampling Dead-positive images with replacement within sorted tissue strata. Its95%intervals are conditional image intervals on this reused fold1development cohort. They are not independent-patient intervals, multiplicity-adjusted search-level inference, PQ, official three-split means or test performance. We used last checkpoints and no fold2/fold3 model evaluation. Prior closed recipes/PhaseB/masks remain closed. Independent external checkpoint archiving is still outstanding. Current implementation, CPU/GPU evidence and submission status are recorded separately in P20_SUBMISSION.md once available.
