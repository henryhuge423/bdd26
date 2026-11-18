# D32: locate the stages that produce center-detection errors

P31 has completed all four 130-epoch runs. Array 55792463 reports COMPLETED/0:0 for every task; the personal Expanse queue was empty at the November 17 review. Stage report 8 covers P27–P31. The next experiment uses existing last checkpoints to measure intermediate model outputs.

The selected comparison spans P22 full-depth adaptation, P29 first-six plus last-six adaptation, and P31 decoder group normalization, each with its own original control and seeds 19 and 1. These twelve models represent distinct measured behaviors: P22 gains concentrate in isolated Dead nuclei; P29 benefits are more dispersed among the retained development images; P31 improves training coverage, while development precision gains mainly accompany fewer predicted points. Stage report 7 supplies the earlier context, and stage report 8 describes each P27–P31 experiment.

Existing fixed-image measurements partition 1,059 unmatched predictions into 551 outside annotated masks, 505 inside an annotated nucleus but more than 3 pixels from its center, and 3 involved in matching competition. Across 53 fixed interior Dead objects and 20 correlated model outputs, 107 of 391 miss observations include a displaced prediction inside the correct mask. Integer-centroid sensitivity brings only 11 of those 107 observations within 3 pixels. These repeated observations motivate a measurement across the complete development set.

P22 adds 19 matches in each seed among 117 isolated interior Dead nuclei, with changes −1/+1 among 28 nuclei having another center within 12 pixels. All 11 small crowded development Dead nuclei occur in Colon image 1482. Training includes 38 objects in the corresponding coarse group across 13 images. Density, staining, morphology and image identity remain confounded; direct responses for that training group are needed.

The decoder produces 128-grid logits followed by bilinear interpolation to a 256-grid and sigmoid. A logit is the real-valued prediction before sigmoid; zero corresponds to score 0.5. For a single coarse peak a surrounded by background b, the corresponding fine maximum is (9a+7b)/16. With background score 0.01, this construction requires a coarse score of about 0.9727 to reach 0.5 after interpolation. Actual peak neighborhoods must be measured. Moving the same linear 1×1 output head across interpolation commutes and preserves the function class.

A separate exact-center focal-risk construction has a certainly present object whose conditional center lies at four equiprobable positions less than 3 pixels apart. The pixelwise risk optimum is about 0.4335 at each position. This mathematical example motivates measuring dispersed responses and input-shift stability; actual model outputs are needed to assess its relevance to observed misses.

The frozen scope is:

- P22 C/M, P29 C/B, P31 C/G, each seed 19/1; existing epoch-130 last.pth, with independent size, SHA256, configuration and epoch checks.
- Original fold 1 split: 2,134 training and 522 development images, with the same cohort hash. Recompute all 522 development images and retain every decoded point with matching and mask-membership records.
- Measure all 46 training images containing interior Dead nuclei plus the remaining original four training examples. Save maps and features for 11 development Dead-positive images, 13 training images containing small crowded Dead nuclei and the four original training examples.
- Apply four fixed one-pixel input translations to the 11 development images with white fill. Shift statistics use targets at least 16 pixels from the image boundary. Labels retain their original data role; translations are exploratory diagnostics.
- Preserve original batch 4, BF16 forward/FP32 logits, thresholds, cap 512 and unique radius-3 matching. Load one model at a time without optimizer updates.

For all 145 interior development Dead nuclei, retain maximum logits within radii 3/6/12, mask-wide maximum and position, local maxima, threshold survivors, cap survivors and final unique matches. Record overlapping diagnostic flags. Coarse support is the union of cells contributing to the fine radius-3 region; its maximum has different spatial support from the fine maximum. A nonnegative coarse maximum with negative fine maximum records an interpolation pattern, requiring examination of peak shape and position.

Passive hooks capture actual head inputs and the 128-grid output. On real hardware, require bit-identical output with and without hooks and exact reconstruction of final interpolation. Four normalized UNI feature taps are stored as float16 for compact analysis; logits remain float32. Preserve all differences from historical counts and retain historical outputs as the source of published metrics.

Artifacts include 180 large PNG panels: 11 development plus 4 original training examples per model. Also retain selected native images and labels, coarse/fine logits, feature arrays, complete peak decisions, object and shift CSV files, configuration and hashes. Panels show image and annotated centers, coarse score, fine score and uniquely matched, missed and extra points.

Interpretation proceeds by evidence: coarse-positive/fine-negative patterns motivate a peak-shape/interpolation investigation; weak coarse and fine responses despite supported training examples motivate representation/appearance analysis; displaced in-mask peaks or shift-dependent positions motivate localization analysis; eligible peaks lost later identify specific selection stages. Shared failures across models motivate common data/target investigation. Freeze a single training hypothesis only after these measurements.

Request one V100, 4 CPU cores, 32 GiB and 4 hours: a maximum of 4 GPU hours. Use the existing pinned environment and compute nodes for model loading, rehashing and inference. Recheck source identity, actual GPU memory, allocation, quota, filesystem space and fsync/readback before use. Preserve scratch originals and a project copy, then return small metrics and figures to LM1.

References: Objects as Points https://arxiv.org/abs/1904.07850; Vision Transformers for Dense Prediction https://arxiv.org/abs/2103.13413; On Focal Loss for Class-Posterior Probability Estimation, CVPR 2021. Project-specific interpolation and weighted-focal statements derive from the current implementation.

## Verified P31 outcomes and D32 submission

On the original development set of 12,250 nuclei and 145 interior Dead objects, P31 group normalization changes Dead matches from 65 to 73 (seed 19) and from 68 to 70 (seed 1). Experimental Dead recall is 50.34%/48.28%, with unique center precision 70.71%/70.14%. The original two-seed decision is false, closing this specific recipe. Overall development matches change by −140/+6; training recall reaches 99.56%/99.73%.

D32 job **55825240** was submitted once at **2026-11-17 21:44:41 EST**, requesting one V100, four CPU cores, 32 GiB and four hours. The first observed state is PENDING/Priority. SUBMISSION.json records the scientific source and actual receipt. The compute-node job will repeat behavior checks and require actual-model instrumentation equality before measuring the retained models.
