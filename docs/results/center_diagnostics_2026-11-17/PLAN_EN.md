# D32: locate the stages that produce center-detection errors

P31 has completed all four 130-epoch runs. Array55792463 reports COMPLETED/0:0 for every task; the personal Expanse queue was empty at the November17 review. Stage report8 covers P27–P31. The next experiment uses existing last checkpoints to measure intermediate model outputs.

The selected comparison spans P22 full-depth adaptation, P29 first-six plus last-six adaptation, and P31 decoder group normalization, each with its own original control and seeds19/1. These twelve models represent distinct measured behaviors: P22 gains concentrate in isolated Dead nuclei; P29 benefits are more dispersed among the retained development images; P31 improves training coverage, while development precision gains mainly accompany fewer predicted points. Stage report7 supplies the earlier context, and stage report8 describes each P27–P31 experiment.

Existing fixed-image measurements partition1059 unmatched predictions into551 outside annotated masks,505 inside an annotated nucleus but more than3pixels from its center, and3 involved in matching competition. Across53 fixed interior Dead objects and20 correlated model outputs,107 of391 miss observations include a displaced prediction inside the correct mask. Integer-centroid sensitivity brings only11 of those107 observations within3pixels. These repeated observations motivate a complete-data measurement.

P22 adds19/19 matches among117 isolated interior Dead nuclei, with changes−1/+1 among28 nuclei having another center within12pixels. All11 small crowded development Dead nuclei occur inColon1482. Training includes38 objects in the corresponding coarse group across13 images. Density, staining, morphology and image identity remain confounded; direct training-group responses are needed.

The decoder produces128-grid logits followed by bilinear256-grid interpolation and sigmoid. For a single coarse peak a surrounded by background b, the corresponding fine maximum is(9a+7b)/16. With background score.01, this construction requires a coarse score of about.9727 to reach.5 after interpolation. Actual peak neighborhoods must be measured. Moving the same linear1×1 output head across interpolation commutes and preserves the function class.

A separate exact-center focal-risk construction has a certainly present object whose conditional center lies at four equiprobable positions less than3pixels apart. The pixelwise risk optimum is about.4335 at each position. This mathematical example motivates measuring dispersed responses and input-shift stability; it supplies no bound on actual network outputs or evidence that this mechanism caused observed misses.

The frozen scope is:

- P22 C/M, P29 C/B, P31 C/G, each seed19/1; existing epoch130 last.pth, with independent size/SHA256/configuration/epoch checks.
- Original fold1 training2134/development522 split and cohort hash. Recompute all522 development images and retain every decoded point with matching and mask-membership records.
- Measure all46 training images containing interior Dead nuclei plus the remaining original four training examples. Save maps/features for11 development Dead-positive images,13 training images containing small crowded Dead nuclei and the four original training examples.
- Apply four fixed one-pixel input translations to the11 development images with white fill. Shift statistics use targets at least16pixels from the image boundary. Labels retain their original data role; translations are exploratory diagnostics.
- Preserve original batch4, BF16 forward/FP32 logits, thresholds, cap512 and unique radius3 matching. Load one model at a time without optimizer updates.

For all145 interior development Dead nuclei, retain maximum logits within radii3/6/12, mask-wide maximum and position, local maxima, threshold survivors, cap survivors and final unique matches. Record overlapping diagnostic flags. Coarse support is the union of cells contributing to the fine radius3 region; its maximum has different spatial support from the fine maximum. A nonnegative coarse maximum with negative fine maximum records an interpolation pattern, requiring examination of peak shape and position.

Passive hooks capture actual head inputs and the128-grid output. On real hardware, require bit-identical output with and without hooks and exact reconstruction of final interpolation. Four normalized UNI feature taps are stored as float16 for compact analysis; logits remain float32. Preserve all differences from historical counts and retain historical outputs as the source of published metrics.

Artifacts include180 large PNG panels:11 development plus4 original training examples per model. Also retain selected native images/labels, coarse/fine logits, feature arrays, complete peak decisions, objectCSV, shiftCSV, configuration and hashes. Panels show image/GT, coarse score, fine score and unique matched/missed/extra points.

Interpretation proceeds by evidence: coarse-positive/fine-negative patterns motivate a peak-shape/interpolation investigation; weak coarse and fine responses despite supported training examples motivate representation/appearance analysis; displaced in-mask peaks or shift-dependent positions motivate localization analysis; eligible peaks lost later identify specific selection stages. Shared failures across models motivate common data/target investigation. Freeze a single training hypothesis only after these measurements.

Request oneV100,4CPU,32GiB,4hours: a maximum4GPU-hour diagnostic envelope. Use the existing pinned environment and compute nodes for model loading, rehashing and inference. Recheck source identity, actual GPU memory, allocation, quota, filesystem space and fsync/readback before use. Preserve scratch originals and a project copy, then return small metrics and figures toLM1.

References: Objects as Points https://arxiv.org/abs/1904.07850; Vision Transformers for Dense Prediction https://arxiv.org/abs/2103.13413; On Focal Loss for Class-Posterior Probability Estimation,CVPR2021. Project-specific interpolation and weighted-focal statements derive from the current implementation.
