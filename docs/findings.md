# Findings log

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
  into the per-instance type decision (see RESEARCH_PLAN.md §2.1 update).
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
