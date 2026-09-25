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

## Baselines (official 3-split protocol, last checkpoint; filled in as runs finish)
Test fold per split: 1 -> 3, 2 -> 3, 3 -> 1. Values: mPQ / bPQ / Dead PQ (no TTA; TTA in brackets).

| model | split 1 | split 2 | split 3 | mean |
|---|---|---|---|---|
| HoVer-Net (ours, full-patch) | .4558 / .6601 / .103 [.4664 / .6707 / .087] | .4529 / .6610 / .108 [.4633 / .6696 / .105] | .4606 / .6627 / .170 [.4700 / .6739 / .191] | .4564 / .6613 / .127 [.4666 / .6714 / .128] |
| CellViT-UNI (ours) | .4951 / .6676 / .142 [.5013 / .6728 / .146] | .4930 / .6638 / .162 [.4973 / .6683 / .163] | .5104 / .6647 / .224 [.5154 / .6707 / .221] | .4995 / .6654 / .176 [.5047 / .6706 / .177] |

Both baselines reproduce (HoVer-Net paper .463/.660, ours .456/.661; CellViT++ UNI .492, ours .500).
Course requirement (a) — reproducible HoVer-Net baseline with full official evaluation — DONE.

Reference (paper-reported): HoVer-Net .463 / .660; CellViT-UNI (CellViT++) .492.

## 2026-09-25 — Training-free recovery of missed nuclei (CellViT-UNI, 3 splits; ugrad L4s)
`scripts/dump_cellvit_maps.py` (fp16 NP/HV/TP maps) + `scripts/sweep_recovery.py`
(`nucseg.postproc.recovery`; default config tested identical to the official post-processing; light
metric tested equal to the full evaluator). 96 configs: beta (NP mixed with TP-branch foreground)
{0,.5,1} x k_dead (fg = max(fg, k P(Dead))) {0,1,2,4} x blob threshold {.3,.4,.5,.6} x orphan-blob
recovery. Selection per family by VALIDATION mPQ, one test scoring. Sanity: official config from the
fp16 maps reproduces the pipeline test mPQ within 0.0003 on every split.

Test mPQ (val-selected config per family):

| family | split 1 (test f3) | split 2 (test f3) | split 3 (test f1) | mean | mean Dead PQ |
|---|---|---|---|---|---|
| official | .4950 | .4927 | .5104 | .4994 | .176 |
| thr (tuning control) | .4953 (t.3) | .4929 (t.3) | .5108 (t.3) | .4997 | .176 |
| orphans | .4951 | .4927 | .5094 | .4991 | .177 |
| beta | .4953 (b.5) | .4929 (b0,t.3) | .5108 (b0,t.3) | .4997 | .177 |
| dead | .4953 (k0,t.3) | .4929 (k0,t.3) | .5108 (k0,t.3) | .4997 | .176 |
| all | .4953 (b.5) | .4929 (t.3,orph) | .5092 (t.3,orph) | .4991 | .178 |

Validation grid, mean over splits (beta 0, no orphans): mPQ is flat (.4902-.4906) across k and thr;
Dead PQ moves only .181-.186 even at k = 4. Orphans: val Dead +.004, Inflammatory +.003, bPQ -.0015;
no test gain.

=> **Negative**: missed Dead nuclei cannot be recovered post hoc from this model's outputs; every
change is within +-0.0004 test mPQ (the threshold-only control does as well as any recovery
variant). The detection deficit has to be addressed in training (foreground supervision / data).

## 2026-09-25 — Why Dead nuclei are missed (CellViT-UNI, 3 test folds pooled)
`scripts/show_nuclei.py` montage: `runs/analysis/dead_split1.png` (split 1, test fold 3; green = GT
nucleus, yellow = predictions). Qualitative (visual, not quantified): missed Dead GT are a mix of
(a) tiny dark nuclear fragments (karyorrhectic debris, often < 50 px), (b) round eosinophilic
apoptotic bodies without dark chromatin that do not look like nuclei, (c) a few ambiguous annotations;
matched Dead look like compact dark pyknotic nuclei.

Detection vs. GT area (fraction of GT nuclei with no overlapping prediction):

| GT area | Dead missed | Dead n | other classes missed |
|---|---|---|---|
| < 30 px   | .869 | 268  | .831 |
| 30-60     | .680 | 488  | .601 |
| 60-100    | .318 | 739  | .328 |
| 100-200   | .212 | 1073 | .125 |
| > 200     | .214 | 513  | .035 |

24% of Dead GT are < 60 px (other classes 6%) -> a size effect shared by all classes (IoU 0.5 is
very strict for such objects), plus an appearance effect: large Dead nuclei are missed 6x more often
than large nuclei of other classes. Training-side ablations on split 1 (running):
M1 NP pixel-CE with Dead pixels x11 (appearance), M2 same with nuclei < 100 px x6 (size),
C1 sampler gamma 1.0 (resampling control).

## 2026-09-25 — Training-side foreground supervision, split 1 (single seed; seed test running)
CellViT-UNI recipe + pixel-weighted NP cross-entropy (`--np-wce 1`), last checkpoint. Deltas vs. the
baseline (seed 19) with 95% paired, tissue-stratified image-bootstrap CIs (`scripts/compare_runs.py`;
test-set sampling noise only, NOT training noise). Decisions use the VAL fold (2); test (3) reported once.

| run | val mPQ | val d mPQ [CI] | val d Dead | test mPQ | test d mPQ [CI] | test d Dead [CI] |
|---|---|---|---|---|---|---|
| baseline (seed 19) | .4835 | | | .4951 | | |
| M1 Dead pixels x11 | .4798 | -.0036 [-.0076, .0000] | -.002 | .4915 | -.0036 [-.0085, +.0009] | +.009 [-.012, +.030] |
| M2 nuclei < 100 px x6 | .4822 | -.0013 [-.0050, +.0025] | +.001 | .4971 | +.0019 [-.0016, +.0059] | +.011 [-.004, +.027] |
| C1 sampler gamma 1.0 | .4840 | +.0005 [-.0041, +.0050] | -.010 | .4906 | -.0045 [-.0092, +.0003] | .000 [-.032, +.025] |

TTA test mPQ: baseline .5013, M1 .4975, M2 .4998, C1 .4953.

### Seed noise (baseline, split 1, seeds 19/1/2; `scripts/compare_runs.py`, runs/analysis/compare_split1_*.json)
| | val mPQ | val Dead | test mPQ | test Dead |
|---|---|---|---|---|
| mean | .4816 | .1643 | .4928 | .1393 |
| std over seeds | .0019 | .0029 | .0021 | **.0074** |

### Ablations vs. the 3-seed baseline (deltas, [95% image-bootstrap CI])
| run | val d mPQ | test d mPQ | test d Dead |
|---|---|---|---|
| M1 Dead x11 | -.0018 [-.0057,+.0019] | -.0013 [-.0061,+.0029] | +.0125 [-.0055,+.0295] |
| M2 small x6 | +.0006 [-.0035,+.0047] | **+.0042 [+.0009,+.0076]** | +.0141 [-.0027,+.0319] |
| C1 sampler 1.0 | +.0024 [-.0021,+.0067] | -.0022 [-.0066,+.0020] | +.0030 [-.0282,+.0287] |

Conclusions:
- **M1 (Dead-weighted CE) and C1 (full-balance sampler) are dead ends**: no val gain, no test gain.
- **M2 (small-nucleus x6) is the only candidate**: test mPQ +.0042 with CI excluding 0 and no val
  degradation — but the effect is ~2x the seed std (mPQ .0021), and its test Dead gain (+.014) is
  within seed noise on Dead (.0074 std) and test-sampling noise (+-.03). NOT confirmed.
- General rule established: on this benchmark a single-seed Dead-PQ change below ~.01-.015 is
  indistinguishable from noise (seed std .0074, test-sampling CI half-width .02-.03). Any Dead claim
  must be a multi-seed, 3-split mean.
