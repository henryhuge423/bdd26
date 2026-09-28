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

Conclusions (single seed):
- **M1 (Dead-weighted CE) and C1 (full-balance sampler) are dead ends**: no val gain, no test gain.
- M2 flagged as the only candidate (single-seed test mPQ +.0042, CI excluding 0) pending seeds.

### M2 three-seed verdict (seeds 19/20/21 vs baseline seeds 19/1/2; split 1)
Runs `split1_npwce_small5{,_s20,_s21}` (`small_w 5, small_area 100`). The single-seed test gain
does NOT replicate (95% paired image-bootstrap CIs):

| | val d mPQ | test d mPQ | test d bPQ | test d Dead |
|---|---|---|---|---|
| 3-seed M2 | -.0007 [-.0029,+.0016] | -.0010 [-.0043,+.0022] | **-.0024 [-.0048,-.0002]** | +.0108 [-.0021,+.0238] |

Per-seed test Dead: M2 .1533/.1444/.1523 vs base .1424/.1308/.1446 — the Dead direction is
consistent (3/3 seeds above the baseline mean) but not significant, and it is bought with a small
significant bPQ loss; mPQ is unchanged.

=> **M2 rejected as a contribution** (a trade, not a gain). All pillar-A training-side ablations
(M1 Dead-weight, M2 small-weight, C1 balanced sampler) are null. The +.0042 single-seed effect was
seed luck — validating the rule below. The Dead detection deficit needs data-level intervention
(pillar B synthesis / copy-paste) or architecture changes, not loss reweighting.

General rule (validated): on this benchmark a single-seed mPQ change below ~.004 and any Dead-PQ
change below ~.01-.015 are indistinguishable from noise (seed std mPQ .0021 / Dead .0074,
test-sampling CI half-width .02-.03 on Dead). Every claim must be a multi-seed, 3-split mean.

## 2026-09-26 — Failure mining on out-of-fold predictions (pillar B input; `scripts/mine_failures.py`)
OOF setting: split-2 CellViT-UNI predicting fold 1 (fold 1 = split 2's val fold, never trained on).
fold-1 OOF reference: mPQ .5019 / Dead PQ .180, missed_bg .104. Identical patterns on fold 3 test
(split-1 model), so the layout statistics generalise. Per-nucleus records:
`runs/analysis/failures_fold{1,3}.csv.gz`.

missed_bg rate by class x area bin (fold 1 / fold 3 agree):

| class | <60px | 60-100 | 100-200 | >200 |
|---|---|---|---|---|
| Neoplastic | .75 | .35 | .13 | .03 |
| Inflammatory | .68 | .18 | .05 | .01 |
| Connective | .77 | .40 | .18 | .06 |
| Dead | .76 | .38 | **.15** | **.12** |
| Epithelial | .75 | .37 | .09 | .02 |

Key layout findings (both folds):
- **Missed Dead nuclei are ISOLATED, not touching**: Dead with 0 touching neighbours miss .39/.42
  (n=863/882 = ~90% of all Dead); missed_bg Dead touch another nucleus in <5% of cases. The
  originally hypothesised "Dead glued to Neoplastic" layout is NOT the failure mode.
- **Merge is an intra-class cluster problem**: of merged Dead, 54/74 (fold 1) and 77/94 (fold 3)
  touch another Dead (karyorrhexis clusters); cross-class touching is rare overall in PanNuke
  (per-nucleus touching-class flags are almost diagonal).
- Size dominates everything (<60px miss .68-.77 for ALL classes), but large Dead are still missed
  .12-.25 vs .01-.06 for other classes -> appearance effect on top of size.

=> Layout sampler priorities for pillar B synthesis: (1) insert ISOLATED small nuclei (Dead-looking,
  60-200px) into empty stroma; (2) dense same-class clusters (Dead chains, Connective fields) to
  train separation; (3) cross-class touching de-prioritised (rare in the data and not the failure).

## 2026-09-26 — PixCell-256 Cell-ControlNet smoke test (20x generator on 40x PanNuke; `scripts/pixcell_sample.py`)
Released ControlNet + UNI2-h context embedding, real fold-1 instance masks as condition, fp32,
20 steps, guidance 2.5 (`runs/pixcell/smoke/smoke_fold1.png`, 6 patches x 2 seeds; UNI2-h
cosine(gen, real) .25-.77). Runs in ~10 s/patch on one A100 — large-scale synthesis is cheap.

Visual + quantitative review (440 mask objects, per-object fill/IoU, off-mask hallucination scan,
Laplacian/colour stats; details in the session log):
- **Placement is faithful and training-usable**: ~95% of mask objects get a nucleus at the right
  place/size/orientation (median per-object IoU .68); phantom nuclei off-mask are rare (36 slivers
  in 12 samples, mostly boundary artefacts). Tiny hyperchromatic dots — our failure case — are
  reproduced reliably.
- **Systematic under-rendering of LARGE masks**: 4.5% of objects dropped + 13% partial, and the
  failures are size-biased (dropped median 322 px vs 137 px for well-rendered). Using these as
  labels would under-supervise exactly the big atypical/clumped nuclei and teach "label extends
  past the visible object" at large-nucleus boundaries. (Same direction as our Dead finding: large
  Dead are missed .12-.25 — the generator is weakest where the segmenter is weakest.)
- **Appearance gap is structural, not just colour**: chromatin rendered as flat blobs/shells/rim
  (no granular texture or nucleoli; Laplacian variance 0.76x real, 0.45x on sharp patches),
  consistent pink->violet shift with higher saturation and milky background, flat magenta
  eosinophilic fills, mechanical cobblestone stroma. A stain-normalisation augmentation would NOT
  fix the chromatin/boundary part.

=> **Decision: LoRA-adapt the generator to 40x PanNuke (train fold only)**, with adaptation
targeting nuclear internal texture and boundary realism at least as much as colour; before/after
adaptation run the per-object fill-rate, hallucination and Laplacian metrics over a few hundred
stratified patches (Dead/large nuclei stratum first). At synthesis time: filter or regenerate
samples whose large objects under-render (fill-rate gate), stratify by size.

## 2026-09-26 (morning) — Pillar B round 1 executing: CP1 copy-paste control + PixCell LoRA pipeline
Two tracks launched in parallel (all training on LM2; per the pillar-B decision that Dead detection
needs data-level intervention):

**CP1 — failure-driven copy-paste (control arm; `src/nucseg/augment/copy_paste.py`)**
- Layout straight from failure mining: donors (50-400 px, class-weighted Dead 0.4 / others 0.15)
  pasted into nucleus-free stroma with an 8 px clearance (never touching, matching the "missed Dead
  are ISOLATED" finding); prob 0.5, Poisson(3) insertions per augmented patch, applied BEFORE the
  geometric/photometric augmentations so pasted nuclei are rotated/colour-jittered with the patch.
- Montage review (sonnet subagent, `runs/analysis/copy_paste_{noaugs,full}_v2.png`) caught real bugs
  in v1 that unit tests missed: donors clipped at the patch border became straight-edged fragments;
  pasting after augmentation let fakes keep raw colour/orientation (a learnable shortcut); occasional
  collision with UNANNOTATED nuclei (PanNuke annotation gaps) and gross hue mismatch. v2 fixes:
  border-clipped donors excluded from the bank, per-channel stroma colour shift (donor stroma ->
  destination stroma), Otsu-darkness guard against unannotated nuclei, paste-before-augment order.
  6 unit tests (`tests/test_copy_paste.py`): bank/class counts, isolation clearance, label
  consistency, donor-class mix, border-donor exclusion.
- Run `runs/cellvit_abl/split1_cp1` (LM2 GPU1, split 1, seed 19, CellViT-UNI, --cp-prob 0.5, TTA),
  ~3 h on a shared A100. Compare vs baseline seeds 19/1/2 (`scripts/compare_runs.py`). This is the
  decisive experiment for "is the Dead detection deficit data-fixable at all".

**PixCell LoRA — 40x adaptation (`scripts/train_pixcell_lora.py`, `src/nucseg/pixcell.py`)**
- LoRA (r=16, alpha=16) on the DiT attention projections only (8.3 M trainable / 616 M), ControlNet
  and VAE frozen, epsilon objective, CFG-condition dropout 0.1 implemented exactly like inference
  (dropped rows get zero controlnet contribution — the block injection is additive, verified in
  `pixcell_controlnet_transformer.py`). Conditions cached once per fold (VAE latents of image and
  binary mask + UNI2-h CLS embedding of the real image; `cache_fold{k}.npz`).
- Fold 1 (= split-1 TRAIN fold; no leakage) 5000 steps bs 16 lr 1e-4 (~40 min on a shared A100):
  `runs/pixcell/lora_fold1` (LM2 GPU7).
- Quantitative generator eval harness (`scripts/pixcell_eval.py`): 200 stratified patches
  (100 Dead-rich / 50 large-nucleus-rich / 50 random) re-generated from GT masks; per-object
  rendering scored by an OUT-OF-FOLD detector (split-2 CellViT-UNI, fold 1 = its val fold):
  coverage >= .5 rendered / .2-.5 partial / < .2 dropped, by size stratum and class; phantom rate,
  Laplacian variance ratio (global + in-nuclei), OD colour distance, UNI2-h cosine. Baseline
  (pre-LoRA) run on LM1 (`runs/pixcell/eval_base`); smoke numbers on 8 patches: Dead <100 px
  rendered .63 vs 1.0 for larger Dead, Laplacian ratio .68 in nuclei (texture deficit confirmed).
- Next when LoRA finishes: `pixcell_eval.py --lora runs/pixcell/lora_fold1/transformer_lora.pth`
  before/after comparison + montage review; then layout sampling at scale.

## 2026-09-26 (midday) — PixCell LoRA round-1 verdict: over-adapted; synthesis pipeline works, appearance not yet
- **Generator eval harness numbers** (`runs/pixcell/eval_base` vs `eval_lora16`, 200 stratified fold-1
  patches, paired context, same seeds; OOF split-2 detector): the r16/5000-step LoRA made generation
  WORSE — od_mean_l1 .098 -> .266, uni_cos .472 -> .327, in-nuclei Laplacian ratio unchanged .72,
  global Laplacian 0.95 -> 1.24 (speckle, not chromatin), Dead<100px render .650 -> .606, phantom
  px .0036 -> .0073. **Verdict: over-adapted (contrast/saturation blow-up).**
- **Synthesis smoke review** (32 images, LoRA + RANDOM context + bf16, sonnet subagent): geometry,
  density and label placement good (outlines on visible nuclei, phantom-labeled area 3.8%,
  teacher keep rate .886), but appearance failed: nucleus interiors crushed to near-black
  (median 42 vs real 121/255), per-patch bimodal colour (uniform violet or brick-red, saturation
  .40 vs .29), soft focus (Laplacian 3-10x low in 8/12 rows), milky fog / streak stroma, halo rims.
  10-20% of visible nuclei per patch left unlabelled (small, dark) — the known teacher-blind-spot
  caveat, confirmed visually.
- Diagnosis split: paired-context eval (above) isolates the LoRA as the main cause; random context
  amplifies it (off-manifold conditioning). Fixes launched:
  1. **LoRA round 2**: r8, lr 5e-5, 2500 steps, step-stamped snapshots every 250 (`lora_fold1_r8`,
     LM2 GPU7) — checkpoint selection by image-space metrics (od_mean_l1 / uni_cos / in-nuclei
     Laplacian on a 60-patch eval), NOT by the noisy epsilon-val (its trajectory was flat/noisy).
  2. **Paired-context synthesis** (default in `scripts/synth_pannuke.py`; random ctx now opt-in).
  3. Batched sampler verified equal to the reference pipeline (mean|diff| 0.015/255, fp32).
- `scripts/synth_pannuke.py` end-to-end: layout perturbation (2.38 inserted/layout) -> batched
  generation -> OOF teacher labelling with coverage>=.4 + type prob>=.6 -> keep-frac .7 gating;
  866/978 objects kept, 1/32 images dropped on the smoke batch.
- Training-side synthetic mixing implemented (`--synth DIR --synth-frac r`; synthetic samples carry
  the base patch's tissue label and the same cell+tissue sampler semantics).

## 2026-09-26 (afternoon) — CP1 copy-paste verdict: NEGATIVE (hurts Dead typing)
Run `split1_cp1` (seed 19, split 1, cp-prob .5, paste-before-augs, all v2 fixes) vs the 3-seed
baseline, paired tissue-stratified image bootstrap (`runs/analysis/compare_split1_cp1{,_val,_tta}.json`):

| fold | d mPQ [95% CI] | d bPQ [95% CI] | d Dead PQ [95% CI] |
|---|---|---|---|
| val 2    | +.0048 [+.0002,+.0090] | +.0002 [-.0018,+.0022] | +.0235 [-.0079,+.0652] |
| test 3   | +.0009 [-.0024,+.0043] | -.0010 [-.0031,+.0009] | **-.0310 [-.0576,-.0100]** |
| test 3 TTA | -.0004 [-.0035,+.0027] | -.0005 [-.0022,+.0011] | **-.0315 [-.0570,-.0116]** |

- The test Dead drop is real (>4x seed std .0074, CI excludes 0, TTA-consistent) and acts mostly
  through TYPING, not detection: among matched GT-Dead, correct type .690 -> .609; matched-Dead
  precision .512 -> .484; matched Dead 497 -> 484 (confusion matrix, test fold 3). Confusion moves
  both ways (Dead->Neo/Infla up AND Conne->Dead up) — a noisier Dead boundary, not a recall gain.
- The VAL fold pointed the opposite way on Dead (+.0235, n.s.): single-seed Dead PQ swings ~.05
  between folds. Third independent confirmation of the multi-seed/multi-fold rule.
- Interpretation: transplanting real-appearance donors (colour-shifted into random stroma) teaches
  "small isolated nucleus in clean stroma" as a Dead-ish cue and blurs the type boundary instead of
  improving detection. => Layout-level intervention with REAL appearance does NOT fix the Dead
  deficit; pillar B stands or falls with full synthesis (appearance generated in-context).
- mPQ is neutral everywhere (+.0009 test) — the pastes are not toxic to overall segmentation, the
  damage is Dead-specific.

## 2026-09-26 — PixCell LoRA round 2: abandoned; Reinhard colour fix instead
- r8 / lr 5e-5 / 2500 steps with snapshots every 250 (`runs/pixcell/lora_fold1_r8`, 60-patch
  image-space eval): EVERY checkpoint is worse than the base model (od_mean_l1 .60-1.10 vs base
  .116; uni_cos .13-.25 vs .496; even step 250 gives od .80). Round 1 (r16/5000, od .098 -> .266)
  was not a tuning accident — any DiT LoRA adaptation pulls generation off the 20x prior. Route
  abandoned; documented in the `train_pixcell_lora.py` header.
- Post-hoc colour fix instead: Reinhard LAB transfer of each generated image to its base real patch
  (`nucseg.pixcell.reinhard_lab`): probe on 24 pairs, od_mean_l1 .0723 -> .0094; mean saturation
  70.0 -> 68.9 (real 69.7). Synthesis recipe locked: BASE model + PAIRED context + Reinhard
  (`scripts/synth_pannuke.py` defaults; batched sampler verified equal to the pipeline, 0.015/255).
- SYN-v2 smoke (32 images, fold 1, that recipe): teacher keep rate .901 (729/809), 0 images
  dropped, 2.69 insertions/layout; colour vs paired real: median RGB |diff| ~3/255, saturation
  68.5 vs 69.6; Laplacian ratio .70 (the base model's texture ceiling — accepted).

## 2026-09-26 (evening) — SYN-v2 review: appearance PASS; Dead labels structurally ZERO (diagnosed)
- Sonnet montage review (12 rows, real|gen|label): every v1 defect fixed — patch median V 165-231
  (v1: 42), per-patch saturation within 1-3 of real, no milkiness, halos gone (dark-rim like
  real), 0/12 failure rows; placement 12/12 correct, 0/222 mask objects under-rendered by an
  inside-vs-ring contrast test; unlabelled-but-visible nuclei ~4-5% overall (worst row 25%).
  Residuals (accepted): nuclei somewhat darker than real on intrinsically dark patches (row 2),
  per-nucleus hue dispersion ~0.5x real, sharpness 0.45-0.69x real on high-frequency rows.
- **Blocker found by the review: 0 of 729 kept objects typed Dead** (expected ~60: ~11% of
  base-layout GT objects are Dead + 38% of the 86 inserted donors were Dead-class).
- Diagnosis (`scripts/diag_synth_dead.py`, same split-2 OOF teacher, same labelling code path):
  - REAL fold-1 Dead-rich patches: GT-Dead objects with coverage >= .4 are argmax-typed Dead 77%
    with MEDIAN max-prob .99 (97% clear the .6 gate) -> the teacher and the threshold are fine.
  - SMOKE synthetics re-read: P(Dead) = .000-.005 for ALL 729 kept objects (max .005; not a
    runner-up/confusion pattern, which would sit at ~.2-.4) -> the generated small nuclei simply
    carry no Dead cue.
- Structural cause: the ControlNet condition is a BINARY mask (class-agnostic) and the context
  embedding carries no per-nucleus class information; the generator's 20x prior renders isolated
  small nuclei as lymphocyte-like objects regardless of the donor's class. Class-targeted
  appearance is not expressible in this generator without re-training it (the abandoned LoRA
  route). Forcing donor-class labels anyway would recreate the CP1 failure mode (labels that do
  not match appearance).
- Consequence for pillar B: synthesis can add DETECTION supervision for small isolated nuclei
  with SELF-CONSISTENT (teacher-read) types — the CP1 harm channel (40% of pastes carrying forced
  Dead labels) is structurally absent, and the label/appearance pairs stay natural. Dead PQ can
  then only move through its detection factor. => SYN1 launched as the completing arm (3000
  images, fold 1, class-agnostic labels, synth-frac .5 first); if it also fails, pillar B closes
  with "small-nucleus detection is not data-fixable by real-appearance pastes or in-context
  synthesis".

## 2026-09-26 (night) — SYN1 verdict: null overall, Dead +1.0 PQ at the seed-noise edge; pillar B closes
- SYN1 = exact split-1 base recipe (seed 19) + 1146 synth (synth-frac .5) from
  runs/pixcell/synth_fold1 (3000 img, keep-rate .886); trained to completion on LM2, out
  runs/cellvit_abl/split1_synth1 (pulled to LM1 minus last.pth — /data7 hit 100%, 2026-09-26;
  full copy incl. last.pth still on LM2). Test fold 3, last ckpt:
  - mPQ .4900 / bPQ .6604 (base .4951 / .6676); TTA .4956 / .6659 (base .5013 / .6728). mPQ sits
    at the BOTTOM of the 3-seed range (seed1 .4910, seed2 .4924, seed19 .4951) — null-to-slightly-
    negative overall.
  - Dead PQ .1521 vs .1424 (TTA .1570 vs .1461): above all 3 seed draws (max .1446) but Dead seed
    noise alone spans .131-.142, so ~+1 pt is at the edge of noise. Decomposition: PQ+ FLAT
    (.2761 vs .2759), DQ+/SQ+ flat => the gain is correct-TYPE pairing (Dead-typed matches 367 vs
    352 of ~500 Dead GT), not better segmentation; Inflammatory PQ -1.0/-1.6 pts (possible
    small-nucleus synth side-effect).
  - Loss re-weighting dominates: npwce_small5 remains the best Dead arm (TTA Dead .1626, mPQ
    .4998) — better than SYN1 on BOTH axes; data-side levers (CP1 paste, SYN1 synthesis) add
    nothing on top.
- Verdict per the pre-registered rule: SYN1 fails to move the needle => pillar B closes: Dead /
  small-nucleus performance is not data-fixable via real-appearance pastes (CP1) or in-context
  synthesis with self-consistent labels (SYN1); the remaining lever is the loss (npwce_small5).

## 2026-09-26 (late) — Pillar C launched: cross-domain sets converted + reviewed; PanNuke-CF renderer validated
- **External datasets** (`scripts/prepare_external.py`, stored on LM2 `data/external/` — /data7 is full):
  - CoNIC2022 (HF MedOtter mirror, 4,981 patches): the 112 `source=pannuke` patches EXCLUDED
    (leakage guard); 20x -> 2x bilinear upsample (maps nearest) -> quad-split 256 -> **19,476
    tiles**; classes mapped Neut/Lymph/Plasma/Eos -> Inflammatory, Epithelial, Connective
    (Neoplastic & Dead absent). Single pseudo-tissue "Colon".
  - MoNuSAC (HF RationAI, official TEST split, 101 whole images 40x): RGBA->RGB (alpha verified
    opaque), Ambiguous instances dropped (2,403), tile 256 non-overlap, keep full tiles with >=1
    nucleus -> **443 tiles**; Epi->Epithelial, Lymph/Macro/Neut->Inflammatory; tissues
    Breast/Kidney/Lung/Prostate.
  - Montage review (sonnet subagent, GT-ellipse overlays): **no conversion bug** — annotations
    sub-pixel aligned (0.4-1.1 px), CoNIC-upsampled vs MoNuSAC-native nuclear scale within ~13%
    in the correct direction, upsampling interpolated not blocky. Noted: CoNIC has some
    mostly-acellular tiles + border-clipped nuclei chains (inherent to quad-split, documented).
- **Eval plumbing**: `evaluate()` gained an optional `tissue_names` (default PanNuke; externally
  readable per-tissue reports); `src/nucseg/data/external.py` mirrors PanNukeFold with a lazy
  per-image `gt_channels` (flat RAM); `predict_external.py` / `collect_external.py` for
  prediction and cross-split aggregation. Identity-prediction smoke = 1.0 PQ. Fixed a silent
  repo bug: `.gitignore`'s `data/` swallowed `src/nucseg/data/` — `pannuke.py`/`prepare.py`
  were NEVER tracked; now root-anchored `/data/` and committed.
- **Zero-shot predictions running** (LM2 GPU1): CellViT-UNI splits 1-3 x {conic, monusac} x
  {plain, TTA}. HoVer-Net after.
- **PanNuke-CF** (`scripts/render_pannuke_cf.py`): fixed fold-3 GT labels, factorial appearance
  intervention — context {paired, swapped-tissue} x Reinhard target {self, donor} = arms
  control/stain/ctx/tissue (+ `real` reference arm on the same 500 patches; 3 generator-seed
  replicates of control for the noise floor). 8-patch smoke (split-2 model): real .4735 ->
  control .4860 (generator renders EASIER images: bPQ .71 -> .80, but strict mPQ drops .47 ->
  .40 — generated nuclei carry type-inconsistent appearance), tissue arm .2807 (strong stress
  signal, missed_bg .07 -> .17). Smoke montage review forced two fixes: (a) blank/no-label
  patches excluded (context embedding is a CONTENT channel — hallucinated a whole tissue on an
  empty patch), (b) context donors luminance-guarded (±35 mean-L; near-white donor bleached a
  patch to 88% white). Full render (500 patches/arm) running on LM2 GPU7.

## 2026-09-27 — External zero-shot cross-domain (pillar C): CoNIC + MoNuSAC, 3-split mean
`scripts/predict_external.py` (LM2) -> `scripts/collect_external.py` (mPQ/bPQ/F_d 3-split mean +- std;
report files rsynced to LM1, `runs/analysis/ext_{cellvit,hovernet}.json`). HoVer-Net has no TTA row
(hardware budget; CellViT TTA answers the does-TTA-help question). In-domain reference from the
baselines table above: CellViT .4995 mPQ / .6654 bPQ, HoVer-Net .4564 / .6613.

| model | CoNIC mPQ / bPQ / F_d | MoNuSAC mPQ / bPQ / F_d |
|---|---|---|
| CellViT-UNI | .3351 +- .0037 / .5314 / .7816  [TTA .3398 / .5381 / .7858] | .2495 +- .0218 / .5616 / .7652  [TTA .2484 / .5650 / .7672] |
| HoVer-Net | .2610 +- .0061 / .5136 / .7669 | .0370 +- .0003 / .2756 +- .0701 / .3658 +- .0980 |

- **Domain drop (in-domain mPQ minus external)**: CellViT -.164 (CoNIC) / -.250 (MoNuSAC);
  HoVer-Net -.195 / -.419. Both degrade, but **HoVer-Net nearly collapses on MoNuSAC**
  (F_d .37, bPQ split std +- .070 vs +- .003 on CoNIC — per-split F_d ranges .26-.50; failure
  mode is missed_bg, i.e. detection, not typing).
- **TTA does not rescue cross-domain**: +.005 mPQ on CoNIC, -.001 on MoNuSAC (CellViT).
- **Per-class PQ (present classes only)**: CoNIC — CellViT Infla .425 / Conne .371 / Epith .246;
  HoVer-Net .339 / .255 / .193. MoNuSAC — CellViT Infla .309 / Epith .213; HoVer-Net .061 / .012.
  Epithelial is the weakest class on CoNIC for both models; Inflammatory transfers best.
- Classes absent in GT: CoNIC has no Neoplastic/Dead, MoNuSAC has no Neoplastic/Connective/Dead
  (mapping in the 2026-09-26 entry); strict mPQ on absent classes is excluded per official protocol.
- CF arms (real/control x3/stain/ctx/tissue on fold-3 500 patches) are predicted + evaluated for
  both models (2 splits each; `eval_ext_*` dirs) — paired bootstrap/rank-agreement analysis via
  `scripts/analyze_cf.py` is the next step.

## 2026-09-27 — PanNuke-CF analysis (pillar C): noise floor, paired deltas, rank agreement
`scripts/analyze_cf.py` on 4 evaluators (CellViT-UNI / HoVer-Net x splits 1/2; 494 images/arm,
fixed fold-3 GT; log `logs/analyze_cf.log` on LM2; PanNuke-test references synced to LM2).

- **Generator noise floor**: control-replicate mPQ std .0039-.0055 (range <= .0131) — an order of
  magnitude below the stress effects below, so arm deltas are attributable to the intervention.
- **real vs control** (same patches): CellViT +.042/+.051 mPQ (CI excl. 0) — but split by head:
  bPQ control .78 vs real .67 (generated nuclei are EASIER to segment), strict .36 vs .45 (their
  TYPE appearance is less consistent with GT labels). Both baselines show this; keep using mPQ +
  strict for CF arms.
- **Paired deltas vs control mean** (image-bootstrap 95% CI, all CIs exclude 0):
  - `stain` (Reinhard to donor stain): CellViT -.023/-.024, HoVer-Net -.061/-.071 — mild stress.
  - `ctx` (paired->swapped-tissue context): CellViT -.187/-.192, HoVer-Net -.135/-.144 — strong.
  - `tissue` (ctx + donor stain): CellViT -.189/-.193, HoVer-Net -.145/-.147 — strongest for CellViT.
  - Class structure: Epithelial collapses under ctx/tissue (CellViT -.41/-.42 PQ) while Dead loses
    -.04..-.10 — context swap mostly destroys type information, not detection (F_d only -.10).
- **Rank agreement (the pillar-C validation) FAILS at architecture level**: real domain drop
  (PanNuke test - mean ext mPQ) CellViT .21 vs HoVer-Net .30, but ctx CF drop CellViT .19 >
  HoVer-Net .14 -> Spearman **-1.00** across the 4 evaluator points (tissue -.60, stain +.60 ns).
  Interpretation: the CF stress axis measures **context/stain sensitivity**, which is NOT the axis
  that dominates real transfer — HoVer-Net's large real drop is its MoNuSAC *detection* collapse,
  and the less context-dependent (weaker) model is hurt less by context swap. Caveat: 4 evaluator
  points = 2 architectures x 2 near-identical splits, so the correlation is effectively the single
  architecture contrast; next step is a 3rd architecture (HoVer-NeXt-T per-fold weights) +
  PUMA (apoptotic->Dead) as an additional real external set before concluding.

## 2026-09-27 (eve) — HoVer-NeXt-T port decoded: flat validation decode + VAL-tuned thresholds (.458 mPQ)
- Port gap diagnosed with two ablation scripts (`hn_postproc_ablation.py`, `hn_thresh_sweep.py`,
  same raw maps scored under 3 decodings + a 5x5 flat fg/seed grid): the maps are FINE — the
  bottleneck was the decode. The vendored per-class inference-repo decode (their
  `pannuke_test_param_dict.json` thresholds) gives mPQ .365 on fold-3 400-img; their own
  **validation decode** (flat fg/seed + softmax-sum typing, `validation.py make_prediction`)
  gives .425 with their defaults (.7/.3); grid best .447. Their pp (hole fill + PANNUKE size
  filters) HURTS the flat decode (.425 -> .326) — dropped.
- The released 0.477 comes from `evaluate.py --tta 16`: 16 stochastic spatial+COLOR-aug views
  (inverse-transformed, averaged) + per-class fg/seed tuned ON THE TEST FOLD (`get_pp_params`).
  Not reproduced (stochastic color TTA at eval + test-tuned thresholds is their recipe, not a
  fair-comparison decode); we instead tune the flat pair on each split's VAL fold
  (`decode.json` per weight dir; .6/.7, .6/.7, .6/.6) — same protocol as our other baselines.
  (2026-09-28 audit: re-checked on LM2 — decode.json records sweep fold 2/1/2 for splits 1/2/3
  = the VAL folds; test folds 3/3/1 were never swept.)
- **Sanity, full test folds (3-split mean): mPQ .4579 / bPQ .6367 / Dead .1361**
  (paper .477/.656/.154; old port decode .3582/.5030/.080). Sits between HoVer-Net
  (.4564/.6613) and CellViT-UNI (.4995/.6654) on mPQ — a usable 3rd evaluator.
  Per-split: s1 .4423/.6247, s2 .4644/.6454, s3(test fold1) .4670/.6400.
- Notable behaviour (ERROR PROFILE, corrected 2026-09-28 audit — an earlier version of this line
  had the comparison inverted): HoVer-NeXt-T's flat decode merges ~2x MORE touching nuclei than
  the watershed decodes (merged GT rate .096-.101 vs CellViT-UNI .044 / HoVer-Net .049) and
  misses more background nuclei than CellViT-UNI (.122-.131 vs .102-.109; HoVer-Net .119-.125
  sits in between) — a different error profile, which is exactly what the rank-agreement
  analysis needs. (The "~.14" previously attributed to HoVer-Net was the merge rate of the
  HNXT per-class-decode ablations, .156-.177 in hn_pp_ablation.log, not of any in-domain
  evaluator.)

## 2026-09-27 (eve) — PUMA third external set (melanoma, apoptotic->Dead): converted, reviewed, predicted
- `prepare_puma.py` (Zenodo 15050523 ROI pack): 3,278 tiles, tissues Melanoma-primary/-metastatic
  (1648/1630); classes mapped nuclei_tumor->Neoplastic, lymphocyte/histiocyte/melanophage/plasma/
  neutrophil->Inflammatory, stroma/endothelium->Connective, **apoptosis->Dead (1,991 nuclei)**,
  epithelium->Epithelial (all 5 PanNuke classes present). Scale verified vs PanNuke fold-3 GT
  (median nucleus 390 vs 464 px; tif tags 0.226 um/px but effective scale behaves ~0.5 um/px).
- Sonnet montage review (GT outlines over 16 tiles): PASSED — 1-px contours boundary-exact,
  class colours coherent (black-stroke Dead confined to pyknotic/dumbbell nuclei, 1-3/tile;
  blue lymphocyte fields; orange spindle stroma), no conversion artifacts. Caveat inherited
  from PUMA (HoVer-Net-initialised, pathologist-corrected GT): ~5-15% of nuclei in the densest
  tiles carry no label -> absolute recall/mPQ on PUMA has a ceiling; model comparisons share
  the same GT so remain fair.
- Zero-shot (3-split mean, no TTA): **CellViT-UNI mPQ .4518 +- .0031 / bPQ .7177 / F_d .8831;
  HoVer-Net .3203 +- .0139 / .7136 / .8714**. Domain drop (in-domain mPQ - PUMA): CellViT
  -.048, HoVer-Net -.136 — ordering matches CoNIC/MoNuSAC (CellViT more robust), and unlike
  MoNuSAC there is NO detection collapse (F_d ~.87-.88): the transfer loss is in TYPING
  (CellViT strict .37 vs mPQ .45; Dead PQ per-arch mean CellViT .0024 / HoVer-Net .0326
  [.0211/.0209/.0559] / HoVer-NeXt-T .0046, Epithelial .09-.20). Apoptotic bodies
  remain undetectable-as-Dead cross-domain for both models — the Dead failure mode follows us
  out of domain (and PUMA's apoptotic GT is the only large external Dead supply we found).

## 2026-09-28 — Pillar C closes: 3-architecture CF verdict (inverted on mPQ, positive on the typing axis)
`hn_pipeline.sh` stage C+D added HoVer-NeXt-T everywhere (external x9, CF arms x2 splits);
`analyze_cf.py` extended with an axis decomposition + axis-matched rank agreement (exact
permutation p; `logs/analyze_cf3.log` on LM2, `runs/analysis/ext_hovernext.json` on LM1/LM2).

- **HoVer-NeXt-T external zero-shot (3-split mean mPQ/bPQ)**: CoNIC .2786/.4811,
  MoNuSAC .1123/.4025, PUMA .3813/.6888. Real domain drop (per-split in-domain reference,
  2-split mean): .199 — ordering CellViT .154 < HoVer-NeXt-T .199 < HoVer-Net .246,
  consistent across all 3 sets.
- **Headline rank agreement now has 6 evaluator points (3 architectures x 2 splits)**:
  ctx rho **-0.89** (perm p .033), tissue -0.89 (.033), stain +0.43 (ns). Architecture means
  (honest n=3): ctx/tissue **-1.00** — perfect inversion. The 2-architecture caveat is closed:
  more architectures made the inversion stronger, not weaker.
- **Axis decomposition (why)** — real drops split into detection (F_d) vs typing (bPQ-mPQ gap):
  | evaluator | real mPQ | F_d | typing | ctx CF mPQ | F_d | typing |
  |---|---|---|---|---|---|---|
  | CellViT-UNI (s1/s2)  | .155/.152 | **.007/.007** | +.092/+.084 | .187/.192 | .101/.103 | +.081/+.082 |
  | HoVer-Net (s1/s2)    | .242/.250 | **.149/.176** | +.076/+.062 | .135/.144 | .082/.086 | +.033/+.035 |
  | HoVer-NeXt-T (s1/s2) | .190/.207 | **.046/.043** | +.078/+.092 | .161/.164 | .092/.089 | +.062/+.066 |
  Between-architecture real differences live almost entirely on the **detection** axis
  (CellViT loses nothing, HoVer-Net loses .15-.18 = MoNuSAC collapse); typing-gap increase is
  similar for all (.06-.09). But CF rendering gives a **homogeneous** detection stress
  (.082-.103 for every architecture) — with labels fixed and the generator re-rendering
  well-formed nuclei, the detection axis cannot be stressed by construction.
- **Axis-matched rank agreement**: typing axis ctx rho **+0.71** (all 6 pts, perm p .136) /
  **+1.00** (arch means); tissue typing +0.83/+1.00. Detection axis: -0.83/-1.00 (CF has no
  variance there, so the correlation is meaningless-inverted). Stain arm's largest signal is
  also on F_d (+0.83, p .058 — HoVer-Net biggest on both sides), same story at weak signal.
- **Verdict (pillar C terminal)**: PanNuke-CF label-fixed counterfactual rendering is NOT a
  valid proxy for real cross-domain transfer — at the mPQ level it *inverts* the architecture
  ranking (the model most sensitive to context swap in-domain is the most robust transferrer).
  Mechanistically, its blind spot is exactly the detection axis, which is where real transfer
  differences concentrate. Where CF CAN apply stress (typing under context/stain mismatch), it
  ranks architectures consistently with real typing transfer. ⇒ report CF as a **typing
  stress test**, never as a transfer surrogate. Honest caveats: n=3 architectures (arch-mean
  |rho|=1 has exact p=.33; the defensible evidence is perfect sign inversion across 3 archs x
  2 splits plus the axis mechanism), external typing-axis differences are small (real typing
  gap range .069-.088 vs detection range .007-.176).

## 2026-09-28 (evening) — Benchmark artifacts: border fragments, the Uterus Dead subpopulation, cross-architecture consensus
`scripts/analyze_artifacts.py` over the per-GT records of our three evaluators (CellViT-UNI /
HoVer-Net / HoVer-NeXt-T; HoVer-NeXt-T + HoVer-Net split-2 eval dirs rsynced from LM2 to
LM1/`/tmp`). Consensus stats use splits 1+3 = test folds 3+1 (129,872 GT nuclei; rows verified
aligned across models: identical image/cls/area); per-model stats use all 3 splits. Frame
cached at `runs/analysis/artifacts_gt.csv.gz`. Survey of the same date (leaderboard / datasets
/ multimodal / generators): `docs/survey_2026_09.md`.

**1. Most "missed small nuclei" outside Dead are patch-border fragments.**
- 89% of GT nuclei < 60 px touch the 256 px patch border (31% of all GT do); 62% of < 60 px GT
  are missed by ALL 3 models (interior < 60 px: 33%).
- Border-touching objects are 57% of CellViT-UNI false negatives (missed_bg + missed_shape);
  40% of FNs are border AND < 100 px.
- Interior-only missed_bg: Neo .037 / Inf .027 / Epi .028 / Con .095 / **Dead .310** (Dead with
  border .351). ⇒ the stage-1 "small nuclei are missed for every class" observation is a
  border-fragment effect except for Dead; M2's < 100 px up-weighting mostly up-weighted
  fragments (its null result is unsurprising in retrospect).

**2. The Dead deficit is class-specific and shared by all three DENSE decoders.**
- Size–missed_bg curves coincide across ViT-L/16 / full-resolution ResNet-50 / ConvNeXt:
  < 60 px .73/.75/.76, 60–100 .33/.37/.33, 100–256 .11/.14/.14, ≥ 256 .03/.04/.05.
- Dead interior missed_bg by size (CellViT): .67 / .30 / .19 / .17 across the same bins
  (all-3-consensus .48/.18/.12/.13) — the appearance effect survives every size bin.
- Dead F_c (HoVer-Net def., 3-split means): CellViT-UNI .359 (.314/.322/.441),
  HoVer-NeXt-T .349 (.303/.322/.422), HoVer-Net .230 (.202/.213/.274). Reference points:
  HoVer-NeXt paper .49 (test-fold-tuned per-class thresholds + 16 stochastic TTA views),
  detection-first KongNet .59 [paper-reported, unverified]. ⇒ "the deficit is architectural"
  is narrowed to "shared across the dense decoders tested"; the detection-first family is
  untested under our strict protocol (phase-2 line A).

**3. Official Dead PQ is an image-subset metric; the split-to-split Dead gap is test-fold composition.**
- Official per-class PQ averages per-image PQ over images CONTAINING the class: Dead is scored
  on 97 images (fold 3) / 65 (fold 1); the top-10 Dead images hold 43% / 51% of all Dead GT.
- Fold 3 contains 34 Uterus Dead images (fold 1: 2). Their Dead PQ is .01–.05 for every
  architecture; interior Dead there are intraluminal clusters of pale shed cells (visual
  audit), missed by all 3 models on 69% of nuclei, median area 156 px (NOT small).
- Excluding Uterus images, Dead PQ per split (official → excluded):
  CellViT-UNI .195/.227/.231 (.142/.162/.224), HoVer-Net .133/.147/.175 (.103/.108/.170),
  HoVer-NeXt-T .143/.171/.197 (.102/.116/.191). Published CellVTA Dead (.145/.169/.240) shows
  the same fold-3 dip. ⇒ cross-split Dead comparisons (and any "Dead improved" claim) must
  control for image composition.
- Dead GT concentration (folds 3+1, n=2,024): Lung 54% + Colon 20% + Uterus 9%. Lung is the
  worst tissue overall (CellViT 3-split mPQ .402; 23% of its nuclei < 100 px); Colon has the
  highest merge rate (.089) and lowest bPQ (.570).

**4. No local-suppression halo; clustering hurts Dead.**
- Interior non-Dead nuclei within 40 px of a consensus-missed Dead are all-3-missed at .045 vs
  .033 far (baseline .027): no meaningful co-missing of neighbours.
- Dead in same-class clusters (≥ 3 Dead within 40 px; 58% of Lung / 78% of Colon Dead) are
  consensus-missed .237 vs .159 for scattered Dead (interior) — dense Dead fields are harder,
  consistent with the karyorrhexis-cluster merge finding, but isolation does not protect Dead
  either (stage-1 mining: isolated Dead miss .39–.42).

**Implications (phase 2, agreed with the user 2026-09-28):** report Dead four ways (official /
Uterus-excluded / interior recall / per-nucleus F_c); pre-registered endpoints for any Dead
intervention = interior Dead recall + Uterus-excluded Dead PQ with bPQ non-regression, always
3-split × multi-seed; the resolution hypothesis needs a detection-first or 2×-resolution test,
not a third dense decoder. Stage report 1 (EN+ZH) revised accordingly this date (wording +
bibliography corrections; see git).
