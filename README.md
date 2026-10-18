# bdd26 — PanNuke Nuclei Instance Segmentation (course project 5)

> **2026-10-18 E19 in flight:** a pre-registered single-mechanism contrast -- a high-resolution CNN adapter on the finest decoder skip of the CellViT-UNI baseline (`skip0(x) + hires(x)`, zero-initialised so it starts as the baseline) -- is training on LM2 as four matched runs (B0/H x seeds 19/1, fold1->fold2, last checkpoint, no TTA). Spec `docs/superpowers/specs/2026-10-18-hires-adapter-design.md`; frozen gates in the spec. No result yet; fold3 never read. LM1 GPU2 unused.


**Current research status (2026-10-17):** E17's sparse point detector emitted zero candidates; E18's dense votes overproduced proposals and failed precision/A0 gates. GT-point mask fitting passes on fit images but does not solve proposal generation. No large training or independent E18 validation/test was run. See the [E7–E18 closeout](docs/results/QUERY_POINT_CLOSEOUT_E7_E18_2026-10-17.md), [E17](docs/results/PREDICTED_POINTS_E17_DIAGNOSTIC_2026-10-17.md), and [E18](docs/results/DENSE_VOTES_E18_DIAGNOSTIC_2026-10-17.md).

2026-10-17 E12/E13: GT-assisted fixed-feature fitting witness passes (Dead86.32/87.37%), not method admission; freezing mask projection fails both quality gates. E14 exact-native matching was completed later; exact matching did not pass the allocation or A0 guard. No large training. [E12](docs/results/FREE_MASK_E12_DIAGNOSTIC_2026-10-17.md), [E13](docs/results/STABLE_MASK_E13_DEV_2026-10-17.md).

E11 (2026-10-17): four-stage auxiliary supervision failed both original allocation and A0 guards (two seeds,300steps). Full CPU:512passed/3skipped. No large training. [Results](docs/results/AUX_QUERY_E11_DEV_2026-10-17.md).

> **2026-10-17 E10:** Predicted-mask constrained attention fails both train-only allocation gates and the separate A0 guard. Audit Dead recall M-G changes -.067511/+.008439; overall recall -.095944/-.056694. A0 reproduced exactly in both seeds before reusing E9 G. Stop this300-step recipe; no expanded training. Full regression507passed/3skipped. [E10 results and mechanism limits](docs/results/MASKED_QUERY_E10_DEV_2026-10-17.md).

> **2026-10-17 E9:** Local-content initialization fails the frozen train-only allocation gate in both seeds. L-G audit Dead recall gains are +.004219/+.018987, with precision losses -.027199/-.023349; fit recall .134321/.168889. Both adapted arms also underperform frozen A0 on audit geometry. Stop this recipe; no full train/val or large training. Current-source regression501passed/3skipped. [Full E9 review](docs/results/LOCAL_QUERY_E9_DEV_2026-10-17.md).

> **2026-10-17 E8a/E8b:** The frozen classifier-only probe fails allocation: fit foreground accuracy95.17%/97.58%, audit foreground changes-7.69/-6.87pp and final geometric recall gains+2.20/+2.47pp. Stop this balanced-head recipe. Separate original-model audit of all65Dead-positive fold1 images/967objects finds raw-mask matches35.57%/36.40%, final typed matches2.79%/12.20%. These are train-only diagnostics, not validation/test performance. [Full mechanism review](docs/results/QUERY_MECHANISM_2026-10-17.md).


> **2026-10-17 E7 review:** All four 130-epoch runs and final-checkpoint fold2 evaluations are complete. Slower decay does not rescue Q (two-seed mPQ .087958 vs matched B .472939). A new fixed-train-batch audit localizes substantial loss between assigned raw masks and query retention/overlap resolution; this is diagnostic, not method performance. The independent clean-GPU closeout is complete (exit0): all quality gates fail, interior-Dead matching change -.528340 [95%CI -.589426,-.468664]; cost passes (Q/B .738576/.764427). Preserve the failed recipe; no expansion. [Detailed E7 review](docs/results/SET_MASK_E7_DEV_2026-10-17.md).


> **2026-10-16 E6 closeout:** All four final-checkpoint fold2 runs and frozen analysis complete; Q fails every quality gate (mean interior-Dead matching change -.507422, 95%CI[-.563164,-.451449]) while passing latency. The train-only learning-rate restoration probe supports allocating a separate E7 gamma.99 schedule control, not a generalization or causal claim. [Results and next decisions](docs/results/SET_MASK_E6_DEV_2026-10-16.md).

> **2026-10-15 E5 closeout:** Both fusion-gating runs and fold2 evaluations are complete. Interior Dead matching changes −.032389/+.025641, mean−.003374 (95%CI [−.014273,+.007486]); the frozen primary gate fails despite mean Dead PQ+.017897 and passing quality floors. Stop this recipe: no adapter controls, extra seeds or confirmation. [Results](docs/results/FUSION_GATING_DEV_2026-10-15.md); [new train-only point-grid audit](docs/results/PMNET_POINT_GRID_AUDIT_2026-10-15.md). PM-Net remains proposed.

> **2026-10-15 tissue/class audit:** [Frozen-prediction analysis](docs/results/TISSUE_CLASS_AUDIT_2026-10-15.md) finds persistent differences across represented tissues. Lung, Skin and Colon rank in the bottom five across all three split-level seed means. Dead detection remains a valid targeted case, but 59.5% of Dead labels occur in Lung and seven tissues have none. Keep tissue/class reliability in scope; unseen-tissue transfer is untested and is not established as the dominant bottleneck. PM-Net remains proposed and subject to existing quality/cost gates. No new training or inference.


[Mechanism closeout](docs/results/VISIBILITY_SUPERVISION_MECHANISM_2026-10-15.md).

> **2026-10-15 update (America/New_York):** All eight visibility-supervision runs and original-image fold2 evaluations are complete. Frozen dev admission fails on original-image clauses 2 and 3: A2−A0 mPQ/bPQ losses exceed .002 in both seeds; seed1 strict Dead falls .004372; mean A2−A1 strict Dead falls .009781. All eight fixed-C2 mechanism evaluations are now complete. A1−A2 mean miss-rate improvement is −.008621 (95% CI [−.040918,+.017699]); the frozen mechanism clause also fails. No added seeds or confirmation. No additional seeds or three-split confirmation; PM-Net remains a proposal. See [full results](docs/results/VISIBILITY_SUPERVISION_DEV_2026-10-15.md). Earlier dated execution statements below are historical snapshots.


Nuclei instance segmentation + classification on **PanNuke** (~7,900 H&E patches, 19 tissues,
5 nucleus classes), evaluated under the **official 3-fold protocol**. On PanNuke, swapping
backbones moves mPQ by only 1–2 points (CellViT-HIPT .485 → SAM-H .498 → SOTA ~.51); the
persistent shortfalls are (1) the rare **Dead** class (apoptotic and necrotic) — PQ .14–.19, worst of all
classes — and (2) **merge/split errors on touching nuclei**. This project builds a strict,
leak-free evaluation pipeline, reproduces the standard baselines on it, and attacks the two
shortfalls along three "new-paradigm" pillars:

| pillar | paradigm | targets | verdict |
|---|---|---|---|
| **A. language-anchored typing** | pathology VLM (CONCH text space + LLM morphology prompts) | rare-class typing (Dead) | tested re-typing methods were null; substantial detection deficit remains |
| **B. failure-driven synthesis** | PixCell + Cell-ControlNet as a controllable tissue simulator | touching nuclei + rare-class examples | closed for the tested augmentation recipes |
| **C. counterfactual stress test** | same generator as an evaluation simulator | cross-tissue/stain robustness | a *typing* stress test, not a transfer proxy |

Research plan + full progress log: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md). Every
experiment number: [docs/findings.md](docs/findings.md). Course brief:
`docs/reference/our_project5_nuclei_segmentation.pdf`.

## Current status (2026-10-15)

The four-arm × two-seed visibility-supervision dev round and fixed-C2 closeout
are complete, using final checkpoints and validation fold2 only. Original-image
quality guards and the crop-mechanism criterion fail; no extra seeds, menu changes
or three-split confirmation. [Original-image results](docs/results/VISIBILITY_SUPERVISION_DEV_2026-10-15.md),
[mechanism report](docs/results/VISIBILITY_SUPERVISION_MECHANISM_2026-10-15.md).

Train-only [PM-Net geometry auditing](docs/results/PMNET_TRAIN_GEOMETRY_AUDIT_2026-10-15.md)
found maximum276 instances per image,23 rounded centers outside masks and116 empty
images. PM-Net remains a proposal; no new training. Current mechanism runner
passed52 focused checks; prior403-test full regression belongs to the training implementation.

Latest complete [progress PDF](Presentations/progress/PanNuke_Progress_2026-10-15.pdf)
and [project proposal PDF](docs/report/project_proposal/PanNuke_Project_Proposal.pdf).
Only current PDFs are tracked in these delivery directories; the complete PPTX is
available locally and ignored. Generators/QA/provenance build files live in ignored ops.

[Research closeout checklist and final-report analysis frame](docs/audits/RESEARCH_STATUS_2026-10-14.md).

E4's corrected full run is complete in `runs/analysis/visibility_probe_20261014/`
(code `988c0cd`); its code/input hashes, 2,523 windows per crop size and frozen gate
arithmetic were independently rechecked. All gates still pass: area-standardized
H1 +.1738 [.1060,.2528], H2 +.1474 [.1392,.1557] (interpolation −.0006), H3
+.8175 [.7968,.8393]. The prior train-mode run and crude H1 gate are superseded.
See [corrected results and limits](docs/results/E4_VISIBILITY_PROBE_2026-10-13.md).
The E4 gate originally permitted a [visibility-supervision design](docs/superpowers/specs/2026-10-14-visibility-supervision-design.md),
while training authorization is recorded separately above; it is not a method-performance claim. M's merged-class diagnostic
supports descriptive metric analysis (+.0519 for Inf/Conn); it leaves five-class
scores unchanged. Human E1 review remains unavailable; closed research lines stay closed.

## Prior status (2026-10-13)

**Stage report V is out** (experiments of October 9–12): see
[Chinese report](docs/report/stage_report_5/stage_report_5_zh.pdf),
[plain-language Chinese digest](docs/report/stage_report_5/summary_plain_zh.pdf),
[English report](docs/report/stage_report_5/stage_report_5.pdf) and
[sources and reproduction](docs/report/stage_report_5/README.md). In brief: the
typing-oracle ceiling (.4995→.6632 GT-assisted) is not cashable by post-hoc read-outs
(T0–T3 gate failed; Dead −.0053 tripped the guard-rail); E1-T's cross-detector jury
sides with the base label on 72.5% of base mistypes vs 18.2% for GT (intrinsic flip
rate 1.9%), meeting the pre-registered key for Inf/Conn label-boundary ambiguity as
detector-resonance evidence; E1-C finds the same resonance on the existence axis
(66–73% of zero-overlap candidates carry ≥3 cross-fold/cross-architecture support;
synthetic null 0.75%); E3 shows candidate utility is learnable (CV AUC .721/.737) yet
every learned row fails the frozen strict-Dead floor, closing the quality-learning
line by stop rule. Next: write label ambiguity into the final report's analysis
frame; an image-level class-absence-risk spec if that line is ever reopened; joint
training stays deferred; the E1 human-review obligation stands. No training this
stage; only the read-only oracle diagnostic touched historical test artifacts; the
suite passes 335 tests.

## Prior status (2026-10-12)

**Human expert review is confirmed unavailable for the foreseeable future (no resources,
user 2026-10-11).** E1's blinded two-stage human path therefore closes *unexecuted* —
the packet and protocol remain on disk and the obligation is recorded — and E1's record
of evidence stands on the executed no-expert amendment: E1-C cross-detector consensus
(null-controlled detector-resonance evidence, no biological claims) plus the E1-M model
pilot (process validation only; model review unusable). Two pre-registered follow-ups
were approved 2026-10-11 and run under frozen specs
(`docs/superpowers/specs/2026-10-11-*.md`): the **E3 candidate-quality dev round**
(five-arm menu replay over the frozen split1_seed1 fold2 candidates, gates frozen
before the run) and the **E1-T typing-consensus audit** (jury typing votes on base
mistypes, testing the Inf/Conn label-ambiguity explanation). Execution note: the
HoVer-Net split2 checkpoint was lost to disk cleanup on both machines, so E3's fold1
jury runs with six CellViT seats (disclosed in the round's provenance). No test fold
was read by either round. **Verdicts (2026-10-12, pre-registered keys):** E1-T finds
the 12-seat jury sides with the base model's label on 72.5% of base mistypes
(CI-separated) vs an intrinsic flip rate of 1.9% — detector-resonance evidence for
Inf/Conn label-boundary ambiguity (not a biological claim); the typing direction
closes with that explanation. E3 finds candidate utility is cheaply predictable
(CV AUC .721 low-dim / .737 with jury support) but every learned/area row fails the
frozen selection rule's strict-Dead gate, so all arms fall back to identity and both
pre-registered gates fail — the candidate-quality line closes at the dev stage;
official-metric gains exist descriptively only at a strict-Dead cost. See
`docs/findings.md` 2026-10-12 entries.

## Prior status (2026-10-09)

The [October 9 audit](docs/audits/RESEARCH_AUDIT_2026-10-09.md) rechecked frozen results,
corrected stale status text and hardened the remaining dev-fold/cost paths.
**E2a is complete; DSB is closed by its pre-registered Gate B stop rule**, and its
code was merged into `master` as tested tooling, not as a trained method.
**Reduced E0 is complete:** the two-model pipeline costs approximately **6.1× x1**
(image-weighted correction: 6.08447×; historical chunk-mean estimate: 6.08687×).
EF-P2 remains mechanism evidence / a teacher candidate, not a deployment win.
See [stage report IV](docs/report/stage_report_4/README.md) and its audit addendum.
The [October 9 research update](docs/proposals/RESEARCH_NEXT_2026-10-09.md) proposes a small
fixed-geometry instance-typing probe alongside E1 independent candidate annotation.
The typing oracle is GT-assisted and does not establish an achievable gain; its Inf/Conn
subset corrects all matched objects with those GT classes, not just pairwise swaps.
Utility learning remains conditional on candidate evidence. No new training is authorized.

## Research review (2026-10-06)

[Evidence audit and next-stage proposals](docs/proposals/RESEARCH_NEXT_2026-10-06.md) recheck all
108 frozen endpoint deltas, review recent primary literature, and propose controlled
crop/scale experiments and incremental-instance utility learning. The proposal includes
independent annotation/validation requirements and staged compute requests. The first
[HV target-threshold control round](docs/superpowers/plans/2026-10-06-hv-threshold-controls.md)
is approved: four arms × two seeds, split1 training/validation only. `--hv-min-size`
controls working-pixel target support; `--train-only` prevents automatic test inference.
Broader architecture proposals remain unapproved, and no new performance result is claimed.

## HV threshold controls complete (2026-10-07)

**The E2a HV target-threshold controls are complete** — 8/8 runs (x1/x2 × working cutoffs
30/8/120, seeds 19/1, split1, validation fold2 only), each with a hash-bound audit re-inference
whose endpoint differences from the original evaluation are exactly zero. Full tables,
diagnostics and limitations: [results document](docs/results/HV_CONTROLS_RESULTS_2026-10-07.md);
headline (2-seed means): lowering the cutoff at x1 does nothing stable; at x2 it raises
official Dead PQ in both seeds (+.0098, bootstrap CI crossing zero); at matched native support
x2 keeps the bPQ tax (CI below zero) and most of the Dead gain, while **strict Dead drops in
all four x2−x1 seed pairs**. Interior-Dead matched rate rises 61.3%→71.0% through scale, not
through the cutoff. Consequences: default `hv_min_size=30` stays; the pre-frozen DSB Gate A
arithmetic (x1 cutoff change reproduces 9.59% of the x2 Dead gain — below both the 50% and 90%
lines) keeps the planned dead-specialist experiments on an `x1_hv30` base without demoting the
line at that time. Subsequent Gates B/C/D are complete: Gate B closed the DSB line
before any expert-branch training; the code has since been merged into `master`.
The E2a and DSB experiments used train/validation folds only.

## Status (2026-10-05)

**The matched-seed and existence-filtered P2 rounds are complete.** The missing
runs were trained, giving nine distinct `(split, seed)` pairs across seeds
`{19,1,2}`. Plain P2 improves Dead PQ but fails its pre-registered bPQ guard.
EF-P2 adds a validation-selected area filter to the frozen P2 additions and passes
the same decision rule. Current three-split means, averaging three seeds per split:

| configuration | mPQ | bPQ | Dead PQ | strict Dead PQ |
|---|---:|---:|---:|---:|
| matched x1+M/S baseline | .49760 | .66530 | .17255 | .13073 |
| P2 | .49769 | .66443 | .17997 | .13363 |
| EF-P2 | .49828 | .66529 | .17954 | .13406 |

EF-P2 retains 94.3% of the P2 Dead gain: **ΔDead PQ +.00699**, with eight positive
pairs and one identity pair. Mean **ΔbPQ is −.000009**, versus −.000869 for P2.
This is a Dead-centred candidate improvement, **not proof of non-inferiority**:
mPQ/bPQ image-bootstrap intervals span zero, strict mPQ remains slightly lower,
test folds were reused across rounds, and dual-scale cost was unmeasured at this
October 5 milestone (E0 has since measured approximately 6.1× x1; see above).
Split1 and split2 share test fold3; the nine pairs are not independent patient cohorts.

**Stage report III:** [Chinese report](docs/report/stage_report_3/stage_report_3_zh.pdf),
[English report](docs/report/stage_report_3/stage_report_3.pdf),
[plain-language Chinese digest](docs/report/stage_report_3/summary_plain_zh.pdf).
[Sources and reproduction](docs/report/stage_report_3/README.md) include raw-summary
checks of all 108 endpoint deltas and the corrected validation border audit.

### Earlier audited standing (2026-10-02; seed19 only)

> **P0–P3 v2, after the pre-merge review (2026-10-02):** In addition to the class-index and
> perimeter repairs, review found overwritten shared contacts across directions and lossy
> reconstruction of validation GT channels. Both were fixed; P0–P2 and bootstrap were rerun
> under the same validation-only menus. v1 artifacts/reports remain archived, not current results.
> The historical nine-run x2 grid is not nine distinct split/seed pairs: split2 used `{19,1,1}`;
> base coverage is 3/1/1. [Current results and limitations](docs/results/P0_P3_RESULTS_2026-10-02.md).
>
> Corrected x1+M/S reaches **.4998 mPQ / .6656 bPQ**. Validation-gated additions reach
> **.4999 / .6652**, with Dead PQ **.1767 → .1835** and strict Dead PQ **.1340 → .1371**.
> This is a seed19 candidate, not a demonstrated overall or multi-seed win: split1 falls back
> to identity, only split3's Dead gain excludes zero in the paired image bootstrap, and bPQ
> and strict mPQ decline slightly. P3 reuses the unchanged 128-image validation probe, not a test benchmark.

Infrastructure and baselines are complete; both research stages ran to pre-registered
verdicts and are written up in stage reports I & II (PDFs in `docs/report/`).

- **Baselines reproduced under the official protocol** — HoVer-Net .4564 mPQ (paper .463),
  CellViT-UNI .4995 (CellViT++ reports .492); evaluation tested bit-exact against the official
  PanNuke-metrics code.
- **Stage 1 (three pillars)** — all closed with informative negatives, plus the key diagnosis:
  the baseline has a substantial Dead detection deficit (38.5% of Dead nuclei have maximum
  IoU < .1 in the original diagnosis), which survives the tested typing and augmentation
  configurations.
- **Stage 2 (Dead detection deficit)** — the tested detection-first recipe does not transfer
  (KongNet .346 mPQ with our evaluator); 2× working resolution improves interior Dead matching
  at a priced mPQ/bPQ cost, with official and strict Dead metrics giving different conclusions.

### Reproductions and released-weight re-evaluations (3-split mean)

Own models use final checkpoints and validation-only tuning. Released HoVer-NeXt/LKCell
validation-selected weights are reported separately by provenance; sharing the evaluator
does not make their training and checkpoint-selection protocols identical.

| model | mPQ | bPQ | reference |
|---|---|---|---|
| HoVer-Net (official fast mode, our training) | .4564 (.4666 TTA) | .6613 (.6714) | paper .463/.660 |
| CellViT-UNI (our training) | .4995 (.5047) | .6653 (.6706) | CellViT++ reports .492/.664 |
| HoVer-NeXt-T (ported, val-tuned flat decode) | .4579 | — | paper .477 |
| LKCell-L (released per-fold val-selected weights) | .4923 | — | paper .508/.685; cause of the gap is not isolated |
| KongNet (released per-fold weights, strict re-eval) | .3460 | .4706 | paper Dead F_c .59 → .34 |

Protocol rules used throughout: official splits (train/val/test = 1/2/3, 2/1/3, 3/2/1),
**last checkpoint for our training runs** (no test-fold early stopping), validation-only
parameter selection, and per-image bootstrap CI. Seed spread provides a cautionary scale
(mPQ std .0021, Dead PQ .0074 in the original split1 baseline), not a universal significance
threshold. Released checkpoint-selection differences are disclosed above. Released "all-data"
checkpoints are never evaluated as test results — they saw every fold.

### Stage 1 (09-25 → 09-28): three pillars, three verdicts

**A — language-anchored typing (CONCH): null, but it located the real bottleneck.**
CONCH radius-restricted attention pooling (R = 32 px, single forward pass) reaches .695
zero-shot balanced accuracy with Dead recall .88, yet every fusion into CellViT-UNI
(zero-shot / linear head / bias calibration, val-tuned) leaves test mPQ unchanged
(.4951 → ≤.4953); loss re-weighting, class-balanced sampling, and training-free recovery are
all null. **Diagnosis**: 38.5% of Dead GT nuclei have maximum IoU < .1 (other classes
8–15%), while 69% of *matched* Dead are typed correctly — detection is a major baseline deficit.

**B — failure-driven synthesis (PixCell + Cell-ControlNet): closed.**
Failure mining showed missed Dead are almost all isolated small nuclei and merges concentrate
in same-class clusters (karyorrhexis fragments). Copy-pasting real Dead nuclei *hurts* typing
(test ΔDead −.031); full in-context synthesis (base model + paired context + Reinhard stain
transfer, visually verified, 1,146 images) lands inside the seed-noise range (mPQ .4900), with
a structural blocker: the binary-mask ControlNet cannot condition appearance by class.
Neither tested recipe repaired the deficit; this does not exclude other data interventions.

**C — counterfactual stress testing: a typing stress test, not a transfer proxy.**
On real cross-domain zero-shot (CoNIC / MoNuSAC / PUMA), architecture differences sit almost
entirely on the detection axis (HoVer-Net F_d −.15–.18 on MoNuSAC vs CellViT −.007); the
domain drop orders CellViT .152 < HoVer-NeXt-T .199 < HoVer-Net .246. The current renderer
does not reproduce those between-architecture detection differences. CF-vs-real mPQ ranking
inverts (six-point ρ −.89), while the architecture-mean typing rank agrees (ρ +1.00).
There are only **three independent architectures** (exact architecture-mean p=.333);
these observations do not establish that label-fixed tests inherently cannot stress detection.

### Stage 2 (09-28 → 10-01): two attacks on the Dead detection deficit

**Line A — detection-first architectures: negative.**
KongNet (six-head CenterNet-style, released per-fold weights, fold mapping resolved
empirically, val-tuned decode) scores .346 mPQ — 11 points below the *weakest* dense decoder:
same-class touching nuclei fuse (merged rate .12–.18 vs .044–.049), interior-Dead miss is
*worse* (.50 vs .35), and the paper's Dead F_c .59 falls to .34. LKCell-L is a better
segmenter but a worse typer than our CellViT-UNI baseline (bPQ +.008, mPQ+ +.006, but mPQ
−.007, Dead PQ −.022). The evaluated released-weight recipes do not resolve the Dead deficit;
this does not exclude all detection-first designs or establish identical training protocols.

**Line B — 2× working resolution (512), with the October 2 audit.**
Scaled decode recovers much of the original bPQ tax, but the historical M/S and 3×3-grid
claims required correction. P0 reselected the repaired M/S on validation folds and added a
fair x1+M/S control. P1 found both usable scale complementarity and substantial false positives;
P2 tested a small, frozen menu of non-overlapping candidate additions. Current seed19,
three-split test means are:

| configuration | mPQ | bPQ | mPQ+ | Dead PQ | strict Dead PQ |
|---|---|---|---|---|---|
| CellViT-UNI baseline | .4995 | .6653 | .5178 | .1760 | .1330 |
| baseline + corrected M/S v2 | .4998 | .6656 | .5175 | .1767 | .1340 |
| x2 + scaled decode + corrected M/S v2 | .4906 | .6601 | .5150 | .1850 | .1249 |
| validation-gated additions (P2 v2) | .4999 | .6652 | .5200 | .1835 | .1371 |

P2 was a seed19 candidate with small costs and fold-dependent gains, **not a proven
overall win**. The subsequent matched-seed and EF-P2 rounds are now complete; see the
October 5 status and stage report III above. P3's separate fixed-subset scale crossing
shows strong training-scale dependence; its numbers are not mixed into this test table.
Full methods, uncertainty and provenance: [P0–P3 results](docs/results/P0_P3_RESULTS_2026-10-02.md).

### Reports

- `docs/report/stage_report_5/` — latest stage V (EN, ZH and plain-language ZH):
  typing-oracle ceiling, T0–T3 probe closure, E1 no-expert revision (E1-C/E1-M),
  E1-T label-ambiguity resonance, E3 utility-learning stop; verified numbers included.
- `docs/report/stage_report_4/` — stage IV (EN, ZH and plain-language ZH),
  E2a controls, DSB gate closure and E0 cost; the README links October 9 corrections.
- `docs/report/stage_report_3/` — stage III (EN, ZH and plain-language ZH;
  PDF/TeX, figures, numerical tables and verification snapshot). Covers October 2–5
  audits, matched-seed validation and EF-P2; see its README for reproduction.

- `docs/results/P0_P3_RESULTS_2026-10-02.md` — current audited results, inference limits, uncertainty,
  artifact paths and reproduction commands (`scripts/p0p3_numbers.py`).
- `docs/report/stage_report_1/` — Stage report I (EN, ZH and plain-language ZH digest).
- `docs/report/stage_report_2/` — Stage report II (EN, ZH and plain-language ZH digest).
  Both report sets retain historical bodies and include prominent October 2 v2 audit addenda;
  old M/S/grid claims and the superseded first-repair numbers are not relabelled as v2 experiments.

## Layout
```
src/nucseg/            our package (pip install -e .)
  constants.py         class / tissue names, official 3-split protocol
  data/prepare.py      stream-convert official float64 zips -> compact uint8/uint16 (37 GB -> 3 GB)
  data/pannuke.py      memory-mapped fold access (PANNUKE_ROOT overrides the location)
  data/external.py     CoNIC / MoNuSAC / PUMA as PanNuke-style folds (cross-domain zero-shot)
  hovernet/            official HoVer-Net (third_party) + our full-patch PanNuke plumbing, TTA
  hovernext/           HoVer-NeXt-T port (third evaluator / rank pool)
  cellvit/             CellViT-UNI (UNI ViT-L/16 + CellViT decoder), CellViT PanNuke recipe
  lkcell/              vendored LKCell network (UniRepLKNet-S + RepLK decoder; monai==1.3.2,
                       pip --no-deps) for strict re-evaluation of released per-fold weights
  kongnet/             KongNet decode for strict-protocol re-evaluation of released checkpoints
  augment/copy_paste.py   failure-driven copy-paste (pillar-B control)
  pixcell.py           PixCell + Cell-ControlNet sampling, Reinhard stain post-process (B/C)
  postproc/recovery.py training-free missed-nucleus recovery (du2 decode rescale + orphan blobs);
                       the M/S lever (merge + border-fragment drop) lives in scripts/lever_postproc.py
  text/                CONCH prompt prototypes, radius-restricted nucleus embeddings, re-typing
  metrics/instance.py  fast PQ / AJI / AJI+ / centroid pairing (tested == official)
  metrics/errors.py    merge / split / missed / FP taxonomy, touching-neighbour counts
  metrics/pannuke_eval.py  official mPQ/bPQ + strict mPQ, mPQ+, per tissue/class, F_d/F_c, errors
  metrics/fast_retype.py   exact fast mPQ when only instance classes change (tuning on val)
configs/text/          nucleus prompt bank (LLM-written morphology descriptions)
docs/                  RESEARCH_PLAN.md (plan + progress log), findings.md (experiment log with
                       all numbers), report/ (stage reports EN+ZH + digests, PDFs),
                       literature/ (survey), reference/ (course brief), results/, audits/,
                       proposals/, background/, superpowers/ (specs/plans) — index: docs/
scripts/               portable data, training, inference, evaluation and analysis CLIs
tests/                 pytest: metrics vs official code, decode scaling, copy-paste, recovery
third_party/           pinned upstream repos (scripts/fetch_third_party.sh)
data/ weights/ runs/ logs/   large local inputs and outputs, git-ignored
```

## Common commands

Run from the repository root in a Python 3.10 environment with the dependencies in
`env/requirements.txt` and this package installed (`pip install --no-deps -e .`).
The validated stack uses PyTorch 2.5.1+cu124 and NumPy 1.23.5; no machine-specific
bootstrap script is required by these commands.
```bash
python scripts/prepare_pannuke.py                       # data/raw/fold_{1,2,3}.zip -> data/pannuke
python -m pytest -q                                     # metric + plumbing tests
python scripts/train_hovernet.py --split 1 --out runs/hovernet/split1 --tta
python scripts/eval_pannuke.py --pred X.npz --fold 3 --out runs/.../eval
python scripts/train_cellvit.py --split 1 --out runs/cellvit_uni/split1 --tta          # ~2-4 h / A100
python scripts/predict_cellvit.py --run runs/cellvit_uni/split1 --fold 2 [--tta] [--upscale 2 --decode-u 2]
python scripts/encode_text_prototypes.py [--probe-fold 1 --crop 256 --calibrate]      # CONCH prototypes
python scripts/probe_conch_dense.py --fold 1                                        # radius-restricted pooling probe
python scripts/retype_conch.py --run runs/cellvit_uni/split1 --split 1 [--tta]       # pillar-A re-typing
python scripts/synth_pannuke.py ...                                                  # pillar-B/C synthesis (locked recipe)
python scripts/predict_external.py --run RUN --data conic [--tta]                    # cross-domain zero-shot
```
CONCH (text/image towers) is installed with `pip install --no-deps -e third_party/CONCH`
after fetching the upstream repositories with `scripts/fetch_third_party.sh`.
Official splits (train/val/test): 1 = 1/2/3, 2 = 2/1/3, 3 = 3/2/1. Always test the LAST checkpoint.

## Local-only workspace content

Git tracks portable project code, configuration, tests, research documentation and reports.
Machine configuration, SSH/sync/job orchestration and operational notes belong in local
`ops/`; machine-specific assistant instructions belong in `CLAUDE.local.md`. Both are ignored
and are not part of a GitHub checkout. Credentials, datasets, checkpoints and run outputs
also remain untracked. Keep portable experiment commands in `scripts/`, not in `ops/`.

This separation applies to the current tree and future commits; it does not remove files
from existing Git history.
