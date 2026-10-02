# bdd26 — PanNuke Nuclei Instance Segmentation (course project 5)

Nuclei instance segmentation + classification on **PanNuke** (~7,900 H&E patches, 19 tissues,
5 nucleus classes), evaluated under the **official 3-fold protocol**. On PanNuke, swapping
backbones moves mPQ by only 1–2 points (CellViT-HIPT .485 → SAM-H .498 → SOTA ~.51); the
persistent shortfalls are (1) the rare **Dead** (apoptotic) class — PQ .14–.19, worst of all
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
`docs/our_project5_nuclei_segmentation.pdf`.

## Status (2026-10-02)

> **P0–P3 v2, after the pre-merge review (2026-10-02):** In addition to the class-index and
> perimeter repairs, review found overwritten shared contacts across directions and lossy
> reconstruction of validation GT channels. Both were fixed; P0–P2 and bootstrap were rerun
> under the same validation-only menus. v1 artifacts/reports remain archived, not current results.
> The historical nine-run x2 grid is not nine distinct split/seed pairs: split2 used `{19,1,1}`;
> base coverage is 3/1/1. [Current results and limitations](docs/P0_P3_RESULTS_2026-10-02.md).
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

P2 is a candidate with small costs and fold-dependent gains, **not a proven overall win**.
The matched-seed experiment remains future work. P3's separate fixed-subset scale crossing
shows strong training-scale dependence; its numbers are not mixed into this test table.
Full methods, uncertainty and provenance: [P0–P3 results](docs/P0_P3_RESULTS_2026-10-02.md).

### Reports

- `docs/P0_P3_RESULTS_2026-10-02.md` — current audited results, inference limits, uncertainty,
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
  postproc/recovery.py training-free missed-nucleus recovery; M/S lever (merge + fragment drop)
  text/                CONCH prompt prototypes, radius-restricted nucleus embeddings, re-typing
  metrics/instance.py  fast PQ / AJI / AJI+ / centroid pairing (tested == official)
  metrics/errors.py    merge / split / missed / FP taxonomy, touching-neighbour counts
  metrics/pannuke_eval.py  official mPQ/bPQ + strict mPQ, mPQ+, per tissue/class, F_d/F_c, errors
  metrics/fast_retype.py   exact fast mPQ when only instance classes change (tuning on val)
configs/text/          nucleus prompt bank (LLM-written morphology descriptions)
docs/                  RESEARCH_PLAN.md (plan + progress log), findings.md (experiment log with
                       all numbers), report/ (stage reports EN+ZH + digests, PDFs),
                       survey_2026_09.md, course brief PDF — index: docs/
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
