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
| **A. language-anchored typing** | pathology VLM (CONCH text space + LLM morphology prompts) | rare-class typing (Dead) | null — the bottleneck is *detection*, not typing |
| **B. failure-driven synthesis** | PixCell + Cell-ControlNet as a controllable tissue simulator | touching nuclei + rare-class examples | closed — not repairable from the data side |
| **C. counterfactual stress test** | same generator as an evaluation simulator | cross-tissue/stain robustness | a *typing* stress test, not a transfer proxy |

Research plan + full progress log: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md). Every
experiment number: [docs/findings.md](docs/findings.md). Course brief:
`docs/our_project5_nuclei_segmentation.pdf`.

## Status (2026-10-02)

Infrastructure and baselines are complete; both research stages ran to pre-registered
verdicts and are written up in stage reports I & II (PDFs in `docs/report/`).

- **Baselines reproduced under the official protocol** — HoVer-Net .4564 mPQ (paper .463),
  CellViT-UNI .4995 (CellViT++ reports .492); evaluation tested bit-exact against the official
  PanNuke-metrics code.
- **Stage 1 (three pillars)** — all closed with informative negatives, plus the key diagnosis:
  the Dead deficit is a *detection* problem (38.5% of Dead nuclei receive no overlapping
  prediction) and it survives language-anchored typing, real-appearance copy-paste, and
  in-context synthesis.
- **Stage 2 (Dead detection deficit)** — detection-first architectures do not transfer
  (KongNet .346 mPQ under the strict protocol); 2× working resolution improves every
  pre-registered Dead endpoint at a priced mPQ/bPQ cost — an explicit trade, not a default win.

### Reproduced baselines (3-fold mean, last checkpoint, val-only tuning)

| model | mPQ | bPQ | reference |
|---|---|---|---|
| HoVer-Net (official fast mode, our training) | .4564 (.4666 TTA) | .6613 (.6714) | paper .463/.660 |
| CellViT-UNI (our training) | .4995 (.5047) | .6653 (.6706) | CellViT++ reports .492/.664 |
| HoVer-NeXt-T (ported, val-tuned flat decode) | .4579 | — | paper .477 |
| LKCell-L (released per-fold weights, strict re-eval) | .4923 | — | paper .508/.685; uniform −.015 = protocol translation |
| KongNet (released per-fold weights, strict re-eval) | .3460 | .4706 | paper Dead F_c .59 → .34 |

Protocol rules used throughout: official splits (train/val/test = 1/2/3, 2/1/3, 3/2/1),
**last checkpoint** (no test-fold early stopping), all tuning on the validation fold,
significance by image bootstrap CI, and seed-noise rules (mPQ seed std ±.0021, Dead PQ
±.0074 ⇒ single-seed Dead deltas <.01 are untrustworthy). Released "all-data" checkpoints
are never evaluated as test results — they saw every fold.

### Stage 1 (09-25 → 09-28): three pillars, three verdicts

**A — language-anchored typing (CONCH): null, but it located the real bottleneck.**
CONCH radius-restricted attention pooling (R = 32 px, single forward pass) reaches .695
zero-shot balanced accuracy with Dead recall .88, yet every fusion into CellViT-UNI
(zero-shot / linear head / bias calibration, val-tuned) leaves test mPQ unchanged
(.4951 → ≤.4953); loss re-weighting, class-balanced sampling, and training-free recovery are
all null. **Diagnosis**: 38.5% of Dead GT nuclei get no overlapping prediction at all (other
classes 8–15%), while 69% of *matched* Dead are typed correctly — the loss is in detection.

**B — failure-driven synthesis (PixCell + Cell-ControlNet): closed.**
Failure mining showed missed Dead are almost all isolated small nuclei and merges concentrate
in same-class clusters (karyorrhexis fragments). Copy-pasting real Dead nuclei *hurts* typing
(test ΔDead −.031); full in-context synthesis (base model + paired context + Reinhard stain
transfer, visually verified, 1,146 images) lands inside the seed-noise range (mPQ .4900), with
a structural blocker: the binary-mask ControlNet cannot condition appearance by class. ⇒ the
Dead detection defect is not data-repairable by either route.

**C — counterfactual stress testing: a typing stress test, not a transfer proxy.**
On real cross-domain zero-shot (CoNIC / MoNuSAC / PUMA), architecture differences sit almost
entirely on the detection axis (HoVer-Net F_d −.15–.18 on MoNuSAC vs CellViT −.007); the
domain drop orders CellViT .152 < HoVer-NeXt-T .199 < HoVer-Net .246. Label-fixed
counterfactual renders cannot pressure the detection axis by construction, so CF-vs-real rank
agreement *inverts* on mPQ (ρ −.89, perm p .033) while agreeing on the typing axis (ρ +1.00).

### Stage 2 (09-28 → 10-01): two attacks on the Dead detection deficit

**Line A — detection-first architectures: negative.**
KongNet (six-head CenterNet-style, released per-fold weights, fold mapping resolved
empirically, val-tuned decode) scores .346 mPQ — 11 points below the *weakest* dense decoder:
same-class touching nuclei fuse (merged rate .12–.18 vs .044–.049), interior-Dead miss is
*worse* (.50 vs .35), and the paper's Dead F_c .59 falls to .34. LKCell-L is a better
segmenter but a worse typer than our CellViT-UNI baseline (bPQ +.008, mPQ+ +.006, but mPQ
−.007, Dead PQ −.022). ⇒ the Dead deficit is shared across decoder families; architecture
strength does not transfer into Dead/typing gains under a matched protocol.

**Line B — 2× working resolution (512): a priced trade.**
Every pre-registered Dead endpoint improves. The initial bPQ tax was traced to decoder
pixel-unit constants that do not scale with resolution and fixed (`--decode-u 2` recovers
61–70% of the bPQ tax, 76–88% of the mPQ+ gap); a validation-only post-processing lever
(same-class merge + thin-fragment drop) passes on test (+.002–.003 mPQ/bPQ). A 3-splits ×
3-seeds grid (9 runs) confirms the cost is robust (4.7–5.3× the seed std) and the Dead gain
is fragile (0.4–0.7×) — reported as an explicit trade-off:

| 3-fold mean | mPQ | bPQ | mPQ+ | Dead PQ | Dead F_c | interior-Dead miss |
|---|---|---|---|---|---|---|
| CellViT-UNI baseline | .4995 | .6653 | .5178 | .176 | .359 | .372 |
| + 2× res + scaled decode + lever | .4910 | .6608 | .5171 | .183 | .393 | .302 |

### Reports

- `docs/report/stage_report_1/` — Stage report I (three pillars; EN 12 pp, ZH 11 pp,
  plain-language ZH digest), PDFs compiled.
- `docs/report/stage_report_2/` — Stage report II (Dead detection deficit; EN 12 pp, ZH 10 pp,
  plain-language ZH digest《把图像放大一倍，AI 就能看见将死的细胞了吗？》), PDFs compiled.
  All tables re-derived from artifacts before writing (`scripts/report2_numbers.py`).

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
                       all numbers), report/ (stage reports EN+ZH + digests, PDFs), ops/
                       (machines + runbook), survey_2026_09.md, course brief PDF — index: docs/
scripts/               CLIs + machine sync (docs/ops/)
tests/                 pytest: metrics vs official code, decode scaling, copy-paste, recovery
third_party/           pinned upstream repos (scripts/fetch_third_party.sh)
data/ weights/ runs/ logs/   large, git-ignored; LM1 is the source of truth
```

## Common commands (LM1, env `~/.conda/envs/nuclei`)
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
(done by `lm_env_setup.sh` / `ugrad_bootstrap.sh`).
Official splits (train/val/test): 1 = 1/2/3, 2 = 2/1/3, 3 = 3/2/1. Always test the LAST checkpoint.

## Machines & ops

Machines, code/data sync (`scripts/push_ugrad.sh` / `pull_ugrad.sh` / `push_lm2.sh`), the
ugrad L4 constraints, and known operational gotchas live in **[docs/ops/](docs/ops/README.md)**.
