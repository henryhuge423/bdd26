# E19 — High-resolution CNN adapter on the CellViT-UNI baseline: dev result (2026-10-18)

**Status:** four matched dev runs complete (exit 0). **Verdict: `dev_admission = false`** — clauses
1 (interior-Dead matched rate), 2 (official Dead PQ) and 3 (quality floors) all fail; clause 4
(cost) passes. Per the frozen spec's rule, this recipe closes at this budget. This is a dev-fold
signal, not a test result; fold3 was never read and no method claim is made.

## Run identity

- Pre-registration: `docs/superpowers/specs/2026-10-18-hires-adapter-design.md` (frozen before the
  run; the only amendment was to set the bootstrap seed to the frozen analyzer's `20261015` and to
  note that clause 4 is measured separately). Plan:
  `docs/superpowers/plans/2026-10-18-hires-adapter.md`. Tests: `tests/test_hires_adapter.py`.
- Single mechanism: `HiResCellViTUNI` = `CellViTUNI` with the finest skip replaced by
  `skip0(x) + hires(x)`. `hires` is three `conv_block`s on the raw normalised input followed by a
  final 1x1 conv that is zero-initialised, so the model is functionally identical to the baseline
  at initialization. Nothing else (encoder, skips, decoder, heads, loss, data, optimiser, schedule,
  sampling, seeds, decode, metrics) changed. B0 is the same recipe without the adapter, retrained;
  no historical run is reused.
- LM1 authoring branch `research/hires-adapter-e19` (commits `9199f1e`, `75c5137`, `cef010a`).
  LM2 staging `ops/staging/hires_adapter_20261018` (scoped snapshot; data/weights/configs/
  third_party symlinked to the LM2 repo); bundle sha256
  `190128af11ff6c6e4e12937c39a2477f71c0710e4757b5a59296b4312ebc39be`.
- CPU: E19 focused tests 22 passed on LM1 and LM2. Full LM1 regression 568 passed, 5 failed — all
  `tests/test_dense_votes.py`, whose companion `tests/probe_dense_votes.py` is absent on master;
  they fail identically with master's files and are unrelated to E19.
- LM2 smoke (GPU5): finite losses/gradients for both arms; B0 362,934,749 params, H +60,800
  (`hires`); peak reserved 8.0 / 8.75 GiB; ~0.20 / 0.22 s per step.
- Runs: B0_seed19 (GPU3), H_seed19 (GPU5), B0_seed1 (GPU6), H_seed1 (GPU7); 130 epochs
  (`unfreeze_epoch=25`), fold1 -> fold2, native 256, **last checkpoint**, no TTA, `--train-only`.
  Wall seconds 7,751.8 / 8,370.2 / 7,869.3 / 8,312.2 (H ~7% slower). Final-epoch training losses
  were near-identical across arms (e.g. `np_ft` 0.0180/0.0181/0.0180/0.0179).
- Evaluation: fold2 (2,523 images) with `nucseg.metrics.pannuke_eval`. Interior-Dead endpoint = GT
  Dead objects not touching the patch border (n = **741** over **60** images), max-IoU > .5 against
  any predicted instance; miss = max-IoU < .1. Frozen paired analysis
  `scripts/analyze_fusion_arms.py` (tissue-stratified image-cluster bootstrap, 1000 draws, seed
  20261015, one shared draw), wall 588 s; `endpoints.json` and `cost.json` sit beside the runs.

## Endpoints (fold2, last checkpoint)

| endpoint | B0 s19 | H s19 | B0 s1 | H s1 |
|---|---:|---:|---:|---:|
| mPQ | .482912 | .482018 | .477467 | .477824 |
| bPQ | .652685 | .652607 | .650000 | .650644 |
| strict mPQ | .437327 | .434905 | .431388 | .430556 |
| Dead PQ | .168037 | .164576 | .163872 | .163786 |
| strict Dead PQ | .125134 | .135533 | .127456 | .133314 |
| interior-Dead matched | .635628 | .612686 | .587045 | .593792 |
| interior-Dead miss | .240216 | .273954 | .298246 | .296896 |
| merge rate | .047902 | .047151 | .047318 | .045781 |
| split rate | .010722 | .011157 | .011303 | .011072 |
| predicted instances | 59506 | 59422 | 59721 | 59791 |

## Gate outcomes (H minus matched B0)

- **Clause 1 (interior-Dead matched, >= +.02, both seeds, CI lower > 0): FAIL.** Per-seed change
  s19 **-.022942**, s1 **+.006748** (not both positive); mean **-.008097**; 95% CI
  **[-.022566, +.007378]**.
- **Clause 2 (official Dead PQ, both seeds not lower, mean >= +.005): FAIL.** Per-seed
  **-.003462** / **-.000086**; mean **-.001774**.
- **Clause 3 (per-seed floors >= -.002, strict mPQ not lower, merge <= +.002): FAIL.** strict mPQ
  s19 **-.002422** is below the -0.002 floor (s1 -.000832). The other floors pass: mPQ -.000894 /
  +.000356 (mean -.000269), bPQ -.000078 / +.000645, strict Dead PQ +.010399 / +.005858 (mean
  +.008128). Merge rate improves: -.000752 / -.001537 (mean -.001144).
- **Clause 4 (cost <= 2x B0): PASS.** GPU forward, batch 1, warmed (median of 96 samples/arm):
  B0 0.0320 s, H 0.0244 s, ratio **0.762**. Preprocessing and decode are identical across arms, so
  this bounds the end-to-end ratio above. Caveat: H measuring *faster* is not physically meaningful
  at this scale; the correct reading is "no measurable cost increase".

**Verdict:** `dev_admission = false`. Any clause failing closes the recipe, so no extra seeds,
threshold sweep, added modules, rescue, or three-split confirmation.

## Interpretation (dev scope)

Adding a trainable 64-channel high-resolution adapter at the finest decoder skip did not move
interior-Dead recall (slightly worse in seed 19, slightly better in seed 1; mean worse) and did not
raise official Dead PQ. Two secondary signals moved the other way but are not the primary endpoint:
merge rate fell slightly (mean -.0011, fewer merges in both seeds) and strict Dead PQ rose (mean
+.0081, positive in both seeds). The official-vs-strict divergence is the known definitional
difference, not a gain. Predicted instance counts were near-identical, so the adapter did not simply
change instance density. The net picture is a null result for this single mechanism at this budget:
extra full-resolution capacity at one skip did not convert into small/Dead-nucleus recall, consistent
with the project's standing finding that the Dead deficit is a detection/assignment problem rather
than a local-resolution one at this scale.

## Limitations (frozen statement)

Two seeds; reused fold2 (dev signal only); image bootstrap conditional on the two trained seeds;
single adapter placement (finest skip only) and one depth; no augmentation/architecture/schedule
search; cost measured forward-only on one device. Passing clause thresholds is evidence about this
bounded change only, not a method admission. Baseline B0 here is a freshly matched retrained CellViT-UNI
and is not comparable to historical test-fold numbers (e.g. the .4995 three-split test mean).

## Provenance

Run artifacts (LM2, `runs/hires_adapter_20261018/<run>/`): each has `config.json`, `final.pth`
(1.45 GB), `last.pth` (4.36 GB), `log.jsonl`, `pred_fold2.npz`, `eval_fold2/summary.json`, `run.exit`
=`eval=0`. Their SHA-256 values are recorded in `analysis/hires_adapter_20261018/endpoints.json`
under `sources` (`config`/`pred`/`eval`/`final` per run) together with the analyzer script hash
`ab2db460f8bc352a857f6d5337f5b51c14919d6f0155b8554f192ab21a41d681`. Checkpoints stay on LM2; the
analysis JSONs and cost JSON are the recorded artifacts returned this round. LM1 GPU2 was not used;
only the existing brokers were reused (no new SSH).
