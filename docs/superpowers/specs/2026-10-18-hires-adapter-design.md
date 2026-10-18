# E19 鈥?High-resolution CNN adapter on the CellViT-UNI baseline (frozen pre-registration)

**Frozen:** 2026-10-18 (America/New_York), before any E19 training. **Status:** new round; supersedes
none of the closed lines (E1鈥揈18, P1/P1-R1 stay closed). This is a bounded dev comparison, not a
method claim; no test fold is read.

## Question and single mechanism

Does adding a trainable **high-resolution CNN adapter to the finest decoder skip** of the existing,
working CellViT-UNI baseline improve **internal-Dead geometric recall** and **official Dead PQ**
while holding overall quality and staying within a bounded inference cost?

Rationale from existing evidence (no new claims): the current CellViT-UNI decoder's only
full-resolution path is `skip0 = conv_block(3,32)+conv_block(32,64)` on the raw input; all other
skips come from 16x16 ViT tokens. The project's Dead deficit is a detection-recall problem on
small/fragmented/faint objects (38.5% of Dead GT at max-IoU<.1; oracle +.156) and 89% of <60px GT
touch a border. High-resolution adaptation is the best-supported untested lever here (PFM-DenseBench:
resolution drives dense mIoU .620->.723->.753; CellVTA: a high-resolution adapter on frozen UNI
reaches the best-known Dead PQ .185). E5 already tested per-skip *channel gating* of the existing
skips and failed; E19 adds **new** high-resolution features, which is a different mechanism.

This is a single change. Everything else (encoder, skips, decode, heads, loss, data, optimizer,
schedule, sampling, seeds, decode, metrics, gates) is the frozen CellViT-UNI recipe.

## Arms (split1: fold1 train, fold2 dev validation; fold3 never read)

- **B0**: existing `CellViTUNI` baseline (fusion="plain"), retrained under the frozen recipe.
- **H** : `HiResCellViTUNI` = B0 with the finest skip replaced by `skip0(x) + hires(x)`.

Both arms are retrained from the same UNI initialization under an identical sample/augmentation
stream; B0 is not reused from a historical run.

## Frozen model change (`HiResCellViTUNI`)

Identical to `CellViTUNI` except `forward` uses `skips = (self.skip0(x) + self.hires(x), skip1,
skip2, skip3, z4)`. `hires` is a CNN adapter on the raw normalised input producing **64ch at
256x256**, exactly channel-compatible with `skip0`:

  `hires = Conv1x1(64,64)` applied to `conv_block(3,32) -> conv_block(32,64) -> conv_block(64,64)`

The final 1x1 conv is **zero-initialised** (weight and bias), so at initialisation `hires(x)=0` and
the model is functionally identical to B0. `conv_block` is the baseline's own (Conv-BN-ReLU-Dropout).
Construction order preserves the baseline: `CellViTUNI.__init__` runs first, then `self.hires`, so
every shared module keeps its initialization and RNG order; the encoder is constructed and loaded
first, as in B0. No other module is added, removed or re-ordered.

## Frozen recipe (both arms; unchanged from CellViT-UNI baseline)

split1, fold1->fold2; 130 epochs; `unfreeze_epoch=25`; batch16; workers8; AdamW lr3e-4
betas(.85,.95) wd1e-4; ExponentialLR gamma .85; `sampling_gamma=.85`; native 256; `hv_min_size=30`;
no TTA; **last checkpoint**; CellViT loss/weights unchanged. Seeds 19 and 1.
Guards: `hires_adapter=False` for B0 and `True` for H; the run refuses any other scientific flag
(cp/synth/np_wce/dead_w/small_w/dead_expert/widen/fusion!=plain/upscale!=1/visibility).

## Frozen endpoints and gates (primary arm H vs matched B0, fold2, last checkpoint)

Primary endpoint: pooled `internal Dead` (interior, majority-class Dead; current definition = 741
objects) max-IoU>.5 match rate, computed with the existing immutable geometric helpers and the
official evaluator `nucseg.metrics.pannuke_eval`.

1. **Main:** both seeds improve; two-seed mean Delta >= **+.02**; fixed 1000-draw tissue-stratified
   image-cluster paired bootstrap (seed **20261015**) 95% CI lower bound **> 0**.
2. **official Dead PQ:** both seeds not lower; two-seed mean Delta >= **+.005**.
3. **Floors:** per seed official mPQ, bPQ and strict Dead PQ >= B0 - **.002**; strict mPQ per seed not
   lower; merge rate <= B0 + **.002**. All five classes and 19 tissues reported with support; no
   subgroup adds a passing rule.
4. **Cost:** matched-device, batch1, warmed, repeated end-to-end inference (preprocess + encoder +
   decode + overlap) <= **2x B0**.

**Verdict:** all clauses pass in both seeds => `pass_primary`, permitting a separately recorded
three-split confirmation design. Any failure => `not_supported`: close this recipe; no extra seeds,
no threshold sweep, no added modules, no rescue.

## Provenance, resources, execution

- Provenance: code/config/input/weight SHA-256 bound per run and recorded in the manifest; fold3
  rejected before data/model access.
- LM1 authors code/tests and small analysis (ignored ops for large temps); **LM2** trains. Use the
  existing authorized LM1->LM2 worker broker only; no new SSH, no sync scripts, no proxy changes.
- Admission: >=2 idle LM2 GPUs (one per seed, run in parallel), each run <= 10h wall (long-run
  amendment precedent), total <= 40 GPU*h; refuse if output disk free < 52 GiB; GPU2 excluded;
  existing user/teammate jobs preserved.
- CPU full regression in the pinned env before GPU use; single-GPU smoke (batch16, finite loss and
  gradients, actual params, peak reserved, step time) before the formal queue; stop before formal
  training if smoke cannot meet budget.
- Detached queue with owned PID/events/logs; per-run fresh output dirs; no retry, no silent resume.
  Small JSON/CSV returned over the same connection; checkpoints stay on LM2.

## Limitations (frozen statement)

Two seeds; reused fold2 (dev signal only); image bootstrap conditional on the two trained seeds;
single adapter placement (finest skip only) and one adapter depth; no augmentation/architecture/
schedule search; cost measured on one device/load. Passing clause thresholds is evidence about this
bounded change only, not a method admission.

## Amendment (2026-10-18, before any run started)

- Bootstrap seed set to **20261015** so clauses 1-3 are computed by the already-frozen, tested
  `scripts/analyze_fusion_arms.py` unchanged (its `BOOT_SEED`, `MATCH_GAIN`, `GUARD`, `MERGE_GUARD`
  and `DEAD_GAIN` already match clauses 1-3 here). No gate value changes.
- Clause 4 (cost) is not covered by that analyzer; it is measured and reported separately (matched
  device, batch 1, warmed, end-to-end) as `cost_fold2.json` beside each run and compared H:B0.
- `scripts/predict_cellvit.py` now forwards the run's `hires_adapter` flag to `build_model` so the
  H checkpoint loads strictly.
