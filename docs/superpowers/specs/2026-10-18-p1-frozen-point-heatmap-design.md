# P1 — Frozen-feature point-heatmap probe (bounded train-only diagnostic)

**Frozen:** 2026-10-18, before the LM2 run. **Purpose:** test whether a lightweight point
(center-heatmap) head trained on **frozen** pathology-foundation-model features can produce
useful instance proposals — specifically, whether internal-Dead point recall materially exceeds
the current CellViT-UNI baseline's NP-foreground proposals — on a **fresh full fold1 → fold2**
cohort. This is the route document's P1 gate
(`docs/proposals/RESEARCH_NEXT_2026-10-14-new-model-route.md`) and the E7–E18 closeout's
"predeclared proposal-count/precision gate on a fresh train-only cohort". It is a bounded
diagnostic, not a method result; GT-point fit scores are never reported as detector performance.

## Why this probe, and what it is not

E7–E18 closed the trained query-mask line: the repeated bottleneck was instance proposal
allocation and calibration; E17's sparse heatmap emitted zero proposals and E18's dense votes
produced thousands at ~3.5% precision. Those probes ran on **frozen features of the failed E7 Q
model** and on 16 images. This probe asks the cleaner upstream question: with a strong,
leak-clean frozen backbone and a **full-training-fold** fit, is the point/heatmap paradigm
viable at all? Failure here ⇒ do not enter PM-Net training (route doc §7). This probe does not
test masks, classes, HV, watershed, typing, visibility or augmentation, and it reads **no test
fold**.

## Frozen cohorts

- **Train:** fold1, all 2,656 images (never used to train any earlier point head; the E-series
  16-image cohort is not reused).
- **Eval (dev validation):** fold2, all 2,523 images. fold2 is reused across the project; all
  outputs are development signals only. fold3 is never read. 65 fold1 and 70 fold2 images
  contain Dead pixels.
- Instances: every GT id > 0 in `inst.npy`, all classes, **no area filter**. Interior = does not
  touch the 256-px outer ring (the frozen `original_full` rule). Internal Dead = interior and
  majority-class Dead.

## Frozen inputs (SHA256, asserted by the probe)

- `data/pannuke/fold1/images.npy` `b4f24c4d117a44fa930120d33b52175dc8189afdceb89a2cdf91d2682293105b`
- `data/pannuke/fold1/inst.npy` `59fb7dc60354e54e642f41072595ee3a4bec700c0b36bd2c94075ae030e5de7e`
- `data/pannuke/fold1/type.npy` `cda88107c0860fd3be6c58d9dafde7623b9709ee639649246e67c4844ae593c0`
- `data/pannuke/fold2/images.npy` `d7c93f1734d218e1313d276ab2c3c02130289e3076108f724324d9f71860aa39`
- `data/pannuke/fold2/inst.npy` `145420bb21d05d5047ecc6d0d043ef05d6a2789c608fe987488480d16f7b7c21`
- `data/pannuke/fold2/type.npy` `82d09f44f79575fd8629b02495c25dca2ddce12b49c25191e3d084f473f57077`
- `weights/MahmoodLab/UNI/pytorch_model.bin` `56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40`
- `weights/MahmoodLab/UNI2-h/pytorch_model.bin` `6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e`
- baseline `runs/cellvit_uni/split1/final.pth` `da9a7efddf9cc53ff4fa18160f5bf51f11ab866e51abb16c1104bd55be599bcb`
- Source snapshot: `snapshot/src/nucseg/{__init__.py, constants.py, cellvit/__init__.py,
  cellvit/model.py}` (file hashes recorded in the launch receipt and manifest, verified against
  LM1 master afterwards).

## GT rule (frozen replica)

Per instance: class = majority type vote over the instance's pixels; point = the instance pixel
nearest the mask centroid, first occurrence in row-major order (the frozen `make_targets` rule);
interior = no pixel on the 256-px outer ring. Instances without a foreground vote are an error.

## Frozen backbones and head

- **Primary arm: UNI2-h** (frozen; resize 256→252 bilinear, ImageNet mean/std; timm
  `vit_giant_patch14_224` kwargs with `reg_tokens=8, no_embed_class=True, SwiGLUPacked`; features =
  block outputs (6,12,18,24); patch grid 18×18, width 1536).
- **Secondary arm: UNI** (frozen; native 256; repo `UNIEncoder` weights; layers (6,12,18,24);
  grid 16×16, width 1024). Reported descriptively; gates apply to the primary arm only.
- Head (identical recipe per arm and seed): each layer projected by 1×1 conv to 192 ch →
  concatenated → 3×3 conv 256 + ReLU → three ×2 nearest-upsample + 3×3 conv blocks
  (256→192→128→64) + ReLU → 1×1 conv → 1 logit; final bilinear resize to 256×256. No
  normalization layers. Output bias initialized to log(.01/.99).
- Features normalized per layer by fold1-only mean/std (saved with results).
- Target: E17 frozen construction — max-combined Gaussian patches (radius 3, σ=1) at the GT
  points on the 256×256 grid, exactly 1 at point cells. Loss: E17 penalty-reduced focal
  (pos `-(1-p)²·log p`; neg `-(1-Y)⁴·p²·log(1-p)`; sum / max(1, #objects)).
- Optimization: AdamW lr 3e-4, betas (.9,.999), wd 0; batch 16; **10 epochs** (fold1 shuffle,
  deterministic per seed); no schedule, no augmentation (features precomputed once), no model
  selection — the final step is evaluated. Head seeds **19 and 1** (init + shuffle only; the
  feature caches are shared).

## Frozen decode and metrics (fold2)

- **D1 (primary, gated):** sigmoid map; 3×3 local maxima with p ≥ 0.5 (plateau-inclusive),
  stable descending probability then row-major, top 512. CPU float64.
- **D2 (secondary, reported):** threshold-free greedy 3×3 NMS, top 100.
- Hit: GT instance point within Euclidean distance ≤ 3.0 px of a candidate (squared ≤ 9).
- Recall(all) / recall(internal Dead) / candidate precision (fraction of candidates with
  `inst > 0`) / candidates per image; radius table {1,2,3,5} px and D2 recall reported
  descriptively.
- **Baseline (measured in-run):** CellViT-UNI `final.pth`; NP softmax ≥ 0.5 → 8-connected
  components (scipy); candidate per component = its pixel nearest the component centroid (same
  frozen rule); same metrics.

## Frozen gates (primary arm, both seeds must pass all four)

1. Δ recall(internal Dead, D1) = probe − baseline ≥ **+0.10** (pooled fold2).
2. Paired image-level, tissue-stratified bootstrap (1,000 draws, seed 20261018) on Δ
   recall(internal Dead, D1): 95% CI lower bound **> 0** (per seed, images containing internal
   Dead).
3. D1 candidate precision ≥ baseline precision − **0.10** (pooled, per seed).
4. Non-degeneracy: ≥ **95%** of internal-Dead-positive fold2 images have ≥ 1 D1 candidate
   (per seed).

**Verdict:** all four clauses in both seeds ⇒ `pass_primary` — permits a separately recorded P2
(full train/validation) design; no test evaluation and no method claim. Any failure ⇒
`not_supported`: the tested point-head paradigm closes at this budget (route doc §7: stop, do
not enter training). No threshold sweep, extra epochs, extra seeds, or rescue modifications may
be added after seeing results; D2 and the radius table are diagnostic only.

## Execution and provenance

One LM2 GPU with ≥ 28 GiB free at launch (lowest free index, recorded UUID), process memory
fraction 0.30, controller `timeout 3600` with an in-script deadline (3,300 s), ≥ 50 GiB free on
data6, fresh tmp/output dirs (refuse pre-existing). No LM1 GPU, no dependency changes, no test
fold, no other users' jobs touched. The manifest records code/design/input hashes, versions,
GPU, peaks and elapsed; small JSONs are returned over the existing connection; large feature
caches stay on LM2. Failure stops with outputs preserved; no automatic retry.

## Limitations (frozen statement)

Two head seeds, reused fold2, no augmentation, head-only training on frozen features, image
bootstrap intervals conditional on the two trained seeds, and small Dead support (~70 fold2
images) limit inference. Baseline NP components are a detection-branch proxy, not the full
baseline pipeline. Passing clause thresholds is evidence about the proposal stage only.
