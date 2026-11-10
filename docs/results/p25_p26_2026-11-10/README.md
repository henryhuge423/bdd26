# P25 results and P26 follow-up — 10 November 2026

P25 improved overall point detection and precision in both seeds, while its dead-nucleus recall remained below the prespecified requirements. We will not pursue this full-depth, layerwise-rate configuration further. P26 tests one new training constraint: keeping the original last-12-layer encoder near its pretrained parameters.

- [English analysis](RESULTS_EN.md) and [Chinese analysis](RESULTS_ZH.md)
- [Verified aggregates and original decision](analysis.json)
- [Image sensitivity and fixed-example object transitions](descriptive_sensitivity.json)
- [Raw results and 48 predetermined final images](raw/)
- [Rebuild the figures](make_figures.py) and [reproduce the results](reproduce_p25.py)
- [P26 design](../../superpowers/specs/2026-11-10-proposal-p26-design.md) and [implementation plan](../../superpowers/plans/2026-11-10-proposal-p26.md)
- [Stage 7 reports: English, Chinese and Chinese reader](../../report/stage_report_7/README.md)

From the repository root, use the pinned project Python environment:

```bash
python docs/results/p25_p26_2026-11-10/reproduce_p25.py
python docs/results/p25_p26_2026-11-10/make_figures.py docs/results/p25_p26_2026-11-10/analysis.json docs/results/p25_p26_2026-11-10/figures
python docs/results/p25_p26_2026-11-10/descriptive_sensitivity.py
```

The 114 recovered small files include 48 final PNGs. Their bytes and SHA256 values matched the preservation records; 485 sealed source files also matched. The public subset retains evaluation, fitting scores, logs, portable identities, sample coordinates and images, with its own hash inventory. Large checkpoints were checked against recorded sizes, but were not independently rehashed or externally archived in this review.

These are repeated fold-1 development comparisons of center points, using the final checkpoint. The audit comprises 522 images, 11 Dead-positive images and 145 interior dead nuclei; it is separate from the 2,134 fitting images. The results guide development, with official instance segmentation PQ and three-split test performance evaluated separately.
