# E18: dense foreground-to-point voting still fails allocation quality

**Decision: stop this exact recipe. No larger training.** Dense votes avoid E17's empty sparse heatmap, but they produce hundreds of proposals per image and very low precision. GT-point mask fitting passes the predeclared fit witness in both seeds; actual predicted-point inference fails the frozen allocation gate and A0 guard.

## Frozen experiment

E18 retained E16's contextual conditional-mask readout, original frozen E7 query features/mask basis, 8 fit and 8 audit images, seeds 19 and 1, 300 cyclic single-image steps, optimizer and native mask loss. It replaced E17's sparse center map with an 8-channel dense head: foreground probability, two normalized pixel-to-instance-point offsets, and five class logits. Every GT foreground pixel, including singleton instances, supervised its legal GT point offset. At inference, foreground pixels above the fixed 0.5 threshold voted for rounded points; deterministic CPU float64 accumulation and stable top-512 selection produced candidates. GT points, instance masks or boxes were not passed to inference.

The 16 images were fixed before fitting: eight fit images and eight held-out-from-adaptation audit images. The audit set contains 19 Dead objects. Both groups were seen by E7, so this is train-only mechanism evidence, not independent validation/test evidence. E17's failed result was hash-bound as the detector control. A0 was reproduced per image before training.

## Results

GT-point readout fit (IoU > 0.5):

| Seed | Fit all / Dead / interior Dead | Audit all / Dead / interior Dead |
|---|---:|---:|
| 19 | 86.73% / 84.21% / 84.13% | 67.98% / 31.58% / 38.46% |
| 1 | 85.71% / 80.26% / 84.13% | 74.70% / 42.11% / 53.85% |

Actual predicted-point inference:

| Seed | Fit candidates / recall / precision | Audit candidates / recall | Audit precision | Audit Dead recall | Audit typed Dead F1 | Δ audit Dead recall vs A0 | Δ audit precision vs A0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19 | 3,334 / 34.18% / 3.67% | 3,262 / 26.48% | 3.67% | 15.79% | 0.71% | −5.26 pp | −13.15 pp |
| 1 | 2,899 / 41.84% / 4.26% | 3,233 / 30.04% | 3.52% | 26.32% | 2.09% | +21.05 pp | −15.06 pp |

Proposal totals are over eight images; the per-image decoder caps candidates at 512. Both seeds miss the fit-recall ≥80% clause of the frozen actual-inference gate. Seed 19 also loses audit Dead recall; seed 1's Dead-recall gain is rejected because audit precision falls by 15.06 percentage points. The A0 guard fails in both seeds. Final typed Dead F1 remains near zero.

## Interpretation

The sequence localizes the unresolved stage more clearly than another mask-readout variant would. E12 showed that free mask coefficients can reproduce many GT instances when the exact instance is assigned. E15–E16 show that a conditional readout can fit masks given GT points. E17's sparse point head emits no points; E18's dense vote head overproduces them, yielding only 3.5–4.3% fit precision and 3.5–3.7% audit precision. A mask representation alone does not solve calibrated instance proposal allocation. The two failed detector recipes bracket an empty-to-overcomplete proposal regime; this small probe does not identify an optimal threshold or justify tuning one after the fact.

The independent verifier rebuilt object coverage, detector counts, all four allocation deltas and A0 rows from the saved artifacts. A0 matches E11 exactly on the selected images. E18 exited 0 in 74.69 seconds with 1.43 GiB peak reserved memory on LM2 GPU0. The initial launch exited during setup because the precreated results directory violated the runner's empty-output requirement; that attempt used no model/GPU training. The final run used a fresh `results/run1/` directory. No validation/test fold, full-split development or large training was run.

Frozen code SHA256 `e39dbc0808a78ffccfefceeac6460b9a512e6f1cd07c8ccafa184ba01ba0700c`; design SHA256 `238955224fe9fc72d37d7bc48ebc960bc7b36c90e4a213d928c6114feebf484b`

Frozen test SHA256 `c6eee2ce12a9e1cb74a1de67bf85f8a68855fc667fe2b4e46ed76e470736dc41`; result SHA256 `f3f07c6a062a445bb20ce98c99d8688a4c34699e3bccf4fbb979242a25063fbc`. The independent verification record is stored with the local research output.
