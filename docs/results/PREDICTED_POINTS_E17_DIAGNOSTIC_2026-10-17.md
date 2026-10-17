# E17: sparse point-heatmap detector emits no candidates

**Decision: stop this exact recipe. No full-split development or large training.** The readout can still fit masks when supplied legal ground-truth (GT) points, but the learned point head did not produce a single candidate on the fixed 16-image cohort after 300 steps in either seed.

## Frozen scope

E17 kept E16's contextual conditional-mask readout, frozen E7 query features and mask basis, 8 fit images and 8 audit images, seeds 19 and 1, 300 cyclic single-image steps, and the unchanged native mask loss. It added a separate center head trained with a Gaussian point target, focal center loss and point-class loss. Inference used 3×3 local maxima with a fixed sigmoid threshold of 0.5, top-512 proposals, and the predicted point/class outputs only. The audit group contains 19 Dead objects and is a train-only diagnostic; all 16 images had already appeared in E7. It is not independent validation or test evidence.

## Results

| Seed | Fit GT-point mask recall, all / Dead | Audit GT-point mask recall, all / Dead | Predicted points across 16 images | Final predicted instances | Audit recall / precision / Dead recall / typed Dead F1 |
|---|---:|---:|---:|---:|---:|
| 19 | 86.73% / 84.21% | 67.98% / 31.58% | 0 | 0 | 0 / 0 / 0 / 0 |
| 1 | 85.71% / 80.26% | 74.70% / 42.11% | 0 | 0 | 0 / 0 / 0 / 0 |

Every one of the 32 seed-image evaluations had zero candidates. The A0 original-model audit was reproduced exactly for the selected images in both seeds. The point/mask fit loss matched E16's mask component, so the failure occurred before mask decoding: all predicted center probabilities remained below the frozen 0.5 proposal threshold. Lowering that threshold or adding steps would change the recipe after seeing the result, so this line is closed here.

## Interpretation and limits

The conditional mask readout remains trainable given a GT point; this does not show that a model can locate useful points. E17 directly tests the missing predicted-point stage and fails it. The new detector gate and its A0 guard both fail. No validation/test fold was read, no full-split run was launched, and no large training started.

Source SHA256 `834bfe022c9d27c2ec8da7a82fb94bc1b016f8c473fc3d8e57134437ac04579c`; frozen design SHA256 `de183198dedbec16f279abac55d596fbc4373e8734e27ddfbafa41271397e2b3`; result SHA256 `161962e950bdad55e2374c511980b493083afbfc32b228f2b954a5a5e466d6ef`. The result was independently checked: all candidates and predictions were zero, and the selected A0 rows match E11 exactly. Focused tests passed on LM1 and LM2; the portable E9–E17 suite passed 21 tests on LM1.
