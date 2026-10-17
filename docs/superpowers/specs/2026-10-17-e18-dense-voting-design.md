# E18 — Dense foreground-to-instance-point voting

**Frozen:** 2026-10-17, before the LM2 run. **Purpose:** test whether dense pixel-to-instance-point votes address E17's zero-candidate failure while keeping the existing conditional-mask readout fixed. This is a bounded train-only diagnostic, not a method result.

## Hypothesis

E17's sparse center heatmap produced no candidates at its fixed 0.5 threshold even though the GT-point mask readout fitted the fit images. E18 tests whether supervising every foreground pixel to vote for its instance's legal GT point yields a useful proposal set. The predicted proposal set must pass the predeclared fit, audit and A0 rules; mask fit with GT points alone cannot admit a method.

## Frozen inputs and training

- Use the same 16 fold1 images frozen for E15–E17: fit images `[273, 1105, 1394, 1058, 965, 2508, 2161, 1034]`; audit images `[1407, 88, 1177, 1170, 2511, 2471, 784, 2473]`. Each group has eight images. Both groups were seen by E7; audit means held out from this probe's adaptation only. It contains 19 Dead objects.
- Keep E16's conditional-mask readout, frozen E7 Q query feature and mask basis, legal GT point selection, and 5× native BCE + 5× Dice mask loss.
- Add a separate dense head, `64 → 64` 3×3 convolution + ReLU, then `64 → 8` 1×1 convolution. Channels are foreground, normalized `dy`, normalized `dx`, and five class logits. Initialize the foreground bias to `log(.1/.9)` and other final biases to zero; use seed+1701 for this head.
- For every foreground pixel, including singleton objects, predict the offset to that instance's legal GT point divided by 32 and its majority GT class. Do not apply the HoVer-Net area filter.
- Optimize the readout and dense head together for 300 cyclic single-image steps per seed, seeds 19 and 1, batch size 1, AdamW `3e-4`, betas `(.9,.999)`, weight decay 0. The dense objective is `2 × mean foreground BCE + per-instance mean L1 offset + 2 × per-instance mean class CE`; the conditional-mask objective is unchanged. Do not add steps or tune thresholds after seeing results.

## Inference and measurements

Select foreground pixels with sigmoid probability `≥0.5`. Each votes to `round(pixel + 32 × offset)`, clamped to image bounds. Sum foreground and class-weighted votes deterministically in row-major CPU float64. Keep 3×3 local maxima with at least 0.5 votes, stable descending-vote/row-major order, and top 512 candidates. Candidate class is the weighted class-vote argmax. Candidate confidence is the maximum foreground probability in its vote bin multiplied by its winning class fraction. Decode masks from predicted points and classes only, with E16's unchanged conditional mask readout, threshold `0.5`, and confidence-weighted pixel competition. Empty proposals produce empty masks. GT objects are used only for training targets and evaluation.

Measure GT-point readout fit as an explanatory diagnostic. For actual predicted-point inference, report object recall, precision, Dead recall, typed Dead F1, raw-mask Dead recall, and interior-Dead recall by fit/audit group. Reproduce A0 on the selected images per seed before fitting and require exact per-image equality to E11's A0 rows.

## Frozen gates and stop rules

On this new 16-image cohort (not the earlier 64/66-image gate), both seeds must satisfy: (1) fit actual-inference overall recall `≥0.80`; (2) audit Dead recall improves by `≥0.05` over A0; and (3) audit overall recall, precision and typed Dead F1 do not decrease. The separate A0 guard requires nonnegative audit changes in overall recall, precision, Dead recall and typed Dead F1. Any failure closes this exact recipe. No validation/test fold, threshold sweep, extra seed, extra step, or rescue modification is allowed. Passing would permit a separately recorded full train/validation design only; it would not authorize test evaluation.

## Execution and provenance

Run one LM2 GPU with at least 28 GiB free, at most 30% process memory fraction, 30-minute hard wall, at least 20 GiB free on data6, and a new data6 temporary/output directory. Preserve all unrelated jobs. Verify the source snapshot and all E9/E11/E12/E15/E16/E17 controls by SHA. Do not use LM1 GPU2 or alter pinned dependencies. Record code, design, result and control hashes, PID/start ticks, actual GPU, exit code and peak memory. Before training, verify the E17 controller gate is failed; do not proceed if the reference changes or A0 is not exact.

The frozen implementation and tests are [probe_dense_votes.py](../../../scripts/probe_dense_votes.py) and [test_dense_votes.py](../../../tests/test_dense_votes.py). This probe has no claim of a complete published detector method or independent validation performance.
