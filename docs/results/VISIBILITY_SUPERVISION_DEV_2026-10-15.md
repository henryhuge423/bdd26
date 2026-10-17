# Visibility-supervision dev result — 2026-10-15

Eight of eight training runs and original-image fold2 evaluations are complete. All runs use split1, fold1 training/fold2 validation, seeds 19/1, 130 epochs, final checkpoint, x1 and no TTA. The queue records successful training and validation completion for each run. This audit reads the eight existing summary files and queue records; it does not rerun inference or independently revalidate checkpoint bytes.

## Original-image validation results

| Arm | Seed | mPQ | bPQ | Dead PQ | strict Dead PQ |
|---|---:|---:|---:|---:|---:|
| A0 | 19 | 0.4840002 | 0.6536900 | 0.1610816 | 0.1174554 |
| A1 | 19 | 0.4767094 | 0.6464515 | 0.1934476 | 0.1253827 |
| A2 | 19 | 0.4748876 | 0.6474460 | 0.1833232 | 0.1199311 |
| A3 | 19 | 0.4789208 | 0.6468693 | 0.1976473 | 0.1182505 |
| A0 | 1 | 0.4784470 | 0.6499801 | 0.1575460 | 0.1239126 |
| A1 | 1 | 0.4751700 | 0.6449741 | 0.2004756 | 0.1336504 |
| A2 | 1 | 0.4750214 | 0.6453492 | 0.1844342 | 0.1195407 |
| A3 | 1 | 0.4722106 | 0.6439357 | 0.1800388 | 0.1177824 |

## Frozen comparisons

| Comparison | Seed | ΔmPQ | ΔbPQ | ΔDead | Δstrict Dead |
|---|---|---:|---:|---:|---:|
| A1-A0 | 19 | -0.0072908 | -0.0072385 | +0.0323660 | +0.0079274 |
| A1-A0 | 1 | -0.0032769 | -0.0050060 | +0.0429296 | +0.0097378 |
| A1-A0 | mean | -0.0052839 | -0.0061222 | +0.0376478 | +0.0088326 |
| A2-A0 | 19 | -0.0091126 | -0.0062441 | +0.0222416 | +0.0024757 |
| A2-A0 | 1 | -0.0034256 | -0.0046309 | +0.0268883 | -0.0043719 |
| A2-A0 | mean | -0.0062691 | -0.0054375 | +0.0245649 | -0.0009481 |
| A2-A1 | 19 | -0.0018218 | +0.0009944 | -0.0101244 | -0.0054516 |
| A2-A1 | 1 | -0.0001487 | +0.0003751 | -0.0160414 | -0.0141097 |
| A2-A1 | mean | -0.0009852 | +0.0006848 | -0.0130829 | -0.0097807 |
| A3-A1 | 19 | +0.0022114 | +0.0004178 | +0.0041997 | -0.0071322 |
| A3-A1 | 1 | -0.0029595 | -0.0010385 | -0.0204368 | -0.0158680 |
| A3-A1 | mean | -0.0003740 | -0.0003103 | -0.0081186 | -0.0115001 |
| A2-A3 | 19 | -0.0040332 | +0.0005766 | -0.0143241 | +0.0016805 |
| A2-A3 | 1 | +0.0028108 | +0.0014135 | +0.0043955 | +0.0017584 |
| A2-A3 | mean | -0.0006112 | +0.0009951 | -0.0049643 | +0.0017194 |

## Admission decision and research interpretation

The conjunctive dev admission rule **fails**. Clause 2 fails: A2−A0 mPQ/bPQ losses exceed .002 for both seeds, and seed1 strict Dead decreases. Clause 3's mean strict Dead floor also fails for A2−A1. The mean A2−A0 official Dead gain exceeds .005, but cannot compensate for these failures. No confidence intervals or significance claims are inferred from two seeds.

A1−A0 increases official Dead in both seeds while losing mPQ/bPQ. A2−A1 decreases official Dead and strict Dead in both seeds. Thus these original-image endpoints provide no incremental performance support for the selective HV intervention; they do not establish its effect on newly truncated Dead misses. A3 also fails the practical baseline quality floors, and A2/A3 equivalence is not tested.

The fixed-C2 mechanism evaluation, paired tissue-stratified image bootstrap, interior matching, H1 and error diagnostics are now complete; see the [mechanism closeout](VISIBILITY_SUPERVISION_MECHANISM_2026-10-15.md). A1−A2 mean miss-rate improvement is −.008621 [−.040918,+.017699], with seed effects −.025862/+.008621, so clause1 also fails. Neither new seeds nor confirmation are launched.

The training-label audit touched 7 Dead objects in A2 versus 70 in A3 under one deterministic augmentation per training image; this is not training-wide exposure frequency. E4's controlled-crop evidence does not prove HV zeroing causes misses, and this negative dev result does not rule out all visibility-aware training or direct-mask architectures.

## Next research work

1. Integrate this negative dev result with stage I–V, corrected E4 and the official/strict metric distinction in the final report.
2. Keep PM-Net as a proposed experiment. Before implementation/training, freeze matching and overlap rules, training-count-based query capacity, matched-capacity controls, reproducible augmentation-resume state, and a feasible storage/runtime budget. Preserve the proposal's minimum/target/stretch scope and fixed quality/cost criteria.
3. Keep all current checkpoints and old artifacts. No new training or confirmation is launched by this result.

## Provenance and validation

Exact timestamp, eight summary SHA256 values, paths, branch HEAD and per-seed arithmetic: [original-image snapshot](visibility_supervision_original_image_snapshot_2026-10-15.json). Frozen rule: [spec](../superpowers/specs/2026-10-14-visibility-supervision-design.md). Repeated fold2 use limits inference; these are not test or three-split scores. This original-image audit changed no scientific implementation. Subsequent mechanism-runner validation is reported separately (52 focused checks).
