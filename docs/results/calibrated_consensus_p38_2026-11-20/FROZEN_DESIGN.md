# P38: quality calibration before positional consensus

Fixed on November 20, 2026, after reviewing P36 and D37 and before calculating the full P38 experiment. This is an adaptive development experiment on saved predictions. All eight original detectors and their last checkpoints remain frozen.

## Question and controls

Can the scores, alignment support and position variation of low-score candidates improve center detection when used together? P36 selected a lower single-view threshold but its positional consensus always admitted scores of at least 0.5. P38 admits the existing recorded peaks down to 0.1 and compares a support-only filter with a learned quality filter on exactly the same candidate groups. Original P36 decoding, its selected single-view threshold and its selected consensus are reproduced as additional controls.

The primary comparison is P15 translation-trained versus its P15 control, both with the calibrated decoder, in seeds 19 and 1. The secondary comparison is P22 all-layer versus its P22 control with that same procedure. Within each recipe, calibrated decoding is also compared with its original decoder, its P36 consensus and its support-only control. All comparisons are reported, in this fixed order.

## Data roles and selection order

Use the existing eight P36 records: P22 C/M and P15 C/J, each in seeds 19 and 1. Preserve the original fold-1 division, 240 recorded training images and all 522 development evaluation images. Derive annotations, tissue labels and image order from the preserved per-image records. Include images with no annotations; their predicted points count as unmatched predictions.

Sort the 240 training image numbers by SHA256 of `p38-20261120:<image>`. The first 120 fit the quality model; the remaining 120 select both filters. Membership is the same for all detectors and is invariant to incoming image order. The images are separate for calibrator fitting and filter selection, but both subsets were used to train the detector. This is an internal training calibration experiment. Earlier P36/D37 development results informed this design.

Fit and persist all eight selections, then write `selection_complete.json` with the eight hashes before opening any P38 development record or image. Thresholds, features, fill, grouping, regularization and stopping rules are fixed here. The eight-image LM1 integration check uses training images only and supplies implementation and timing evidence; its fitted coefficients and thresholds are discarded.

## Candidate groups and six inputs

Use mirrored fill and the fifteen shifted alignments, excluding the trained alignment. Retain scores at least 0.1. Visit points by descending score, then alignment number, y and x. A point belongs to the earliest group whose fixed first point is within four pixels; otherwise it starts a group. Keep the highest-score point per alignment in each group. Exclude contributions from alignments invalid at the group's first point, and divide support by the number of alignments valid there. The output position is the unweighted mean of the remaining per-alignment positions.

Inputs to the quality model are: log odds of the maximum detector score; log odds of the mean score; support fraction; root mean squared position dispersion in pixels; score standard deviation; and maximum minus mean score. Scores are clipped to [10^-6, 1-10^-6] only when computing log odds. There are no tissue, class, annotation, image identity or nucleus-area inputs at deployment. Correlated alignments provide measured features rather than independent votes in a probabilistic model.

On each fitting image, use the original three-pixel maximum-cardinality one-to-one matcher over all unscreened group positions. A group receives label 1 if assigned to an annotated nucleus and 0 otherwise. Standardize features on the fitting subset; replace a standard deviation below 10^-8 by one. Fit logistic regression by minimizing summed binary cross entropy plus one half the squared coefficient norm, with coefficient penalty 1 and an unpenalized intercept. Use deterministic Newton steps with backtracking, at most 100 iterations; stop when the largest accepted coefficient change is below 10^-7. Constant-label or nonconvergent fitting fails the experiment explicitly.

This score estimates correctness of the initial grouped candidates under the specified matching procedure. Recompute one-to-one assignment after filtering. A coherent group can share a displacement from its annotation; calibration cannot correct that offset by itself.

## Selecting and applying filters

On the 120 selection images, maximize micro F1 over all nuclei subject to precision at least 0.75 and a nonempty prediction set. For the support-only control, try support thresholds 1/15 through 15/15, normalized by valid alignments. For the quality filter, try probability cutoffs 0.05 through 0.95 in steps of 0.05. Ties favor greater support or the higher cutoff. If no nonempty filter meets the precision floor, retain an explicit reject-all rule and report selection failure. The training precision constraint is a selection condition; development precision is measured independently.

Apply selected filters unchanged to both recorded data roles. Use `proposal_p3_utils.score_points`, `aggregate` and `admission` unchanged. The original two-seed decision requires at least 10 Dead-positive images/100 interior Dead nuclei, Dead recall >=0.60, gain >=0.05 with a positive tissue-stratified paired image-bootstrap interval, precision >=0.75, precision and overall recall within 0.02 of the comparator or better, and nonempty predictions on >=95% of Dead-positive images. Preserve the original 1,000 draws and seed 20261018. Report small interior Dead counts, all-nucleus counts, gained/lost assignments and selection failures. No test-fold inference or mask evaluation is included.

## Outputs and resource envelope

Run on an Expanse CPU compute node through SLURM: one job, four CPU cores, 8 GiB, two hours, shared/shared-normal, Lustre, no requeue. The conservative allocation envelope is 8 CPU SU before site minimum charges. No GPU is requested, and no detector forward pass is made. Verify current CPU account, queue, source hashes, all consumed preservation hashes, source/target quotas, target OST space and fsync/readback before submission and preservation. Use a fresh source snapshot and fresh output directories.

Save fitted coefficients, membership lists, selection grids, all per-image decoder counts, group coordinates/scores/support, all annotated-object transitions, input/source hashes, completion and preservation manifests. Save eight large fixed-image PNGs: images 178, 776, 1037 and 1482 for P22 M/19 and P15 J/19, regardless of outcomes. Panels show annotated centers, P36 consensus, low-score support and calibrated consensus. The figure labels explain matching and unmatched points; complete coordinates accompany them. Recheck source identity and reproduce old decoder counts before reporting new results. Detector checkpoints and original records remain preserved.

If P38 improves candidate selection, the next architecture question is whether training with explicit alignment consistency and a center-uncertainty representation can retain the gain with fewer forward passes. That is a later experiment requiring a separate frozen design.
