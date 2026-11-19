# P36 frozen design and reading, written before submission

Recorded 2026-11-19 (America/New_York), before the P36 array was submitted. The measurements, the decoders, the
selection procedure and the comparisons below are fixed by the scientific commit named in the submission receipt. No
P36 output exists at the time of writing.

## What is known when this note is written

- P34 (offset readout on frozen P22 detectors; three of four tasks completed): moving each predicted point by the
  readout gains 156 to 172 matches and loses 158 to 182 on the 522 development images of each detector. The readout
  places the center within three pixels for 99.4% to 99.8% of nuclei on training images, for 72% to 84% on the same
  training images moved by (5, 3) pixels and for 71% to 81% on development images.
- P35 (four frozen P22 detectors at sixteen alignments to the 16-pixel patch grid; all four tasks completed): on 48
  selected training images, nuclei at least 24 pixels from every border are matched at 97.4% to 99.2% at the trained
  alignment and at 67.1% to 72.6% at the fifteen other alignments. On the 522 development images the two figures are
  60.6% to 67.7% and 61.0% to 67.9%. Of 7,014 such development nuclei, 11.2% to 13.6% are matched at no alignment,
  25.6% to 37.6% at all sixteen and 50.8% to 60.8% at some. Averaging the sixteen score maps at the fixed threshold 0.5
  keeps 9% to 15% of nuclei at 95% to 96% precision; the P35 primary criteria are not met.

## Detectors

Eight completed 130-epoch detectors, last checkpoint, frozen: P22 control (last 12 encoder layers updated) and P22 all
24 layers, seeds 19 and 1; P15 control and P15 trained with random translation of up to 16 pixels, seeds 19 and 1.
Identity of every run (configuration, cohort, weights, preserved files) passed a read-only check before submission.

## Measurements

1. Every image at sixteen alignments (shifts of -8, -4, 0, 4 pixels per axis), once with white fill and once with
   mirrored fill of the vacated strip. All local maxima with a score of at least 0.1 are stored in original coordinates.
2. Images: a fixed sample of 240 training images chosen by a hash of the image number, and all 522 development images.
3. Per nucleus: the alignments at which it is matched, and the displacement of the nearest retained point (within six
   pixels) at each alignment. For isolated nuclei at least 24 pixels from every border the mean squared displacement is
   split into a part that is the same at all alignments and a part that changes with the alignment.

## Decoders and selection

- `original`: trained alignment, score threshold 0.5.
- `selected_threshold`: trained alignment, one score threshold from 0.1, 0.2, ..., 0.9.
- `consensus`: points of the fifteen shifted alignments with a score of at least 0.5, grouped within a radius of 2, 3
  or 4 pixels around the highest-scoring point; a group is kept when at least v of 15 alignments contribute (v from 1
  to 15, relative to the alignments valid at that position); its position is the mean of its points.

The threshold and the consensus rule (fill, radius, v) are selected per detector on the training sample at the fifteen
shifted alignments, by the F1 score over all annotated nuclei, and written to disk before the detector reads any
development image. Ties: threshold closest to 0.5; white fill, smaller radius, larger v. The trained alignment is left
out of the selection and of the consensus because training images are reproduced almost exactly at that alignment.

## Comparisons under the original two-seed criteria

Criteria, unchanged: interior Dead detection rate at least 0.60, increase of at least 0.05 with a paired bootstrap 95%
lower bound above 0, precision at least 0.75 and not more than 0.02 below the reference, overall detection rate not
more than 0.02 below the reference.

- Primary: P22 all 24 layers, `consensus` against `original` of the same detector.
- Secondary: P22 all 24 layers, `selected_threshold` against `original`; P15 translation-trained detector with each of
  its three decoders against `original` of the P15 control of the same seed; every other detector, each new decoder
  against its own `original`.

## Expectations stated in advance

1. Training sample: detection at the trained alignment near 96% to 98% for the six detectors trained without
   translation and clearly lower at the shifted alignments, close to the development level. For the translation-trained
   P15 detector the trained alignment has no advantage on training images.
2. The exact-center loss gives a loss-optimal peak score below 0.5 once the center position is uncertain by more than
   about 0.95 pixels. The selected threshold is therefore expected below 0.5 for most detectors, and lowest for the
   translation-trained detector, whose detection rate at 0.5 is the lowest of the eight. If that detector reaches or
   exceeds its control once the threshold is selected, the result recorded for P15 belongs to the fixed threshold, not
   to the detector.
3. Consensus: a moderate gain in the F1 score on development images, a few points at most. P22 all 24 layers needs
   about eight more interior Dead matches of 145 together with two more points of precision for the primary criteria;
   this is not expected with confidence.
4. Displacement: the part that changes with the alignment is expected near 1.5 to 2 squared pixels (P35: the averaged
   score near the center is about half of the single-alignment score), of the same order as the part that stays.

## How the results will be read

| Observation | Reading |
|---|---|
| Selected thresholds below 0.5 and higher F1 on development images at those thresholds | The fixed threshold removes nuclei the detector locates with a lower score; threshold selection on shifted training images becomes part of the decoder. |
| Consensus raises detection and precision together | Combining alignments recovers alignment-dependent misses; training for agreement across alignments is the next model change. |
| The translation-trained detector is equally good at all alignments and gains most from a lower threshold | Training with translation removes the dependence on one alignment; its recorded loss at threshold 0.5 reflects lower scores. |
| Displacement mostly changes with the alignment | Position error comes from the patch grid; averaging positions over alignments reduces it. |
| Displacement mostly the same at all alignments | Position error comes from the image content and the annotation rule; alignment averaging does not reduce it. |

## Limits fixed with the design

For the translation-trained detector the shifted training images resemble its own training data more closely than
development images do; the difference between its training and development curves is reported. The sixteen alignments
of one nucleus are repeated observations of the same object. Nuclei within 24 pixels of a border are reported
separately because shifted inputs cut them. Points farther than six pixels from a center do not enter the displacement
statistics. All counts come from fold 1 (training and development split) and nucleus-center detection in single
precision on CPU; the stored BF16 counts are compared image by image.
