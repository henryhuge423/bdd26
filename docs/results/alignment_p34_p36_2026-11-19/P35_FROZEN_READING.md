# P35 frozen reading, written before any P35 output was read

Recorded 2026-11-19 (America/New_York morning). Array 55867961 was submitted once at 2026-11-19T04:20:31Z. Up to this
note, the only information read about the job is the scheduler state (four tasks RUNNING at 04:23:29Z, empty error
files). No evaluation log, count or figure of P35 has been read. The decoders, the sixteen alignments and the paired
two-seed rule are fixed by scientific commit 47b2f55905c9947d45922fc191d0969207197d93.

## Primary comparison

Detector with all 24 encoder layers updated (P22 arm M), seeds 19 and 1. Decoder `mean_probability` (mean of the
realigned probabilities over the sixteen alignments) against decoder `original` (the trained alignment), both at the
unchanged threshold 0.5, peak rule and three-pixel one-to-one matching. Original two-seed criteria: interior Dead
detection rate at least 0.60, increase of at least 0.05 with a paired bootstrap 95% lower bound above 0, precision at
least 0.75 and not more than 0.02 below the original decoder, overall detection rate not more than 0.02 below the
original decoder. The three other comparisons (arm M with `mean_logit`; arm C with both decoders) describe mechanism.

## Expectation stated in advance

The score peaks of these detectors are about one to two pixels wide, and P34 measured a localization spread of about
1.4 pixels on images at alignments the detector was not trained on. Averaging sixteen maps whose peaks sit at slightly
different pixels lowers the peak height. At the fixed threshold 0.5 the averaged decoders are therefore expected to
return fewer points, with higher precision and an unchanged or lower detection rate; the primary criteria are not
expected to be met. The main product is the per-nucleus count of alignments at which the nucleus is matched.

## How the alignment counts will be read

| Observation | Reading |
|---|---|
| Training images: detection rate at the trained alignment far above the other fifteen alignments | The fit of the training images is tied to one alignment; this repeats the P34 shifted-image result at fifteen further offsets. |
| Development images: most nuclei matched at nearly all or nearly no alignments | Misses belong to the nucleus and its image content; alignment diversity cannot recover them. |
| Development images: many nuclei matched at an intermediate number of alignments | Misses depend on the alignment; a detector that responds consistently across alignments can recover the nuclei matched at most alignments. |

Known limits fixed with the design: shifted inputs lose a strip of up to eight pixels and receive white fill, so nuclei
near the border are truncated at some alignments; the analysis will report nuclei at least 24 pixels from every border
separately. The sixteen alignments of one nucleus are not independent samples.
