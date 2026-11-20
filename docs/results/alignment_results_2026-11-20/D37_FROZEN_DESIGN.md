# D37: separating alignment-sensitive and low-score misses

Fixed on November 20, 2026 before computing D37. This is a diagnostic of preserved P36 points, with no model forward pass, training or decoder selection.

Use all eight completed P36 detectors, both roles (240 training images and 522 development images), both fills and all sixteen alignments. Retain the stored recording floor 0.1 and the existing decoder threshold 0.5; use the original maximum-cardinality one-to-one matching within three pixels. Verify every consumed preserved file and reproduce the original, selected-threshold and consensus counts image by image before analysis.

Partition each nucleus into exactly one state, in this order: matched at the original alignment; missed there but matched at another alignment at 0.5; matched only after admitting recorded scores below 0.5; a recorded point within three pixels at some alignment but assigned to another nucleus; no recorded point within three pixels at any alignment at the recording floor 0.1. The last state is conditional on the recording floor and the three-pixel radius; it does not imply absent response everywhere in the nucleus. The union over alignments uses annotations and measures availability, not a deployable decoder.

Report all nuclei, central nuclei (at least 24 pixels from every border and valid at all alignments) and interior Dead nuclei separately. Preserve full object coordinates and assignments. Fixed image numbers are 178, 776, 1037 and 1482 for the all-layer detector and the translation-trained detector, both seed 19; save all four images regardless of outcomes.

The questions are whether remaining misses are dominated by alignment or low scores, whether these causes differ for translation-trained detectors, and where the chosen consensus loses recoverable nuclei. D37 is adaptive to P36, descriptive, and creates no new selection rule. Position shifts, low scores, annotation competition and image-content explanations will be described as observations or hypotheses according to the evidence available.
