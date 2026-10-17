# PM-Net point-grid feasibility audit — 2026-10-15

## Scope and rule

Descriptive CPU analysis of all 2,656 original fold1 training label maps. No model training, validation-based selection, test-fold access or inference. Raw record: `runs/analysis/pmnet_point_grid_20261015/audit.json`, with SHA256 for instance/type/tissue inputs. The instance hash agrees with the prior training-geometry audit.

For each positive instance ID, compute its visible-pixel arithmetic centroid. Choose the actual mask pixel minimizing squared Euclidean distance to this centroid; break ties by row-major order. Assign that legal integer pixel to `(y // stride, x // stride)`, with the grid anchored at the image origin. Class is the majority pixel type. This is one deterministic target convention for feasibility analysis, not an approved model design.

## Results

63,218 objects, including 967 Dead; 116 empty images. The previous finding of 23 rounded centroids outside their own masks is reproduced. The chosen visible pixel lies inside its instance by construction.

| Stride | Images with collisions | Objects in colliding cells | Dead in colliding cells | Excess objects, one class-agnostic slot/cell | Excess objects, one slot/class/cell |
|---|---:|---:|---:|---:|---:|
|2|0|0|0|0|0|
|4|1|2|0|1|1|
|8|31|89|8|45|39|

The stride-4 collision occurs in training image1515 (Colon,32 objects). Objects in collision cells are not all necessarily lost: excess over unique available slots is the appropriate single-slot representation deficit.

## Consequences and limits

- A naive stride-8 single-point heatmap cannot represent all these fixed targets. A class-aware heatmap still has39 excess objects under this convention. This does not prove a learned detector must miss39 objects: offsets, different target locations, multiple slots and set queries change the representation.
- Stride4 has only one excess target here, but needs an explicit conflict rule. Stride2 has zero original-label collisions; this does not prove it is the best quality/cost choice.
- These counts exclude augmented views and peak-suppression neighbourhoods. Even distinct grid cells can compete under local-maximum decoding. Audit augmentation, empty targets and duplicate resolution before training.
- The prior maximum276 objects still rules out an unconditional256-query capacity guarantee.512 covers original counts but is not a validated GPU-memory or latency budget.
- No detection, typing, mask quality or cost improvement is established. These are engineering constraints for a minimal point-representation study, not reasons to bundle extra modules into PM-Net.

## Proposed next question

After the current fusion dev decision, isolate whether instance-level point representation supplies useful detection information under the existing UNI/native-resolution recipe. Keep backbone, resolution, sampling, augmentation and schedule fixed; defer visibility supervision, hierarchical typing and one-to-many queries. Freeze target legality, capacity/overflow, collision policy, matching/loss normalization, mask decoding and quality/cost criteria before any fit. Architecture implementation and new training require the corresponding reviewed design and execution authorization.
