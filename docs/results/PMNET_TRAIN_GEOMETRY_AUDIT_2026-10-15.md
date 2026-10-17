# PM-Net training-label feasibility audit — 2026-10-15

Scope: fold1 training labels only, 2,656 original 256×256 patches. No model fitting,
validation selection, test access, or new training. Raw records:
`runs/analysis/pmnet_train_capacity_20261015/{capacity,centroid}.json`.

## Query capacity

There are 63,218 visible annotated instances. Per-image count median=20, p90=43,
p95=56.25, p99=94, maximum=276 (image1063, Uterus).

| Query slots | Overflow images | Fraction | Unrepresentable GT lower bound |
|---|---:|---:|---:|
|128|9|0.3389%|506|
|256|1|0.0377%|20|
|512|0|0%|0|

The proposal's provisional 256 slots cannot represent all original training
instances even with perfect localization and masks. Do not silently truncate GT
or drop this image. A 512-slot budget covers this audit, but augmented instance
counts, mask-memory cost, matching cost and inference latency remain unmeasured.
This is a capacity lower bound, not a recommendation established by model quality.
Counts include every labeled object, including boundary fragments; no class filter.

## Legal point targets

Rounding arithmetic centroids with NumPy `rint` places 23/63,218 centers outside
their own visible mask (0.0364%); 1/967 Dead centers is affected. There are 116
empty-label training images, which must produce zero-object targets and remain
valid negative examples. This audit initially encountered the empty-array case;
no output was saved from that failed attempt, and the corrected audit explicitly
handles empty images.

Point-target construction therefore needs an explicit inside-mask rule and a
zero-object path. A centroid alone is not guaranteed legal. The audit does not
select between nearest-visible-pixel, distance-peak, or learned point targets.
Geometry tests should cover concave/disconnected objects, cropped fragments,
non-contiguous IDs, centroid ties, and empty masks before any prototype training.

## Provenance and interpretation

fold1 `inst.npy` SHA256:
`59fb7dc60354e54e642f41072595ee3a4bec700c0b36bd2c94075ae030e5de7e`.
fold1 `tissue.npy` SHA256:
`64746d127bc28e88be6fdc4639af4f21a91fb1fda00e68869ea5fe61a574f615`.
The centroid audit uses scipy.ndimage.center_of_mass with unit pixel mass and
original visible labels; class IDs use the existing instance majority-vote helper.
These are original-label geometry counts, not augmentation exposure frequencies,
biological object counts, or method performance. PM-Net remains an untrained proposal.

## Mask-logit memory arithmetic (not a benchmark)

A hypothetical dense `B×K×256×256` fp32 mask tensor with B=16,K=512 alone
occupies 2GiB; K=256 occupies 1GiB. These counts exclude gradients, additional
layers, encoder features, optimizer states and matching temporaries. A local-mask
implementation has a different footprint. Therefore “512 covers labels” must not
become an unmeasured claim that the model fits or meets the 2× latency target.
