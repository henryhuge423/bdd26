# P26 formal submission — 10 November 2026

Array **55663583** was submitted once at **18:14:02 UTC / 13:14:02 EST**. Tasks map to C19, R19, C1 and R1, with at most two concurrent runs. Each requests one V100, four CPU cores, 32 GiB and 12 hours; the four requests total at most 48 GPU hours, with no automatic requeue. Initial state: all PENDING. Final training results are pending.

R adds a fixed pretrained-encoder parameter penalty to the original C configuration. Both use the same last 12 trainable blocks, learning rates, data order, 130 epochs, last checkpoint and original two-seed decision criteria. See the [design](../../superpowers/specs/2026-11-10-proposal-p26-design.md).

The final CPU suite reported **1096 passed, 3 skipped, 19073 warnings in 1286.94 seconds**. The 97 focused checks passed after an independent review identified and fixed a GPU witness that bypassed the production backward helper. The added regression was observed failing before the fix. No common metric or HoVer-Net code changed.

Real V100 validation **55661530** completed with exit **0:0** in **689 seconds**. Its 97 node tests passed, followed by both-seed exact original-C gradient/parameter/AdamW/RNG parity, nonzero R anchoring gradients and actual update effects, unchanged frozen tensors, BF16, last/final checkpoint reload and source identity. The one-epoch 134-image training smoke included a six-image tail. Its 1.5× safety estimates were **7.0656 hours for C** and **7.1988 for R**; training peaks were **4.0943/4.6575 GiB**. Four preliminary training-image PNGs are retained with coordinates; these are not final experiment outcomes.

Scientific source commit: `bd01a06cda900eac5376a72ca6b8bc36c2cb0407`. All **490** source entries are sealed by manifest SHA256 `7aab074a1eb9b852e6ffa063d38fc6af57cfbe7f53a213ed8993647083510714`. The validation collection's 29 small files passed byte/SHA256 checks. Formal submission repeated accurate three-scope quota, capacity, scoped-target write/readback, source and account-budget checks. The [portable receipt](p26_submission.json) records the exact mapping and limits.

Once all four runs finish, verify 130 epochs, the final checkpoint, preservation hashes and the original decision criteria. Each run saves eight audit and four training examples selected from ground truth before training. Current experiment authorization ends at this submission; further stages follow the original research decisions.
