#!/usr/bin/env bash
# Re-create third_party/ at the exact commits used (shallow clones).
set -euo pipefail; cd "$(dirname "$0")/../third_party" 2>/dev/null || { mkdir -p "$(dirname "$0")/../third_party"; cd "$(dirname "$0")/../third_party"; }
[ -d CellViT ] || { git clone -q https://github.com/TIO-IKIM/CellViT.git CellViT && git -C CellViT checkout -q 05097e18e3d194a65121042f631b5753069f5ee3; }  # 2025-07-23
[ -d CellViT-plus-plus ] || { git clone -q https://github.com/TIO-IKIM/CellViT-plus-plus.git CellViT-plus-plus && git -C CellViT-plus-plus checkout -q 463c5c44bfdebfbe3943597eaa84daf3f5e26a5f; }  # 2025-07-23
[ -d hover_net ] || { git clone -q https://github.com/vqdang/hover_net.git hover_net && git -C hover_net checkout -q 67e2ce5e3f1a64a2ece77ad1c24233653a9e0901; }  # 2023-10-27
[ -d hover_next_inference ] || { git clone -q https://github.com/digitalpathologybern/hover_next_inference.git hover_next_inference && git -C hover_next_inference checkout -q 0e99d535fd64640b5486ca9ee4f3864fa96ba83c; }  # 2026-08-26
[ -d hover_next_train ] || { git clone -q https://github.com/digitalpathologybern/hover_next_train.git hover_next_train && git -C hover_next_train checkout -q db4adfa425ffcad51d01a361e660a40c1ecda9dd; }  # 2026-08-26
[ -d PanNuke-metrics ] || { git clone -q https://github.com/TissueImageAnalytics/PanNuke-metrics.git PanNuke-metrics && git -C PanNuke-metrics checkout -q c00014d766ca1be142b81bea19d9ef4315cde65a; }  # 2020-10-20
[ -d PromptNucSeg ] || { git clone -q https://github.com/windygoo/PromptNucSeg.git PromptNucSeg && git -C PromptNucSeg checkout -q fce029082bf2820314a4dc46be443a836e4876a4; }  # 2025-01-10
[ -d CONCH ] || { git clone -q https://github.com/mahmoodlab/CONCH.git CONCH && git -C CONCH checkout -q 141cc09c7d4ff33d8eda562bd75169b457f71a62; }  # 2025-03-25

# PixCell custom pipeline modules (3 loose files) for scripts/pixcell_sample.py
if [ ! -d pixcell_pipeline ]; then
  mkdir -p pixcell_pipeline
  for f in pipeline.py pixcell_controlnet.py pixcell_controlnet_transformer.py; do
    curl -sL "https://hf-mirror.com/StonyBrook-CVLab/PixCell-pipeline-ControlNet/resolve/main/$f" -o "pixcell_pipeline/$f"
  done
fi
