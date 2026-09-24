#!/usr/bin/env bash
# Download gated foundation-model weights into weights/<repo>. Usage: scripts/download_weights.sh [repo ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_TOKEN="${HF_TOKEN:-$(tr -d '\n ' < "$ROOT/hg-token.txt")}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
# hf-mirror does not serve gated LFS files; go to the origin (through the proxy on LM1/LM2).
export HF_ENDPOINT="${HF_ENDPOINT_GATED:-https://huggingface.co}"
PY="${PY:-python}"
REPOS=("$@"); [ ${#REPOS[@]} -eq 0 ] && REPOS=(MahmoodLab/CONCH MahmoodLab/UNI MahmoodLab/UNI2-h)
for r in "${REPOS[@]}"; do
  echo "[$(date +%T)] $r"
  for try in 1 2 3 4 5; do
    "$PY" -c "from huggingface_hub import snapshot_download as s; s('$r', local_dir='$ROOT/weights/$r', max_workers=2)" && break
    echo "retry $try for $r"; sleep 10
  done
done
du -sh "$ROOT"/weights/*/* 2>/dev/null
