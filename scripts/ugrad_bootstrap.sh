#!/usr/bin/env bash
# Runs ON a ugrad machine (called by push_ugrad.sh). Idempotent: builds the venv in local /tmp
# if missing/broken and installs a daily keep-alive cron so /tmp's 10-day purge never fires.
set -euo pipefail
source /tmp/cgf2604/bdd26/scripts/ugrad_env.sh
cd "$BDD"
if ! "$UGRAD_ROOT/venv/bin/python" -c "import torch, nucseg, cv2, skimage" 2>/dev/null; then
  echo "[bootstrap] building venv on $(hostname)"
  rm -rf "$UGRAD_ROOT/venv"
  uv venv --python 3.10 "$UGRAD_ROOT/venv"
  uv pip install --python "$UGRAD_ROOT/venv/bin/python" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python "$UGRAD_ROOT/venv/bin/python" -r env/requirements.txt
  uv pip install --python "$UGRAD_ROOT/venv/bin/python" -e .
  # imgaug drags in GUI opencv (needs libGL, absent on LM2); keep only the headless build
  uv pip uninstall --python "$UGRAD_ROOT/venv/bin/python" opencv-python || true
  uv pip install --python "$UGRAD_ROOT/venv/bin/python" --reinstall-package opencv-python-headless opencv-python-headless==4.8.1.78 numpy==1.23.5
fi
if [ -d third_party/CONCH ] && ! "$UGRAD_ROOT/venv/bin/python" -c "import conch, transformers" 2>/dev/null; then
  uv pip install --python "$UGRAD_ROOT/venv/bin/python" -r env/requirements.txt
  uv pip install --python "$UGRAD_ROOT/venv/bin/python" --no-deps -e third_party/CONCH
fi
CRON='23 5 * * * find /tmp/cgf2604 -xdev -exec touch -a -c -h {} + >/dev/null 2>&1'
{ crontab -l 2>/dev/null | grep -v 'touch -a -c -h' || true; echo "$CRON"; } | crontab -
"$UGRAD_ROOT/venv/bin/python" -c "import torch; print('[bootstrap]', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
du -sh "$UGRAD_ROOT" 2>/dev/null; du -sh "$HOME" 2>/dev/null
