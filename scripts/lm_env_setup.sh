#!/usr/bin/env bash
# Idempotent: create/refresh the `nuclei` conda env on an LM machine (LM1 or LM2).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV="$HOME/.conda/envs/nuclei"
if ! "$ENV/bin/python" -c "import torch, nucseg, cv2, skimage, imgaug" 2>/dev/null; then
  [ -x "$ENV/bin/python" ] || "${CONDA_EXE:-$(command -v conda || echo /usr/local/anaconda3/bin/conda)}" create -y -p "$ENV" python=3.10 >/dev/null
  "$ENV/bin/pip" install -q uv
  "$ENV/bin/uv" pip install --python "$ENV/bin/python" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  "$ENV/bin/uv" pip install --python "$ENV/bin/python" -r env/requirements.txt
  "$ENV/bin/uv" pip install --python "$ENV/bin/python" -e .
fi
"$ENV/bin/python" -c "import torch; print('[env]', torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
