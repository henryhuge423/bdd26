"""Thin bridge to the official HoVer-Net code in third_party/hover_net.

We reuse the official architecture (fast mode), target generation, losses, augmentations and
post-processing unchanged; only the PanNuke data plumbing is ours (see data.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

HOVER_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "hover_net"
if str(HOVER_ROOT) not in sys.path:
    sys.path.insert(0, str(HOVER_ROOT))

from dataloader.augs import (  # noqa: E402
    add_to_brightness, add_to_contrast, add_to_hue, add_to_saturation, gaussian_blur, median_blur,
)
from models.hovernet.net_desc import HoVerNet  # noqa: E402
from models.hovernet.post_proc import process as post_process  # noqa: E402
from models.hovernet.targets import gen_instance_hv_map  # noqa: E402
from models.hovernet.utils import dice_loss, mse_loss, msge_loss, xentropy_loss  # noqa: E402
from run_utils.utils import convert_pytorch_checkpoint  # noqa: E402

__all__ = [
    "HoVerNet", "post_process", "gen_instance_hv_map", "dice_loss", "mse_loss", "msge_loss",
    "xentropy_loss", "convert_pytorch_checkpoint", "gaussian_blur", "median_blur", "add_to_hue",
    "add_to_saturation", "add_to_brightness", "add_to_contrast",
]
