"""KongNet instance decode for strict-protocol PanNuke evaluation (phase-2 line A).

The released KongNet checkpoints are 6-head models (head 0 = overall detection,
heads 1-5 = neoplastic..epithelial), each head emitting 3 channels. The released
inference pipeline only documents channel 2 of each head (the class detection
heatmap; PanNuke indices [5, 8, 11, 14, 17]). Channels 0/1 are undocumented, but the
repo's `multihead_seg_post_process` shows the intended instance recipe: a filled
class mask (seg channel, thresholded) cut by a contour channel (thresholded), then
morphological open+close with a 3x3 ellipse. Which of channels 0/1 is the filled
mask is resolved empirically on train folds before any val/test use
(`scripts/kongnet_eval.py --mode probe`).

Decode thresholds are tuned on VAL folds only (same rule as every other decode in
this project); the released defaults live here as the centre of the sweep grid.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

NUM_HEADS = 6
HEAD_CHANNELS = 3
TOTAL_CHANNELS = NUM_HEADS * HEAD_CHANNELS  # 18
# third channel of heads 1..5 = per-class detection heatmap, official class order
# (repo `get_cell_channel_map`: neoplastic, inflammatory, connective, dead, epithelial)
CLASS_HEATMAP_CH = [5, 8, 11, 14, 17]
DEFAULT_SEG_THR = 0.5  # multihead_seg_post_process thresholds[0..2]
DEFAULT_CONTOUR_THR = 0.3  # multihead_seg_post_process thresholds[3]


def ellipse3_kernel() -> np.ndarray:
    """cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) — the repo's default."""
    return np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], bool)


def open_close(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """MORPH_OPEN then MORPH_CLOSE, as `data_utils.morphological_post_processing`."""
    mask = ndi.binary_opening(mask, structure=kernel)
    return ndi.binary_closing(mask, structure=kernel)


def per_class_masks(probs: np.ndarray, seg_ch: list[int], contour_ch: list[int],
                    seg_thr: float, contour_thr: float,
                    kernel: np.ndarray | None = None) -> np.ndarray:
    """(H, W, 18) probs -> (5, H, W) bool: per-class seg mask, contour-cut, open+closed."""
    kernel = ellipse3_kernel() if kernel is None else kernel
    masks = np.zeros((len(seg_ch), probs.shape[0], probs.shape[1]), bool)
    for c, (sc, cc) in enumerate(zip(seg_ch, contour_ch)):
        m = probs[:, :, sc] > seg_thr
        m[probs[:, :, cc] > contour_thr] = 0
        masks[c] = open_close(m, kernel)
    return masks


def resolve_conflicts(masks: np.ndarray, seg_probs: np.ndarray) -> np.ndarray:
    """Pixels claimed by >1 class go to the class with the highest seg-channel prob.

    seg_probs: (n_class, H, W) per-class seg probabilities (already selected)."""
    n_claim = masks.sum(0)
    out = masks.copy()
    best = seg_probs.max(0)
    for c in range(len(masks)):
        drop = masks[c] & (n_claim > 1)
        if drop.any():
            out[c][drop] = seg_probs[c][drop] >= best[drop]
    return out


def decode_instances(probs: np.ndarray, seg_ch: list[int], contour_ch: list[int],
                     seg_thr: float = DEFAULT_SEG_THR,
                     contour_thr: float = DEFAULT_CONTOUR_THR) -> tuple[np.ndarray, np.ndarray]:
    """Full-image decode -> (inst int32, type int64 0..5 per pixel) for pannuke_eval."""
    masks = per_class_masks(probs, seg_ch, contour_ch, seg_thr, contour_thr)
    masks = resolve_conflicts(masks, probs[:, :, seg_ch].transpose(2, 0, 1))
    inst = np.zeros(probs.shape[:2], np.int32)
    typ = np.zeros(probs.shape[:2], np.int64)
    nid = 0
    for c, m in enumerate(masks):
        lab, n = ndi.label(m)
        if not n:
            continue
        inst[m] = lab[m] + nid
        typ[m] = c + 1
        nid += n
    return inst, typ
