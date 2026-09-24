"""Instance re-typing with region-pooled CONCH embeddings (pillar A).

A per-nucleus classifier on frozen CONCH embeddings, optionally *anchored* to the text prototypes
(weights initialised at tau * text prototype and L2-pulled towards it), fused with the segmenter's
per-instance type probabilities by a log-linear product of experts whose weight is tuned on the
validation fold. Ablation axes: anchor (text / none), init (text / random), zero-shot (no training).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def train_anchored_linear(x: np.ndarray, y: np.ndarray, protos: np.ndarray | None, anchor: float = 0.0,
                          tau: float = 20.0, epochs: int = 200, lr: float = 1e-2, wd: float = 1e-4,
                          balanced: bool = False, logit_adjust: float = 0.0, seed: int = 0,
                          device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Full-batch multinomial logistic regression on L2-normalised embeddings.
    x (N, D), y (N,) in 1..C, protos (C, D) text prototypes or None (random init, no anchor).
    Returns (W (C, D), b (C,)); logits = x @ W.T + b."""
    g = torch.Generator().manual_seed(seed)
    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y - 1).long().to(device)
    C = int(Y.max()) + 1
    if protos is not None:
        W0 = tau * torch.from_numpy(protos).float().to(device)
    else:
        W0 = (torch.randn(C, X.shape[1], generator=g) * 0.01).to(device)
    W = W0.clone().requires_grad_(True)
    b = torch.zeros(C, device=device, requires_grad=True)
    freq = torch.bincount(Y, minlength=C).float()
    weight = (freq.sum() / (C * freq)) if balanced else None
    la = logit_adjust * torch.log(freq / freq.sum())
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(epochs):
        logits = X @ W.T + b + la
        loss = F.cross_entropy(logits, Y, weight=weight) + wd * (W ** 2).sum()
        if protos is not None and anchor > 0:
            loss = loss + anchor * ((W - W0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fuse(p_seg: np.ndarray, p_txt: np.ndarray, alpha: float, eps: float = 1e-6) -> np.ndarray:
    """Log-linear pooling of per-instance class probabilities over the 5 nucleus classes.
    p_seg (M, 6) includes background (dropped and renormalised); p_txt (M, 5). Returns type ids 1..5."""
    ps = p_seg[:, 1:] / np.maximum(p_seg[:, 1:].sum(1, keepdims=True), eps)
    z = (1 - alpha) * np.log(ps + eps) + alpha * np.log(p_txt + eps)
    return z.argmax(1) + 1


def paint_types(inst: np.ndarray, inst_img: np.ndarray, inst_id: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Rebuild (N, H, W) type maps from per-instance types."""
    out = np.zeros(inst.shape, np.uint8)
    for j in np.unique(inst_img):
        sel = inst_img == j
        lut = np.zeros(int(inst[j].max()) + 1, np.uint8)
        lut[inst_id[sel]] = types[sel]
        out[j] = lut[inst[j]]
    return out
