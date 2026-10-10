"""T 实验（固定几何分型探针）纯逻辑：实例/环池化、T1 聚合规则、线性头。

Spec: docs/superpowers/specs/2026-10-10-typing-probe-t-design.md（2026-10-10 批准，值冻结）：
环宽 24 px、头种子 {0,1}、AdamW lr 1e-3 / wd 1e-4 / 20 epoch / batch 4096，全部不扫描。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

RING_WIDTH = 24          # spec §3 冻结
HEAD_SEEDS = (0, 1)


def pool_instances(feat: np.ndarray, inst: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """feat (H,W,C) float32 -> (len(ids),C) per-instance mean, rows in `ids` order."""
    flat = feat.reshape(-1, feat.shape[-1])
    key = inst.ravel().astype(np.int64)
    sums = np.zeros((int(ids.max()) + 1, feat.shape[-1]), np.float64)
    cnt = np.bincount(key, minlength=int(ids.max()) + 1)
    np.add.at(sums, key, flat)
    assert (cnt[ids] > 0).all(), "instance id with zero pixels"
    return (sums[ids] / cnt[ids][:, None]).astype(np.float32)


def pool_rings(feat: np.ndarray, inst: np.ndarray, ids: np.ndarray, width: int = RING_WIDTH) -> np.ndarray:
    """Ring = dilate(own mask, ellipse 2w+1) minus ALL instance pixels; empty ring -> zeros."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    bg = inst == 0
    out = np.zeros((len(ids), feat.shape[-1]), np.float32)
    for row, i in enumerate(ids):
        ring = cv2.dilate((inst == i).astype(np.uint8), k) & bg
        n = int(ring.sum())
        if n:
            out[row] = feat[ring.astype(bool)].mean(0)
    return out


def t1_classes(inst_prob: np.ndarray) -> np.ndarray:
    """(M,6) instance-mean tp softmax -> classes 1..5; background channel dropped (spec §3)."""
    return inst_prob[:, 1:6].argmax(1).astype(np.int64) + 1


def train_linear_head(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = 20,
                      lr: float = 1e-3, wd: float = 1e-4, batch: int = 4096):
    """Torch linear classifier d->5 (spec §3 frozen recipe); y in 0..4. Returns (W, b)."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], 5)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    yt = torch.from_numpy(y.astype(np.int64))
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), batch):
            idx = perm[s:s + batch]
            loss = F.cross_entropy(lin(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return lin.weight.detach().numpy().copy(), lin.bias.detach().numpy().copy()


def apply_head(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (X @ W.T + b).argmax(1).astype(np.int64) + 1


def sklearn_logreg(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, y)
    return (lr.predict(X) + 1).astype(np.int64)
