"""E3 quality model (spec 2026-10-11 §3): sklearn logistic with fold1-internal grouped CV.

C is chosen by GroupKFold(5) mean AUC over images inside the training fold only; the torch
linear head is a two-seed direction-consistency check reported alongside, never the primary
point estimate.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CS = (0.1, 1.0, 10.0)


def fit_logistic(X: np.ndarray, y: np.ndarray, groups: np.ndarray, Cs=CS) -> dict:
    """Standardize, then pick C by grouped 5-fold CV AUC (ties -> smaller C)."""
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("training labels must contain both classes")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    splits = list(GroupKFold(n_splits=5).split(Xs, y, groups=groups))
    best_C, best_auc = None, -np.inf
    for C in Cs:
        aucs = []
        for tr, te in splits:
            if len(np.unique(y[te])) < 2:
                continue
            clf = LogisticRegression(max_iter=5000, C=C).fit(Xs[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xs[te])[:, 1]))
        auc = float(np.mean(aucs)) if aucs else 0.5
        if auc > best_auc + 1e-12:
            best_C, best_auc = C, auc
    clf = LogisticRegression(max_iter=5000, C=best_C).fit(Xs, y)
    return {"scaler": scaler, "clf": clf, "C": float(best_C), "cv_auc": float(best_auc)}


def predict(fit: dict, X: np.ndarray) -> np.ndarray:
    return fit["clf"].predict_proba(fit["scaler"].transform(np.asarray(X, float)))[:, 1]


def reliability(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Equal-width reliability bins; each row has bin, n, mean_p and the realized rate."""
    y_true = np.asarray(y_true, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        out.append({"bin": b, "n": int(m.sum()),
                    "mean_p": float(p[m].mean()) if m.any() else float("nan"),
                    "rate": float(y_true[m].mean()) if m.any() else float("nan")})
    return out


def torch_head_directions(X: np.ndarray, y: np.ndarray, X_apply: np.ndarray,
                          seeds=(0, 1), epochs: int = 200, lr: float = 0.05) -> list[np.ndarray]:
    """Two torch linear heads on the same standardized features; returns apply probabilities."""
    X = np.asarray(X, float)
    scaler = StandardScaler().fit(X)
    Xs = torch.tensor(scaler.transform(X), dtype=torch.float32)
    ya = torch.tensor(np.asarray(y, float), dtype=torch.float32)
    Xa = torch.tensor(scaler.transform(np.asarray(X_apply, float)), dtype=torch.float32)
    outs = []
    for seed in seeds:
        torch.manual_seed(seed)
        head = torch.nn.Linear(Xs.shape[1], 1)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(Xs).squeeze(1), ya)
            loss.backward()
            opt.step()
        with torch.no_grad():
            outs.append(torch.sigmoid(head(Xa).squeeze(1)).numpy().astype(float))
    return outs
