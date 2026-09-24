"""Dense CONCH prior for nucleus types (pillar A; see docs/findings.md 2026-09-24).

One CONCH trunk pass per 256 px patch (resized to 448); each nucleus gets a text-aligned embedding
by running CONCH's contrastive attentional pooler with its keys restricted to tokens within
`radius` px of the nucleus centroid. Zero-shot logits = cosine similarity to the frozen class text
prototypes (weights/text_protos/*.pt, built by nucseg.text.prototypes).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .prototypes import load_conch


def instance_table(inst_maps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate instances of (N, H, W) label maps -> (img index, inst id, centroid (y, x))."""
    img, ids, cen = [], [], []
    for j in range(len(inst_maps)):
        m = np.asarray(inst_maps[j])
        u = np.unique(m)
        u = u[u > 0]
        if len(u) == 0:
            continue
        img.append(np.full(len(u), j))
        ids.append(u)
        cen.append(np.array(ndimage.center_of_mass(np.ones_like(m), m, u), np.float32).reshape(-1, 2))
    if not img:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2), np.float32)
    return np.concatenate(img), np.concatenate(ids), np.concatenate(cen)


def instance_mean_probs(inst: np.ndarray, prob: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Mean class probability inside each instance. inst (H, W), prob (H, W, C), ids (n,) -> (n, C)."""
    flat = inst.ravel()
    cnt = np.bincount(flat, minlength=int(flat.max()) + 1).astype(np.float64)
    sums = np.stack([np.bincount(flat, prob[..., c].ravel(), minlength=len(cnt)) for c in range(prob.shape[-1])], -1)
    return (sums[ids] / np.maximum(cnt[ids], 1)[:, None]).astype(np.float32)


class ConchNucleusPrior:
    def __init__(self, protos: str = "weights/text_protos/conch_v1.pt", flavour: str = "desc",
                 radius: float = 32, size: int = 448, device: str = "cuda"):
        self.model, _, _ = load_conch(device)
        self.vis = self.model.visual
        p = torch.load(protos)
        self.protos = p[flavour].to(device)  # (6, D), row 0 = background
        self.radius, self.size, self.device = radius, size, device
        self.mean = torch.tensor(self.vis.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.vis.image_std, device=device).view(1, 3, 1, 1)
        g = size // 16
        yx = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"), -1).float()
        self.tok_yx = ((yx + 0.5) * (256 / g)).view(-1, 2).to(device)

    @torch.no_grad()
    def tokens(self, imgs_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(imgs_u8)).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = self.vis.trunk.forward_features((x - self.mean) / self.std)
        return t[:, self.vis.trunk.num_prefix_tokens:].float()

    @torch.no_grad()
    def pool(self, tok: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """tok (L, C) of one patch, centroids (n, 2) -> (n, D) L2-normalised embeddings."""
        d = torch.cdist(centroids, self.tok_yx)
        keep = d <= self.radius
        keep[torch.arange(len(centroids)), d.argmin(1)] = True
        pooled = self.vis.attn_pool_contrast(tok[None].expand(len(centroids), -1, -1), attn_mask=keep)[:, 0]
        return F.normalize(self.vis.ln_contrast(pooled) @ self.vis.proj_contrast, dim=-1)

    @torch.no_grad()
    def embed(self, images, inst_maps, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> (img index (M,), inst id (M,), embeddings (M, D) float32) for every instance in inst_maps."""
        img, ids, cen = instance_table(inst_maps)
        out = np.zeros((len(img), self.protos.shape[1]), np.float32)
        for s in range(0, len(images), batch_size):
            tok = self.tokens(np.asarray(images[s:s + batch_size]))
            for b in range(tok.shape[0]):
                sel = np.where(img == s + b)[0]
                if len(sel):
                    out[sel] = self.pool(tok[b], torch.from_numpy(cen[sel]).to(self.device)).cpu().numpy()
        return img, ids, out

    def zero_shot_logits(self, emb: np.ndarray) -> np.ndarray:
        """(M, D) -> (M, 5) cosine similarities to the 5 nucleus-class prototypes."""
        return emb @ self.protos[1:].cpu().numpy().T
