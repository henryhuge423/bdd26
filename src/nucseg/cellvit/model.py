"""CellViT-style network with a UNI (ViT-L/16) encoder.

Architecture follows CellViT / CellViT++ (third_party/CellViT-plus-plus/cellvit/models/cell_segmentation):
ViT tokens from blocks 6/12/18/24 are upsampled through shared skip decoders into three HoVer-style
branches (NP 2ch, HV 2ch, TP 6ch) plus a tissue classifier on the class token. The only structural
change is that each branch ends in a *feature* map (64 ch) followed by a separate `head`, so the type
head can be swapped (linear vs. text-anchored, see docs/RESEARCH_PLAN.md pillar A).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch.utils.checkpoint import checkpoint as _checkpoint

from ..constants import NUM_CLASSES, TISSUES

UNI_MEAN = (0.485, 0.456, 0.406)
UNI_STD = (0.229, 0.224, 0.225)


def conv_block(cin: int, cout: int, k: int = 3, drop: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=(k - 1) // 2), nn.BatchNorm2d(cout), nn.ReLU(True),
                         nn.Dropout(drop))


def deconv_block(cin: int, cout: int, k: int = 3, drop: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2), nn.Conv2d(cout, cout, k, padding=(k - 1) // 2),
                         nn.BatchNorm2d(cout), nn.ReLU(True), nn.Dropout(drop))


class UNIEncoder(VisionTransformer):
    """timm ViT-L/16 matching the UNI checkpoint; returns (cls token, [tokens after extract_layers])."""

    def __init__(self, extract_layers=(6, 12, 18, 24), drop_path_rate=0.1, attn_drop_rate=0.1):
        super().__init__(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5,
                         num_classes=0, dynamic_img_size=True, drop_path_rate=drop_path_rate,
                         attn_drop_rate=attn_drop_rate)
        self.extract_layers = set(extract_layers)

    def forward(self, x):
        x = self.norm_pre(self.patch_drop(self._pos_embed(self.patch_embed(x))))
        feats = []
        for i, blk in enumerate(self.blocks, 1):
            # timm only checkpoints inside forward_features, which this override bypasses, so apply
            # it here per block; use_reentrant=False also stays inert while the encoder is frozen
            # (nothing requiring grad flows through -> no recompute in backward)
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                x = _checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i in self.extract_layers:
                feats.append(x)
        return x[:, 0], feats


class CellViTUNI(nn.Module):
    embed_dim = 1024
    patch_size = 16

    def __init__(self, uni_ckpt: str | None = None, num_types: int = NUM_CLASSES + 1,
                 num_tissues: int = len(TISSUES), drop_rate: float = 0.0, drop_path_rate: float = 0.1,
                 attn_drop_rate: float = 0.1, type_head: nn.Module | None = None,
                 dead_expert: bool = False, widen: int = 0):
        super().__init__()
        self.encoder = UNIEncoder(drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate)
        if uni_ckpt:
            msg = self.encoder.load_state_dict(torch.load(uni_ckpt, map_location="cpu", weights_only=True),
                                               strict=True)
            print(f"[UNI] loaded {uni_ckpt}: {msg}")
        self.tissue_head = nn.Linear(self.embed_dim, num_tissues)
        e, d = self.embed_dim, drop_rate
        self.skip0 = nn.Sequential(conv_block(3, 32, drop=d), conv_block(32, 64, drop=d))
        self.skip1 = nn.Sequential(deconv_block(e, 512, drop=d), deconv_block(512, 256, drop=d),
                                   deconv_block(256, 128, drop=d))
        self.skip2 = nn.Sequential(deconv_block(e, 512, drop=d), deconv_block(512, 256, drop=d))
        self.skip3 = deconv_block(e, 512, drop=d)
        # widen>0 is the C1 equal-parameter control (spec 2026-10-07 §7): the dead expert's
        # parameters folded back into the class-agnostic detection branches, nothing else changes
        self.np_branch, self.hv_branch = (self._branch(d, 512 + widen) for _ in range(2))
        self.tp_branch = self._branch(d)
        self.np_head = nn.Conv2d(64, 2, 1)
        self.hv_head = nn.Conv2d(64, 2, 1)
        self.tp_head = type_head if type_head is not None else nn.Conv2d(64, num_types, 1)
        if dead_expert:
            self.dead_branch = self._branch(d)
            self.dead_np_head = nn.Conv2d(64, 2, 1)
            self.dead_hv_head = nn.Conv2d(64, 2, 1)

    def _branch(self, d: float, b: int | None = None) -> nn.Sequential:
        b = 512 if b is None else b  # bottleneck dim for embed_dim >= 512
        return nn.Sequential(OrderedDict([
            ("up4", nn.ConvTranspose2d(self.embed_dim, b, 2, stride=2)),
            ("up3", nn.Sequential(conv_block(2 * b, b, drop=d), conv_block(b, b, drop=d), conv_block(b, b, drop=d),
                                  nn.ConvTranspose2d(b, 256, 2, stride=2))),
            ("up2", nn.Sequential(conv_block(512, 256, drop=d), conv_block(256, 256, drop=d),
                                  nn.ConvTranspose2d(256, 128, 2, stride=2))),
            ("up1", nn.Sequential(conv_block(256, 128, drop=d), conv_block(128, 128, drop=d),
                                  nn.ConvTranspose2d(128, 64, 2, stride=2))),
            ("out", nn.Sequential(conv_block(128, 64, drop=d), conv_block(64, 64, drop=d))),
        ]))

    def _decode(self, skips, br: nn.Sequential) -> torch.Tensor:
        s0, s1, s2, s3, z4 = skips
        x = br.up3(torch.cat([s3, br.up4(z4)], 1))
        x = br.up2(torch.cat([s2, x], 1))
        x = br.up1(torch.cat([s1, x], 1))
        return br.out(torch.cat([s0, x], 1))

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        """x: (B, 3, H, W) normalised, H and W multiples of 16 -> NCHW logits dict."""
        cls, zs = self.encoder(x)
        hp, wp = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        z1, z2, z3, z4 = (z[:, self.encoder.num_prefix_tokens:].transpose(1, 2).reshape(-1, self.embed_dim, hp, wp)
                          for z in zs)
        skips = (self.skip0(x), self.skip1(z1), self.skip2(z2), self.skip3(z3), z4)
        f_tp = self._decode(skips, self.tp_branch)
        out = {
            "np": self.np_head(self._decode(skips, self.np_branch)),
            "hv": self.hv_head(self._decode(skips, self.hv_branch)),
            "tp": self.tp_head(f_tp),
            "tissue": self.tissue_head(cls),
        }
        if hasattr(self, "dead_branch"):
            f_dead = self._decode(skips, self.dead_branch)
            out["np_dead"] = self.dead_np_head(f_dead)
            out["hv_dead"] = self.dead_hv_head(f_dead)
        if return_features:
            out["tp_feat"] = f_tp
            out["cls"] = cls
        return out

    def freeze_encoder(self, freeze: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze


def branch_param_count(model: CellViTUNI) -> dict[str, int]:
    """Trainable-capacity map per decoder-side group (encoder excluded)."""
    groups = {}
    for name in ("skip0", "skip1", "skip2", "skip3", "np_branch", "hv_branch",
                 "tp_branch", "dead_branch"):
        mod = getattr(model, name, None)
        if mod is not None:
            groups[name] = sum(p.numel() for p in mod.parameters())
    return groups


def solve_c1_widen() -> int:
    """Bottleneck widening whose NP+HV growth best matches the dead-branch parameter count."""
    dsb = sum(branch_param_count(CellViTUNI(uni_ckpt=None, dead_expert=True)).values())
    best, best_gap = 0, float("inf")
    for w in range(0, 257, 32):
        c1 = sum(branch_param_count(CellViTUNI(uni_ckpt=None, widen=w)).values())
        gap = abs(dsb - c1)
        if gap < best_gap:
            best, best_gap = w, gap
    return best
