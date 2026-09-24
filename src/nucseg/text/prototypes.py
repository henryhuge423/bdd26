"""Frozen CONCH text prototypes for the nucleus classes (RESEARCH_PLAN.md pillar A).

Encodes `configs/text/nuclei_prompts.yaml` with the CONCH v1 text tower into L2-normalised 512-d
prototypes in three flavours (the pillar-A ablation axis):
  names   (C, D)      class names x templates, averaged
  desc    (C, D)      morphological descriptions x templates, averaged
  tissue  (T, C, D)   descriptions "... in <tissue>" x templates, averaged per tissue
plus the per-prompt embeddings (`desc_all`, (C, K, D)) for multi-prototype heads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from ..constants import CLASS_NAMES, TISSUES

ROOT = Path(__file__).resolve().parents[3]
CONCH_ROOT = ROOT / "third_party" / "CONCH"
PROMPTS = ROOT / "configs" / "text" / "nuclei_prompts.yaml"
CONCH_CKPT = ROOT / "weights" / "MahmoodLab" / "CONCH" / "pytorch_model.bin"
TYPE_NAMES = ["Background", *CLASS_NAMES]  # index = type id (0 = background)


def load_conch(device="cuda"):
    if str(CONCH_ROOT) not in sys.path:
        sys.path.insert(0, str(CONCH_ROOT))
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(CONCH_CKPT), device=device)
    model.eval()
    tok = get_tokenizer()
    return model, preprocess, (lambda texts: tokenize(tok, texts).to(device))


@torch.no_grad()
def encode_texts(model, tokenize, texts: list[str], bs: int = 256) -> torch.Tensor:
    return torch.cat([model.encode_text(tokenize(texts[i:i + bs])) for i in range(0, len(texts), bs)]).float()


def _mean_norm(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nn.functional.normalize(x.mean(dim), dim=-1)


@torch.no_grad()
def build_prototypes(device="cuda", prompts: Path = PROMPTS) -> dict:
    cfg = yaml.safe_load(open(prompts))
    templates = cfg["templates"]
    model, _, tokenize = load_conch(device)

    def enc(phrases: list[str]) -> torch.Tensor:  # (P, T, D)
        texts = [t.format(p) for p in phrases for t in templates]
        return encode_texts(model, tokenize, texts).view(len(phrases), len(templates), -1)

    names, desc, desc_all, tissue = [], [], [], []
    for c in TYPE_NAMES:
        names.append(_mean_norm(enc(cfg["classes"][c]["names"]).flatten(0, 1), 0))
        e = enc(cfg["classes"][c]["descriptions"])               # (K, T, D)
        desc_all.append(_mean_norm(e, 1))                        # (K, D)
        desc.append(_mean_norm(e.flatten(0, 1), 0))
    for t in TISSUES:
        tn = cfg["tissues"][t]
        tissue.append(torch.stack([
            _mean_norm(enc([f"{d} in {tn}" for d in cfg["classes"][c]["descriptions"]]).flatten(0, 1), 0)
            for c in TYPE_NAMES]))
    return {"names": torch.stack(names).cpu(), "desc": torch.stack(desc).cpu(),
            "desc_all": torch.stack(desc_all).cpu(), "tissue": torch.stack(tissue).cpu(),
            "type_names": TYPE_NAMES, "tissues": TISSUES, "prompts": cfg}
