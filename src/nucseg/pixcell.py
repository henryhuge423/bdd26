"""Shared helpers for the PixCell-256 Cell-ControlNet pipeline (pillar B).

Loading convention: the ControlNet pipeline, the SD3.5 VAE and UNI2-h weights all live under
weights/ (LM1 paths; synced to LM2 by scripts/push_lm2.sh --weights). The conditioning protocol
follows the model card: controlnet_input = 3-channel binary mask (0/255 RGB), uni_embeds =
UNI2-h CLS embedding of the paired context image, CFG negative = the learned uncond_embedding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"
PIXCELL_DIR = WEIGHTS / "StonyBrook-CVLab" / "PixCell-256-Cell-ControlNet"
UNI2H_BIN = WEIGHTS / "MahmoodLab" / "UNI2-h" / "pytorch_model.bin"

# LoRA targets: attention projections of the PixArt-style DiT blocks (self- and cross-attn)
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def load_uni2h(device="cuda"):
    import timm
    import torchvision.transforms as T

    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    m = timm.create_model("vit_giant_patch14_224", pretrained=False, **kw)
    sd = torch.load(UNI2H_BIN, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith("head")], f"missing {missing[:5]}"
    m.eval().to(device)
    # UNI2-h cfg (its HF config.json): resize 224 (crop_pct 1), ImageNet mean/std
    tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return m, tf


@torch.inference_mode()
def uni_embed(model, tf, img, device="cuda") -> torch.Tensor:
    """uint8 HWC / PIL -> (1, 1, 1536) fp32 CLS embedding."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return model(tf(img).unsqueeze(0).to(device)).unsqueeze(1)


def load_pipeline(device="cuda", dtype=torch.float32):
    """PixCell-256 Cell-ControlNet pipeline (fp32: the pipeline hardcodes fp32 controlnet input)."""
    from diffusers import DiffusionPipeline

    sys.path.insert(0, str(ROOT / "third_party" / "pixcell_pipeline"))
    pipe = DiffusionPipeline.from_pretrained(
        PIXCELL_DIR, custom_pipeline=str(ROOT / "third_party" / "pixcell_pipeline"),
        trust_remote_code=True, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def wrap_lora(module, rank: int = 16, alpha: int = 16, extra_targets: tuple = ()):
    """Wrap a PixCell module (transformer or controlnet) with a freshly-initialised LoRA adapter."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=list(LORA_TARGETS) + list(extra_targets))
    pm = get_peft_model(module, cfg)
    pm.print_trainable_parameters()
    return pm


def save_lora(pm, path: Path):
    from peft.utils.save_and_load import get_peft_model_state_dict

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(pm), path)


def load_lora(pm, path: Path):
    """Load adapter weights (saved by save_lora) into an existing LoRA-wrapped module."""
    from peft.utils.save_and_load import set_peft_model_state_dict

    set_peft_model_state_dict(pm, torch.load(path, map_location="cpu"))


def mask_rgb(inst: np.ndarray) -> np.ndarray:
    """Instance map -> 3-channel 0/255 binary mask (the ControlNet condition format)."""
    m = np.where(inst > 0, 255, 0).astype(np.uint8)
    return np.stack([m] * 3, -1)


@torch.inference_mode()
def encode_condition(pipe, mask3: np.ndarray) -> torch.Tensor:
    """VAE latent of the mask condition, exactly as the pipeline does at inference time."""
    dev = next(pipe.vae.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(mask3) / 255.0).float().to(dev).permute(2, 0, 1).unsqueeze(0)
    x = 2 * (x - 0.5)
    lat = pipe.vae.encode(x).latent_dist.mean
    return (lat - getattr(pipe.vae.config, "shift_factor", 0)) * pipe.vae.config.scaling_factor


@torch.inference_mode()
def encode_image(pipe, img_u8: np.ndarray, seed: int | None = None) -> torch.Tensor:
    """uint8 HWC RGB -> scaled VAE latent (1, 16, 32, 32); sampled with a per-image seeded generator."""
    dev = next(pipe.vae.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(img_u8) / 255.0).float().to(dev).permute(2, 0, 1).unsqueeze(0)
    x = 2 * (x - 0.5)
    g = torch.Generator(dev).manual_seed(seed if seed is not None else 0)
    lat = pipe.vae.encode(x).latent_dist.sample(generator=g)
    return (lat - getattr(pipe.vae.config, "shift_factor", 0)) * pipe.vae.config.scaling_factor


def reinhard_lab(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Reinhard colour transfer (LAB mean/std of src -> tgt). Fixes the generator's global colour
    gap: paired-context base generations go from od_mean_l1 .072 to .009 vs the real target."""
    import cv2

    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    m_s, s_s = lab.mean(0), lab.std(0)
    lab_t = cv2.cvtColor(tgt, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    m_t, s_t = lab_t.mean(0), lab_t.std(0)
    out = (lab - m_s) / np.maximum(s_s, 1e-3) * s_t + m_t
    return cv2.cvtColor(np.clip(out.reshape(src.shape), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


@torch.no_grad()
def batched_generate(pipe, embs, mask_rgbs, steps, guidance, seeds, amp=False):
    """PixCellControlNetPipeline semantics, batched. CFG uncond rows get ZERO controlnet
    contribution (identical to the pipeline's controlnet_outputs=None uncond pass, because the
    block injection is additive). embs (B,1,1536) fp32; mask_rgbs list of HWC uint8."""
    dev = embs.device
    B = len(embs)
    sch = pipe.scheduler
    sch.set_timesteps(steps, device=dev)
    raw = pipe.transformer.caption_projection.uncond_embedding.detach().to(dev)
    uncond = raw[None].expand(B, -1, -1) if raw.ndim == 2 else raw.expand(B, -1, -1)

    mlat = torch.cat([encode_condition(pipe, m) for m in mask_rgbs])  # (B,16,32,32)
    g = torch.Generator(dev)
    noise = torch.stack([torch.randn(pipe.transformer.config.in_channels, 32, 32, device=dev,
                                     generator=g.manual_seed(int(s))) for s in seeds])
    latents = noise * sch.init_noise_sigma
    for t in sch.timesteps:
        lat_in = torch.cat([latents, latents], 0)
        tt = t.expand(2 * B)
        emb_all = torch.cat([uncond, embs], 0)
        cn_cond = pipe.controlnet(hidden_states=latents, conditioning=mlat,
                                  encoder_hidden_states=embs, timestep=t.expand(B),
                                  return_dict=False)[0]
        cn_all = [torch.cat([torch.zeros_like(b), b], 0) for b in cn_cond]
        pred = pipe.transformer(sch.scale_model_input(lat_in, t), encoder_hidden_states=emb_all,
                                controlnet_outputs=cn_all, timestep=tt, return_dict=False)[0]
        if pred.shape[1] == 2 * latents.shape[1]:
            pred = pred.chunk(2, dim=1)[0]
        u, c = pred.chunk(2, dim=0)
        latents = sch.step(u + guidance * (c - u), t, latents, return_dict=False)[0]
    img = pipe.vae.decode((latents / pipe.vae.config.scaling_factor)
                          + getattr(pipe.vae.config, "shift_factor", 0), return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
    return (img * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
