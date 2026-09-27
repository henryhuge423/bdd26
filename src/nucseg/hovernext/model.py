"""HoVer-NeXt model, vendored from third_party/hover_next_inference/src/multi_head_unet.py.

The segmentation_models_pytorch pieces are replaced with local equivalents so the nuclei env
does not need an smp install (whose timm pin would fight CellViT's). The module tree —
encoder / decoders[i].blocks[j].conv{1,2}[k] / heads[i][0] — is preserved exactly, so the
official Zenodo per-fold checkpoints load with strict=True. Initialisation stubs are no-ops
because every parameter comes from the checkpoint.

Output channel layout (inst_channels=5, out_channels_cls=6 for PanNuke):
[0:2] regression, [2:5] background/foreground/boundary softmax, [5:] class softmax whose
channels 0..4 follow CLASS_NAMES order and channel 5 is background.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1,
                 use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=not use_batchnorm)
        super().__init__(conv, nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                         nn.ReLU())


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
                         nn.Identity())


class TimmEncoderFixed(nn.Module):
    """Modified version of timm encoder (github.com/huggingface/pytorch-image-models)."""

    def __init__(self, name, pretrained=True, in_channels=3, depth=5, output_stride=32,
                 drop_rate=0.5, drop_path_rate=0.25):
        super().__init__()
        self.model = timm.create_model(
            name, in_chans=in_channels, features_only=True, pretrained=pretrained,
            out_indices=tuple(range(depth)), drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        self._in_channels = in_channels
        self._out_channels = [in_channels] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x):
        return [x] + self.model(x)

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2 ** self._depth)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, padding=1,
                                use_batchnorm=use_batchnorm)

    def forward(self, x, skip=None):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, use_batchnorm=False,
                 center=False, next=False):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError("Model depth does not match decoder_channels length")
        encoder_channels = encoder_channels[1:][::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        self.center = nn.Identity()
        blocks = [DecoderBlock(i, s, o, use_batchnorm=use_batchnorm)
                  for i, s, o in zip(in_channels, skip_channels, out_channels)]
        if next:  # extra up-block halves resolution beyond the vanilla UNet
            blocks.append(DecoderBlock(out_channels[-1], 0, out_channels[-1] // 2,
                                       use_batchnorm=use_batchnorm))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:][::-1]
        head, skips = features[0], features[1:]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class MultiHeadModel(nn.Module):
    def __init__(self, encoder, decoder_list, head_list):
        super().__init__()
        self.encoder = nn.ModuleList([encoder])[0]
        self.decoders = nn.ModuleList(decoder_list)
        self.heads = nn.ModuleList(head_list)

    def forward(self, x):
        features = self.encoder(x)
        masks = []
        for decoder, head in zip(self.decoders, self.heads):
            masks.append(head(decoder(*features)))
        return torch.cat(masks, 1)


def get_model(enc="convnextv2_tiny.fcmae_ft_in22k_in1k", out_channels_cls=6,
              out_channels_inst=5, pretrained=True):
    depth = 4 if "next" in enc else 5
    encoder = TimmEncoderFixed(name=enc, pretrained=pretrained, in_channels=3, depth=depth,
                               output_stride=32, drop_rate=0.5, drop_path_rate=0.0)
    decoder_channels = (256, 128, 64, 32, 16)[:depth]
    kwargs = dict(encoder_channels=encoder.out_channels, decoder_channels=decoder_channels,
                  n_blocks=len(decoder_channels), use_batchnorm=False, center=False,
                  next="next" in enc)
    # blocks[-1].conv2[0].out_channels: the extra "next" block halves the last width
    last = decoder_channels[-1] // 2 if kwargs["next"] else decoder_channels[-1]
    model = MultiHeadModel(encoder,
                           [UnetDecoder(**kwargs), UnetDecoder(**kwargs)],
                           [SegmentationHead(last, out_channels_inst),
                            SegmentationHead(last, out_channels_cls)])
    return model
