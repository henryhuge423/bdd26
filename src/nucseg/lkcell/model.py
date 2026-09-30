"""LKCell (hustvl/LKCell, arXiv 2407.18054) inference model, vendored for the strict-protocol
re-evaluation (line A, findings 2026-09-28/30): UniRepLKNet encoder + RepLK decoder + the
CellViT 3-branch head (np/hv/tp) + tissue head.

Construction mirrors their inference exactly (models/segmentation/cell_segmentation/
cellvit.py::CellViT with its default unireplknet_s backbone, incl. the inherited default
UniRepLKNet body that their released model_best.pth also carries — both key groups must
exist for a strict load). Everything except the network definition is ours: LKCellAdapter
maps their NCHW logits onto the NHWC np/tp/hv/tissue contract of
nucseg.cellvit.engine.predict_fold (shared HoVer-Net decode, identical constants to the
CellViT-UNI baseline — only the network changes). NOTE normalization: LKCell trains with
albumentations Normalize(0.5, 0.5)/(0.5, 0.5) (their run config `transformations.normalize`),
NOT UNI ImageNet stats — pass mean=std=(0.5,0.5,0.5) to predict_fold.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.blocks import UpSample
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets.basic_unet import TwoConv, UpCat
from monai.utils import InterpolateMode
from typing import List, Optional, Sequence, Tuple, Union

from .replknet import ConvFFN, RepLKBlock, conv_bn
from .unireplknet import UniRepLKNet

encoder_feature_channel = {
    "unireplknet_n": (80, 160, 320, 640),
    "unireplknet_s": (96, 192, 384, 768),
    "resnet50": (64, 256, 512, 1024, 2048),
    "vitdet_small": (768, 768, 768, 768, 768),
}


def _get_encoder_channels_by_backbone(backbone: str, in_channels: int = 3) -> tuple:
    encoder_channel_tuple = encoder_feature_channel[backbone]
    return tuple([in_channels] + list(encoder_channel_tuple))


class RepLKDeocder(nn.Module):  # name verbatim from upstream
    def __init__(
        self,
        encoder_channels: Sequence[int],
        spatial_dims: int,
        decoder_channels: Sequence[int],
        stage_lk_sizes,
        drop_path,
        upsample: str,
        pre_conv: Optional[str],
        interp_mode: str,
        align_corners: Optional[bool],
        small_kernel,
        dw_ratio: int = 1,
        small_kernel_merged=False,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        dropout: Union[float, tuple] = 0.0,
        bias: bool = False,
        is_pad: bool = True,
        ffn_ratio=4,
    ):
        super().__init__()

        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1)
        halves.append(False)
        stage_lk_sizes = stage_lk_sizes
        blocks = []
        for in_chn, skip_chn, out_chn, halve in zip(in_channels, skip_channels, decoder_channels, halves):
            blocks.append(
                UpCat(
                    spatial_dims=spatial_dims,
                    in_chns=in_chn,
                    cat_chns=skip_chn,
                    out_chns=out_chn,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    upsample=upsample,
                    pre_conv=pre_conv,
                    interp_mode=interp_mode,
                    align_corners=align_corners,
                    halves=halve,
                    is_pad=is_pad,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        repblock = []
        for i in range(4):
            repblock.append(RepLKBlock(in_channels=in_channels[i], dw_channels=int(in_channels[i] * dw_ratio),
                                       block_lk_size=stage_lk_sizes[i], small_kernel=small_kernel,
                                       drop_path=drop_path, small_kernel_merged=small_kernel_merged))
        self.repblock = nn.ModuleList(repblock)

        convffnblock = []
        for i in range(4):
            convffnblock.append(ConvFFN(in_channels=in_channels[i],
                                        internal_channels=int(in_channels[i] * ffn_ratio),
                                        out_channels=in_channels[i], drop_path=drop_path))
        self.convffnblock = nn.ModuleList(convffnblock)

        self.upsample = [UpSample(
            spatial_dims,
            decoder_channels[i],
            decoder_channels[i],
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners, ) for i in range(len(decoder_channels) - 1)]

        self.upsample1 = UpSample(
            spatial_dims, 256, 256, 2, mode=upsample, pre_conv=pre_conv,
            interp_mode=interp_mode, align_corners=align_corners,
        )
        self.upsample2 = UpSample(
            spatial_dims, 128, 128, 2, mode=upsample, pre_conv=pre_conv,
            interp_mode=interp_mode, align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, 304, decoder_channels[-2], act, norm, bias, dropout)
        self.convs1 = TwoConv(spatial_dims, 152, decoder_channels[-1], act, norm, bias, dropout)

    def forward(self, features: List[torch.Tensor], input_feature: torch.Tensor, skip_connect: int = 3):
        skips = features[:-1][::-1]
        features = features[1:][::-1]
        x = features[0]
        for i, (block, repblock, convffnblock) in enumerate(zip(self.blocks, self.repblock, self.convffnblock)):
            if i < skip_connect:
                skip = skips[i]
                x = block(x, skip)
            else:
                skip = input_feature[1]
                x = self.upsample1(x)
                x = torch.cat([skip, x], dim=1)
                x = self.convs(x)
                skip = input_feature[0]
                x = self.upsample2(x)
                x = torch.cat([skip, x], dim=1)
                x = self.convs1(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        act: Optional[Union[Tuple, str]] = None,
        scale_factor: float = 1.0,
    ):
        conv_layer = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        bn_layer = nn.BatchNorm2d(in_channels)
        conv_layer1 = conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=1,
                              stride=1, padding=0, groups=1)
        nonlinear_layer = nn.GELU()
        conv_layer2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=1, padding=0, groups=1)
        up_layer: nn.Module = nn.Identity()
        if scale_factor > 1.0:
            up_layer = UpSample(
                spatial_dims=spatial_dims,
                scale_factor=scale_factor,
                mode="nontrainable",
                pre_conv=None,
                interp_mode=InterpolateMode.LINEAR,
            )
        if act is not None:
            act_layer = get_act_layer(act)
        else:
            act_layer = nn.Identity()
        super().__init__(bn_layer, conv_layer1, nonlinear_layer, conv_layer2, up_layer)


class CellViT(UniRepLKNet):
    """Verbatim construction from upstream inference (get_model): default-arg UniRepLKNet
    super body (present in released checkpoints) + unireplknet_s encoder + RepLK decoder."""

    def __init__(
        self,
        num_nuclei_classes: int = 6,
        num_tissue_classes: int = 19,
        in_channels: int = 3,
        backbone="unireplknet_s",
        decoder_channels: Tuple = (1024, 512, 256, 128, 64),
        spatial_dims: int = 2,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        dropout: Union[float, tuple] = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        interp_mode: str = "nearest",
        drop_path_rate: float = 0.1,
        large_kernel_sizes=[13, 27, 29, 31],
        small_kernel=5,
    ):
        super().__init__()

        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        if backbone not in encoder_feature_channel:
            raise ValueError(f"invalid model_name {backbone} found, must be one of {encoder_feature_channel.keys()}.")
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")

        self.backbone = backbone
        self.spatial_dims = spatial_dims
        encoder_channels = _get_encoder_channels_by_backbone(backbone, in_channels)
        self.encoder = UniRepLKNet(
            in_chans=3,
            num_classes=None,
            depths=(3, 3, 27, 3),
            dims=(96, 192, 384, 768),
            drop_path_rate=0.3,
            layer_scale_init_value=1e-6,
            head_init_scale=1.,
            kernel_sizes=None,
            deploy=False,
            with_cp=False,
            init_cfg=None,
            attempt_use_lk_impl=True,
            use_sync_bn=False,
        )
        self.decoder = RepLKDeocder(
            encoder_channels=encoder_channels,
            stage_lk_sizes=large_kernel_sizes,
            small_kernel=small_kernel,
            drop_path=drop_path_rate,
            spatial_dims=spatial_dims,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=None,
            align_corners=None,
        )
        self.branches_output = {
            "nuclei_binary_map": 2,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }
        self.nuclei_binary_segmentation_head = SegmentationHead(
            spatial_dims=spatial_dims, in_channels=decoder_channels[-1],
            out_channels=self.branches_output["nuclei_binary_map"], kernel_size=1, act=None, scale_factor=1.0,
        )
        self.hv_map_head = SegmentationHead(
            spatial_dims=spatial_dims, in_channels=decoder_channels[-1],
            out_channels=self.branches_output["hv_map"], kernel_size=1, act=None, scale_factor=1.0,
        )
        self.nuclei_type_maps_head = SegmentationHead(
            spatial_dims=spatial_dims, in_channels=decoder_channels[-1],
            out_channels=self.branches_output["nuclei_type_maps"], kernel_size=1, act=None, scale_factor=1.0,
        )

    def forward(self, x: torch.Tensor) -> dict:
        out_dict = {}
        classifier_logits, z, input_feature = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits
        decoder_output = self.decoder(z, input_feature)
        out_dict["nuclei_binary_map"] = self.nuclei_binary_segmentation_head(decoder_output)
        out_dict["hv_map"] = self.hv_map_head(decoder_output)
        out_dict["nuclei_type_map"] = self.nuclei_type_maps_head(decoder_output)
        return out_dict


class LKCellAdapter(nn.Module):
    """Their NCHW branch dict -> our NHWC np/tp/hv/tissue contract (engine does to_nhwc itself,
    so return NCHW here exactly like CellViTUNI does; keys renamed only)."""

    def __init__(self, m: CellViT):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> dict:
        o = self.m(x)
        return {"np": o["nuclei_binary_map"], "hv": o["hv_map"],
                "tp": o["nuclei_type_map"], "tissue": o["tissue_types"]}


def build_lkcell(state_dict) -> LKCellAdapter:
    model = CellViT()
    msg = model.load_state_dict(state_dict, strict=True)
    return LKCellAdapter(model)
