"""Original point decoder versus fixed per-image group normalization."""
import torch
from torch import nn
import torch.nn.functional as F
import train_proposal_p26 as shared

ARMS = ('C', 'G')
DEPTH = {arm: tuple(range(12, 24)) for arm in ARMS}
GROUPS = 32
EPSILON = 1e-5
NORMALIZED_CHANNELS = (192, 192, 192, 192, 256, 192, 128, 64)


class GroupNormalizedHead(nn.Module):
    """Retain every original convolution and normalize intermediate features."""
    def __init__(self, original_head):
        super().__init__()
        self.proj = original_head.proj
        self.stem = original_head.stem
        self.up = original_head.up
        self.out = original_head.out
        self.norms = nn.ModuleList(nn.GroupNorm(GROUPS, c, eps=EPSILON, affine=True)
                                   for c in NORMALIZED_CHANNELS)

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError('P31 needs four UNI feature maps')
        projected = [norm(proj(feat)) for norm, proj, feat in
                     zip(self.norms[:4], self.proj, feats)]
        x = F.relu(self.norms[4](self.stem(torch.cat(projected, 1))))
        for layer, norm in zip(self.up, self.norms[5:]):
            x = F.relu(norm(layer(F.interpolate(x, scale_factor=2, mode='nearest'))))
        return F.interpolate(self.out(x), size=(256, 256), mode='bilinear',
                             align_corners=False)[:, 0]


class Proposal(shared.Proposal):
    def __init__(self, weights, trainable=True, arm='C'):
        if trainable is not True or arm not in ARMS:
            raise ValueError('P31 requires trainable C/G')
        super().__init__(weights, True, 'C')
        if arm == 'G':
            self.head = GroupNormalizedHead(self.head)
        self.arm = arm


def run_config(arm, seed, microbatch=4):
    if arm not in ARMS:
        raise ValueError('invalid P31 arm')
    config = shared.run_config('C', seed, microbatch)
    config.update(experiment='P31', arm=arm,
                  architecture='original_UNI_last12' if arm == 'C' else 'UNI_last12_groupnorm_head',
                  decoder_normalization='none' if arm == 'C' else 'GroupNorm32',
                  normalization_eps=EPSILON if arm == 'G' else None,
                  normalization_channels=list(NORMALIZED_CHANNELS) if arm == 'G' else [])
    return config


def main(argv=None):
    args = shared.parse_args(argv, arms=ARMS)
    shared.train(args, run_config(args.arm, args.seed, args.microbatch), Proposal, 'C',
                 experiment='P31')


if __name__ == '__main__':
    main()
