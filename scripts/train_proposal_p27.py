"""Compare early and late UNI block adaptation under the same parameter budget."""

import train_proposal_p26 as shared


ARMS = ('C', 'E')
DEPTH = {'C': tuple(range(12, 24)), 'E': tuple(range(12))}


def configure_encoder_depth(encoder, arm):
    if arm not in ARMS or len(encoder.blocks) != 24:
        raise ValueError('P27 requires C/E and a 24-block encoder')
    if arm == 'C':
        shared.configure_encoder(encoder, True)
        return
    encoder.requires_grad_(False)
    for index in DEPTH[arm]:
        encoder.blocks[index].requires_grad_(True)
    encoder.eval()


class Proposal(shared.Proposal):
    def __init__(self, weights, trainable=True, arm='C'):
        if trainable is not True or arm not in ARMS:
            raise ValueError('P27 requires trainable C/E')
        super().__init__(weights, True, 'C')
        configure_encoder_depth(self.encoder, arm)
        self.arm = arm


def run_config(arm, seed, microbatch=4):
    if arm not in ARMS:
        raise ValueError('invalid P27 arm')
    config = shared.run_config('C', seed, microbatch)
    config.update(
        experiment='P27',
        arm=arm,
        trainable_encoder_blocks=list(DEPTH[arm]),
        architecture='original_UNI_last12' if arm == 'C' else 'UNI_first12',
    )
    return config


def main(argv=None):
    args = shared.parse_args(argv, arms=ARMS)
    config = run_config(args.arm, args.seed, args.microbatch)
    shared.train(args, config, Proposal, 'C', experiment='P27')


if __name__ == '__main__':
    main()
