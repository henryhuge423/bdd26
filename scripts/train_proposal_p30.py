"""Compare late and middle UNI adaptation at equal parameter counts."""

import train_proposal_p26 as shared
import train_proposal_p27 as p27


ARMS = ("C", "M")
DEPTH = {"C": p27.DEPTH["C"], "M": tuple(range(6, 18))}


def configure_encoder_depth(encoder, arm):
    if arm not in ARMS or len(encoder.blocks) != 24:
        raise ValueError("P30 requires C/M and a 24-block encoder")
    encoder.requires_grad_(False)
    for index in DEPTH[arm]:
        encoder.blocks[index].requires_grad_(True)
    encoder.eval()


class Proposal(p27.Proposal):
    def __init__(self, weights, trainable=True, arm="C"):
        if trainable is not True or arm not in ARMS:
            raise ValueError("P30 requires trainable C/M")
        super().__init__(weights, trainable, "C")
        if arm == "M":
            configure_encoder_depth(self.encoder, arm)
        self.arm = arm


def run_config(arm, seed, microbatch=4):
    if arm not in ARMS:
        raise ValueError("invalid P30 arm")
    config = p27.run_config("C", seed, microbatch)
    config.update(experiment="P30", arm=arm, trainable_encoder_blocks=list(DEPTH[arm]))
    if arm == "M":
        config["architecture"] = "UNI_middle12"
    return config


def main(argv=None):
    args = shared.parse_args(argv, arms=ARMS)
    shared.train(args, run_config(args.arm, args.seed, args.microbatch), Proposal, "C", experiment="P30")


if __name__ == "__main__":
    main()
