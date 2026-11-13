"""Compare contiguous and distributed UNI adaptation at equal parameter counts."""

import train_proposal_p26 as shared
import train_proposal_p27 as p27


ARMS = ("C", "I")
DEPTH = {"C": p27.DEPTH["C"], "I": tuple(range(1, 24, 2))}


def configure_encoder_depth(encoder, arm):
    if arm not in ARMS or len(encoder.blocks) != 24:
        raise ValueError("P28 requires C/I and a 24-block encoder")
    encoder.requires_grad_(False)
    for index in DEPTH[arm]:
        encoder.blocks[index].requires_grad_(True)
    encoder.eval()


class Proposal(p27.Proposal):
    def __init__(self, weights, trainable=True, arm="C"):
        if trainable is not True or arm not in ARMS:
            raise ValueError("P28 requires trainable C/I")
        super().__init__(weights, trainable, "C")
        if arm == "I":
            configure_encoder_depth(self.encoder, arm)
        self.arm = arm


def run_config(arm, seed, microbatch=4):
    if arm not in ARMS:
        raise ValueError("invalid P28 arm")
    config = p27.run_config("C", seed, microbatch)
    config.update(experiment="P28", arm=arm, trainable_encoder_blocks=list(DEPTH[arm]))
    if arm == "I":
        config["architecture"] = "UNI_interleaved_odd12"
    return config


def main(argv=None):
    args = shared.parse_args(argv, arms=ARMS)
    shared.train(args, run_config(args.arm, args.seed, args.microbatch), Proposal, "C", experiment="P28")


if __name__ == "__main__":
    main()
