# P28 validation and submission status

- The frozen comparison assigns 12 UNI blocks to each arm, with the original last 12 in C and odd-numbered blocks 1–23 in I. Each arm uses seeds 19 and 1. The design and implementation are recorded in source commit `c7b486781484229f727ede4ef58bad2740c98da2`.
- The new test detected the missing entry point before implementation. LM1 subsequently passed 107 focused CPU checks in 51.99 seconds, with module imports verified from the isolated worktree.
- Expanse admission stopped before the GPU smoke submission. The live `jhu159` allocation was 1,900 GPU SU, with 1,739 spent and 334.43 reserved at the full limits of the existing queue. The next round requests up to 0.5 GPU-hours for the smoke and 48 GPU-hours for the formal array. At that queue snapshot the combined request required about 222 additional SU. `jhu159` was the sole associated account. The 21:25 EST scheduler history contained no P28 job, and its three planned run directories were absent.
- Resume after fresh accounting and queue checks show headroom. Verify the original attempt and preserve its logs. Then perform source and storage admission, submit the V100 validation once and verify its readiness record before submitting the formal array.

Cluster partition and charging details are documented by SDSC. [Expanse user guide](https://www.sdsc.edu/systems/expanse/user_guide.html)
