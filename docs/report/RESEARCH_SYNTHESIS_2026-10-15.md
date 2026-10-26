# Research synthesis and next decisions — 2026-10-15

> Historical synthesis with completion corrections below. For the latest experiments, see
> [P3–P15 synthesis](../results/PROPOSAL_PROGRESS_2026-10-26.md) and
> [current results](../results/README.md). The October 26 prose correction does not add experiments.

> **2026-10-16 E6 closeout:** All four final-checkpoint fold2 runs and frozen analysis complete; Q fails every quality gate (mean interior-Dead matching change -.507422, 95%CI[-.563164,-.451449]) while passing latency. The train-only learning-rate restoration probe supports allocating a separate E7 gamma.99 schedule control, not a generalization or causal claim. [Results and next decisions](../results/SET_MASK_E6_DEV_2026-10-16.md).

> **Latest E5 closeout:** [Fusion dev](../results/FUSION_GATING_DEV_2026-10-15.md) fails its frozen internal-Dead-matching gate (mean−.003374, CI[−.014273,+.007486]), despite official/strict Dead PQ increases. Do not advance to C or confirmation. Earlier suggested gated fusion is no longer a default justified component. [Train-only point-grid audit](../results/PMNET_POINT_GRID_AUDIT_2026-10-15.md) adds legal-target/collision constraints. Next architecture remains a hypothesis: isolate instance representation, preserve recipe controls, freeze design and resource/cost criteria before training. Earlier unexecuted-C2 wording below is historical; that study is closed as described above.

> Later 2026-10-15 closeout: all eight C2 evaluations completed; selective-HV mean miss improvement −.008621 [−.040918,+.017699], frozen criterion fails. [Full mechanism report](../results/VISIBILITY_SUPERVISION_MECHANISM_2026-10-15.md); [train-only geometry audit](../results/PMNET_TRAIN_GEOMETRY_AUDIT_2026-10-15.md). The earlier unexecuted-C2 statements below describe the pre-closeout synthesis. No expansion or new training.

## Main conclusion

The project has a reproducible baseline and a sequence of falsifiable mechanism tests; it has not established a broadly superior replacement method. The latest visibility-supervision development round fails its practical quality gates. The next architecture study should target a measurable instance-representation hypothesis rather than assume that removing HV/watershed will fix the observed failures.

## Evidence chain

| Question | Existing evidence | Permitted conclusion |
|---|---|---|
| Is the baseline credible? | Own last-checkpoint HoVer-Net and CellViT-UNI, official three-split evaluator | Baseline and evaluation assets exist; released-model selection differences remain explicit |
| Can frozen-feature retyping repair the deficit? | CONCH and T0–T3 fail their tested rules | Close those tested recipes; GT-assisted typing oracle is not achievable method performance |
| Do candidate additions solve rare-class detection? | EF-P2 retains a small Dead gain at approximately 6.1× inference cost; E3 fails strict-Dead gates | Separate mechanism evidence from deployment value and strict-metric risk |
| Is the HV area threshold the whole scale mechanism? | E2a preserves scale effects under matched support | Lowering the threshold alone does not explain the observed scale tradeoff |
| Is a specialist branch justified by gradient conflict? | DSB Gate B finds no negative composite gradients in its measured batches | Close the preregistered specialist route; this is not a universal architecture theorem |
| Are labels certain? | E1-C/E1-T detector resonance, M metric response, unavailable experts | Retain annotation uncertainty; neither agreement nor merged-label gains prove biological ground truth |
| Does selective visible-fragment HV improve original-image results? | Eight runs complete; A2−A1 official and strict Dead decrease in both seeds | No incremental performance support on these endpoints; original-image admission fails |
| Was the C2 mechanism tested after training? | All eight fixed-C2 evaluations completed; selective-HV mean miss improvement −.008621 [−.040918,+.017699] | The frozen mechanism clause fails; see the linked closeout, which supersedes the earlier unexecuted status |

Older evidence is synthesized from existing stage reports, not newly reproduced here. The latest original-image arithmetic was recomputed from eight stored summaries and bound to SHA256 values in the linked result record.

## What the new round changes

A1−A0 mean official Dead is +.03765, but mPQ/bPQ are −.00528/−.00612. A2−A1 mean Dead and strict Dead are −.01308/−.00978. Thus an official Dead increase relative to baseline must not be attributed to the selective HV change: stronger cropping already has a larger official Dead increase, accompanied by quality costs. These summaries do not establish significance, and do not substitute for the specified C2 mechanism endpoint.

## PM-Net design decisions to settle before training

The proposal remains a hypothesis, with minimum/target/stretch scope. Preserve the shared UNI initialization and unfreezing schedule, native resolution, matched seed pairs, last checkpoints and official/strict reporting.

1. Fix one-to-one assignment costs and normalization, including rare-class weighting, before examining validation performance.
2. Set query capacity from training-fold object counts with a fixed safety margin; report overflow rather than choose capacity from validation failures.
3. Define an interior point for truncated or non-convex masks, point-selection supervision, duplicates, and deterministic overlap resolution. Test tiny, single-axis, empty and crowded masks.
4. Compare basic queries, point initialization and gated fusion separately, with a parameter-matched plain adapter. Avoid combining visibility supervision or auxiliary type grouping into the first architecture comparison.
5. Implement augmentation-worker resume state before a new round. Demonstrate resumed versus uninterrupted sampling and target equivalence, then freeze code/config/input hashes.
6. Retain the proposal's per-seed quality floors and ≤2× end-to-end latency target. Establish storage and runtime capacity before allocation. The current visibility stop rule does not authorize a new training round.

## Deliverables and outstanding work

The updated progress deck and proposal include the full eight-run status. The negative dev result can now enter the final report. C2 is complete and fails its frozen mechanism criterion; expert review remains unavailable; team proposal contributions remain TODO. No test-fold access, extra seeds, confirmation training, worker connection or publishing is performed in this update.

Sources: [latest dev results](../results/VISIBILITY_SUPERVISION_DEV_2026-10-15.md), [stage V](stage_report_5/README.md), [stage IV](stage_report_4/README.md), [stage III](stage_report_3/README.md), [frozen visibility spec](../superpowers/specs/2026-10-14-visibility-supervision-design.md), [PM-Net concept](../proposals/RESEARCH_NEXT_2026-10-14-new-model-route.md).
