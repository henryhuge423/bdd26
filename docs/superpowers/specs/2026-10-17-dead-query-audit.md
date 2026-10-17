# E8b: canonical train-Dead query audit, frozen before inference

Use all65 fold1 original images with any Dead label, selected from canonical labels only,967 Dead objects including borders. Use original E7 Q_seed19/Q_seed1 final weights, no fitting, no A2 weights, no augmentation, batch1, original thresholds and decoder. Do not change E8a or its failed verdict.

Re-use verified E8a cache/matching and existing overlap/decode/original_full functions. Record allGT class supports within these65 images, separate Dead interior/border, and both canonical-assignment and all-query-best overlap. For every Dead object record raw assigned IoU, best raw IoU, best accepted raw IoU, best accepted-correct-type raw IoU, assigned query class/retention, final best IoU/type. Freeze IoU>.5 success and IoU<.1 miss. Include canonical class confusion and final typed versus untyped counts. No official performance claim: this is a GT-selected training cohort.

Validate967 objects in each seed, image identities, query/target identities, confusion sums, class support and raw>=accepted>=typed-accepted upper bounds; reconstruct Dead stage counts independently from per-object rows. Use only fold1 arrays; inspect saved fold2 provenance strings without opening fold2/3 arrays. Hash-check original checkpoints/source/vendor/probe helper, record driver hash. No learned selection, threshold search, optimization, extra seeds or confirmation. All outcomes are descriptive; no admission or causal claims.

One fresh LM2 GPU4 process, >=28GiB actual free at admission,30% memory cap,20-minute hard wall limit, <=8MiB JSON outputs,20GiB disk reserve. No saved predictions/weights, no environment changes, no new SSH. Preserve existing jobs and the closed E8a outputs. Any failure stops the audit; no automatic replay.
