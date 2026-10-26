# P7/P8 final development closeout [undecodable fragment: A1 AA] 2026-10-22

P7 RGB gain and P8 image-normalized focal loss both fail the original admission rule in both seeds. All eight fresh runs completed130epochs and SLURM COMPLETED/0:0 with empty stderr. Each experiment returned40smallfiles,36payloads checked against preservation SHA256/bytes; exact configurations,fit2134/audit522,cohort/input/UNI identity and finite130epoch logs match. No checkpoint/model inference was rerun for this analysis.

## P7: RGB gain does not give a stable Dead improvement

|Seed / arm|Fit recall|Audit recall|Unique precision|Interior Dead|Small Dead<67|Candidates|
|---|---:|---:|---:|---:|---:|---:|
|19/C|0.975612|0.587918|0.690906|71/145=0.489655|7/32|10424|
|19/J|0.949341|0.594367|0.714244|76/145=0.524138|8/32|10194|
|1/C|0.977398|0.595347|0.693185|64/145=0.441379|6/32|10521|
|1/J|0.952833|0.599510|0.722977|62/145=0.427586|4/32|10158|

Original paired tissue-stratified1000draw95%CI: seed19 J-C+.034483,[+.005435,+.079646]; seed1-.013793,[-.067901,+.043034]. Both fail: J Dead recall.524138/.427586<.60,gain<.05,precision.714244/.722977<.75; seed1 also fails positive-CI requirement. Support,nondeg and relative overall/precision floors pass. No candidate cap saturation. SmallDead changes+1/-2 are descriptive.

J improves overall recall/unique precision in both seeds while reducing fit recall. This is compatible with regularization but does not establish a stain mechanism or rare-class reliability. The single seed19 interval from the partial report cannot stand in for the failed two-seed rule. Close this exact RGB recipe without expansion.

## P8: equal image weighting lowers fit and held-out recall

|Seed / arm|Fit recall|Audit recall|Unique precision|Interior Dead|Small Dead<67|Candidates|
|---|---:|---:|---:|---:|---:|---:|
|19/C|0.976318|0.593388|0.690773|67/145=0.462069|9/32|10523|
|19/I|0.881730|0.552490|0.686479|62/145=0.427586|8/32|9859|
|1/C|0.980498|0.598939|0.689438|63/145=0.434483|6/32|10642|
|1/I|0.887125|0.549551|0.696606|56/145=0.386207|5/32|9664|

I-C Dead changes-.034483/-.048276; original95%CIs[-.093750,0]/[-.076923,-.032895]. I absolute Dead.427586/.386207 and precision.686479/.696606 fail; both also lose overall recall by>.02 (.040898/.049388). No cap saturation. SmallDead decreases1/1. Close this exact objective at the frozen budget.

Fit recall drops from.976318/.980498 to.881730/.887125. The recipe changes both relative image influence and objective scale,including empty-image negatives; this result does not identify which component caused deterioration or reject all normalization methods. Loss magnitudes from different objectives are not quality comparisons.

## Interpretation and next bounded experiment

P3-P8 have not yielded an admitted proposal method. High fit recall versus audit~.59 remains compatible with a generalization deficit,but repeated controls differ across runs/hardware and long training is not bitwise reproducible. Cross-round C scores cannot be used as paired controls. Only within-round paired results enter admission. The tested P4foreground/P5grid/P6bandwidth/P7RGB/P8normalization recipes remain closed. No original P3PhaseB,mask stage or three-split confirmation is opened.

P9 is independently frozen after these development summaries: original C versus fixed4xpositive focal weight at annotated Dead center pixels,D,seeds19/1,four fresh130epoch runs. Original negatives,instance-count denominator,data,model,augmentation,optimizer,last checkpoint,decode and admission remain. This differs from historical failed dense-model Dead-pixel reweighting; neither a class-imbalance nor gradient-conflict cause is established. The new recipe tests a bounded alternative,not the whole reweighting family. See [design](../superpowers/specs/2026-10-22-proposal-p9-design.md).

## Limits and provenance

Point matching within3pixels is not PQ or segmentation performance. Results reuse522fold1audit images with145interiorDead in11images,not an independent validation cohort; two seeds and image rather than patient splitting limit evidence. No fold2/fold3 model evaluation or new test read. Selection and stopping thresholds were not retuned.

P7 array55176183,source73190915d67f658d4573e5f601b9cdc48a0f0079,manifest83947384b842c3989fda71e2c6663b2fb958742e567dc113057003e23f5a9a45. P8 array55191961,source0a62bf8d4cc38bbe44e9ce2e992fff8fb20e6dc9,manifest795fb49c4ce4d947f5e19c9481ae4d79901c3a4def92a468737812a3574c8864. Last checkpoints/outputs remain in source project and scratch directories; small-file hashes do not independently rehash multi-GB checkpoints. Lustre is not archival backup.

P7 task elapsed4:15:07/4:13:53/4:11:25/4:14:25; P8 task elapsed4:12:13/4:08:16/4:13:54/4:14:57. Training peak4.047GiB each. Shared-node timings are operational measurements,not controlled speed comparisons.
