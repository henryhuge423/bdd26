# P20 closeout and P21 research rationale — 30 October 2026

All four P20 runs completed 130 epochs,SLURM COMPLETED/0:0,empty stderr. The fixed512working-scale recipe fails the original admission conjunction in both seeds and closes. It improves some internalDead matches but does not produce an admissible detector. P21 is a separately bounded exploratory auxiliary-supervision recipe chosen after this review; its formal submission is recorded separately once verified.

## Evidence and protocol

We collected 42 small files,verified the deployed 426 source files against the sealed manifest `1bd6e680e1b4c008f11b97ee50c165c21c4776209f718a915aca101333228e58`,checked payload byte lengths/SHA256 against preservation manifests,weights/input/tissue/cohort identities,exact fit2134/audit522row alignment,all 130 consecutive epochs,134 optimizer updates  per epoch,effective 16 / physical 4 and native tail6. The source commit was08ae279; formal arrays C55357457 and U55357476. Their task-level elapsed times were4:26:32, 4:34:07, 17:32:12 and17:54:49. SLURM step elapsed and script elapsed differ because staging,hashing,checkpoint copying and cleanup are outside pure training.

Original C encodes 256 into 16×16 tokens; U bilinearly enlarges the same augmentedRGB  to 512,obtains 32×32 tokens,and retains native 256 targets/logits and original peak decoding. No new optical information is introduced. The recipe changes tokenization,position interpolation,attention and internal decoder resolution together. It is not a pure token-density causal intervention. The original loss,UNIlast 12 adaptation,130 epochs,last checkpoint, cohort and acceptance gate remained fixed after the documented resource-only amendments.

|Run|Audit Dead /145|Audit recall|Unique precision|Small Dead /32|Fit recall|Fit Dead /656|Train hours|Peak GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|C_19|66 (0.455172)|0.592327|0.691443|7|0.978516|629|4.0206|4.0474|
|U_19|79 (0.544828)|0.592571|0.712016|9|0.997567|653|17.0830|8.9959|
|C_1|69 (0.475862)|0.595265|0.696466|8|0.979261|627|4.0090|4.0474|
|U_1|75 (0.517241)|0.575347|0.724432|7|0.998195|655|17.2552|8.9959|

## Original decision, independently reconstructed

|Seed|U−C Dead recall|Original paired95%CI|Dead gain≥.05|Dead recall≥.60|Precision≥.75|Full gate|
|---|---:|---|---|---|---|---|
|19|+0.089655|[+0.071393,+0.098960]|True|False|False|False|
|1|+0.041379|[-0.040816,+0.117241]|False|False|False|False|


The independent1000-draw bootstrap with seed 20261018,stratified by tissue over Dead-positive images, exactly reproduces the original admission implementation. Both seeds pass support,nondegeneracy,relative precision and overall-recall clauses. Both fail absoluteDead recall and absoluteprecision. Seed 19 passes gain and positive lowerCI; seed 1 fails those two additional clauses. Seed 1 overall recall change is−.019918,only aboutone matched object inside the−.02floor: that narrow pass does not establish robust noninferiority. MeanDead gain is+0.065517,but averaging cannot replace the both-seed conjunction. No threshold,seed,epoch or scale rescue follows this failure.

## What improved, and what did not

The fit set becomes nearly saturated: U matches653/655 of 656interiorDead versus629/627forC,with overallfit recall about99.76%. AuditDead increases66→79and69→75 of 145,but remains54.48%/51.72%; the fit–audit gap persists. Audit smallDead changes7→9and8→7 of 32; thus the aggregateDead gain is not uniformly a gain in the predefined small subgroup. This is an aggregate count contrast,not object crossover: saved result rows do not contain the identities of exactly whichGT objects switch status. We do not claim13or6newly recovered objects without losses; those are net changes.

|Tissue|InteriorDead support|C19→U19 matches|C1→U1 matches|
|---|---:|---:|---:|
|Breast|1|0→0|0→1|
|Cervix|15|8→6|6→11|
|Colon|56|17→22|19→18|
|Esophagus|8|5→5|5→2|
|Lung|60|34→44|37→41|
|Stomach|4|1→1|1→1|
|Uterus|1|1→1|1→1|


116/145Dead objects are concentrated in five Lung/Colon images. In seed 19,Lung contributes+10andColon+5net matches whileCervix loses2. Seed 1 differs: Lung+4,Cervix+5,Colon−1,Esophagus−3,Breast+1. Tissue-level percentages from 1–3 positive images are descriptive. The stratified bootstrap conditions on this sparse reused collection; it does not establish independent tissue,patient or future-dataset transfer. In strata with onepositive image,the resampling cannot express between-image variation.

Overall audit matches7256→7259inseed 19and7292→7048inseed 1. Uniqueprecision improves because predictions fall10494→10195and10470→9729; removing candidates is not automatically successful recovery. P20 has neither zero-output collapse nor512candidate saturation; both arms remain nondegenerate. Aggregatefit and audit behavior supports a strongerfit/coverage-vs-transfer description; it does not isolate overfitting as a causal mechanism.

## Measured cost and scientific value

Pure U/Ctraining-time ratios are4.2488and4.3041;peaktrainingallocation grows from about4.05to9.00GiB. Four taskelapsed times sum44.46 GPU hours; formal requested upperbound was96 GPU hours. Same-cluster shared-node timing is not a controlled architecture benchmark and excludes original readiness trials. The prior failed24 h feasibility smoke,800 s validation TIMEOUTand validation-budget increase remain explicitly retained in the P20 submission record. Actual success within36hdoes not erase those failures.

P20 contributes evidence that working scale can alter native point coverage under the tested recipe,with substantial cost and incomplete transfer. It does not supplyPQ,segmentation,multifoldtestperformance or a new admitted method. It also does not prove all resolution-based approaches fail.

## Next bounded question and visual inspection

P21 returns to256and tests whether joint class-specific center supervision changes the shared features enough to improve the unchanged untyped detector. A training-only5-channel auxiliary1×1 head has325 extra parameters and coefficient 1focal loss,using original Gaussian centers for classes1..5and the same complete-effective-batch object denominator. No rare-class reweighting,auxiliary decoding or frozen-feature retyping is introduced. It is a whole auxiliary recipe,including extraobjective/capacity,not a pure semantic causal test.

The protocol fixesC/A×seeds19/1,fresh130 epochs,48 GPU-hour maximum and original admission. One fit-only V100 validation must prove genuine second-step auxiliary gradients into shared features,exact original C parity, AMP,frozen weights,last/final reloading and12 h / memory feasibility before formal submission. Repeated audit use and adaptivechoice are disclosed. A failure closes this fixed auxiliaryrecipe rather than expanding a coefficient/class-balance menu.

At the user's request,each finalrun automatically produces8 deterministic audit and4 fit examples with original RGB,GT centers,predicted main-map points, and unique match/miss/extra panels plus coordinates. Selection usesGTstrata only; visuals cannot select checkpoints or change the gate. Smoke previews contain onlyfitimages afterone epoch and are clearly preliminary. Auxiliary predictions never appear as proposal points. The overlays may help identify localization orfalse-positive patterns,but class-specific displayed recall requiresGTassistance and remains diagnostic.

## Reproduction and limitations

`scripts/analyze_proposal_p20.py --collection PATH --out NEW_JSON` verifies the collectedpayload and regenerates the full numericalanalysis; the command was rerun and compared equal to the original savedanalysis. `P20_NUMBERS.json` contains the verified compactsource. Rawsmallcollection and reproducible sources are retained inLM1ignoredopsand localdelivery,with hashes. No largecheckpoint was independently transferred or rehashed outsideLustre; in-jobpreservation is not an externalarchive. Existingstage6PDFs retain their datedsnapshot. The English/ChineseMarkdownrecords andChinese reader here form this experiment's documentation; a newformalPDFreport is not claimed.

## Additional descriptive influence and count decomposition

Control/treatment false-positive counts are3238→2936inseed 19and3178→2681inseed 1; changes−302/−497 accompany matched-count changes+3/−244. These identify different precision/coverage balances,not automatic recall recovery. Removing each one of the11 Dead-positive audit images in turn gives netDead-recall contrasts ranging+.075000to+.115385inseed 19and+.007692to+.065693inseed 1. In particular,removing the Cervix image carrying+5ofseed 1's+6netmatches leaves+1/130=+.007692. These leave-one-image contrasts describe influence; they are not confidence intervals,patient resampling,an alternative gate or a basis for selecting images. No variants were selected from them.

P21's coefficient 1does not make auxiliary and main loss magnitudes equal. Five additional absent-class/background fields and the shared global gradient clip can change optimization scale and direction. Any future effect belongs to the full fixed auxiliaryobjective/capacity/optimization recipe; comparison against C alone cannot identify semantic supervision as its unique cause.
