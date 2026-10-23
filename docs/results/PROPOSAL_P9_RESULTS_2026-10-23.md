# P9 Dead-center weighting closeout — 2026-10-23

Four fresh C/D x seeds19/1 runs completed130epochs,SLURM55208619 tasks0-3 COMPLETED/0:0,stderr0bytes. **Original admission=false in bothseeds. Close this fixedweight4 recipe.** No extraweight/thresholdmenu,seeds,epochs,PhaseB,masks or fold2/fold3 model evaluation. This is reused fold1 point-development audit,notPQ/test/independentvalidation.

## Matched results

| arm/seed | fit recall | fit internalDead recall | audit recall | audit precision | audit internalDead recall | audit smallDead | candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| C/19 | 0.972375 | 0.949695 | 0.582449 | 0.693190 | 0.475862 | 8/32 | 10293 |
| D/19 | 0.976103 | 0.996951 | 0.594531 | 0.682888 | 0.482759 | 9/32 | 10665 |
| C/1 | 0.978359 | 0.963415 | 0.592816 | 0.692675 | 0.434483 | 6/32 | 10484 |
| D/1 | 0.979536 | 0.995427 | 0.590122 | 0.698994 | 0.482759 | 10/32 | 10342 |

D-C internalDead change seed19+.006896552,95%CI[-.046296296,+.043795620];seed1+.048275862,CI[0,+.092307692]. Originalgate requires both absoluteDead>=.60,gain>=.05,bootstraplower>0,precision>=.75and>=C-.02,recall>=C-.02,nondeg>=.95 and>=10images/100objects. Both fail absolute,gain,CIandabsoluteprecision;relativeprecision,overallrecall,supportandnondeg pass. Mean change+.027586 is descriptive,not a pooled statisticaltest. Audit145internalDead in11images,smallDead32objects. No image reaches512candidatecap.

## Interpretation and limits

D reaches fitDead .996951/.995427 but audit .482759 in bothseeds. Additionalpositiveweight improves fit and transfersweakly/inconsistently under this budget; it does not resolve the precision/recall admission gap. This does not disprove all classawarelosses or establish classimbalance as causal. Fit/audit loss values cannot be treated as comparable qualitymetrics across weighted/unweightedobjectives. Both overallfitrecalls~.97-.98 versus audit~.58-.59 support investigating adaptation scope; they do not establish a mechanism. Crossround controls differ and must not be substituted for matched C.

Original1000draw tissue-stratified pairedimagebootstrap,RNG20261018,unmodified admissionF1=C/T1=D. Two seeds,repeated image-level fold1 reuse,smallDead support and lack of patient-separated audit limitgeneralization. HistoricalE19maskIoU or oldPQ scores are differentendpoints and are not subtracted from these pointmetrics.

## Verification and provenance

40smallfiles,2,965,093bytes returned to LM1;36payload SHA256/bytes match preservationmanifests. All configurations equal frozenP9 run_config,cohort/input/UNI identitysame across4runs;fit2134/audit522;eachloss historyexactlyepochs1-130,finite. OriginalcohortSHA55894f3274a1f65338ccd6cb885abc218056be64318405ad5e4d67b895ea5d22. Sourcecommit051082692dce2dd1f29a8b6383d9cf4a9dad1988,manifest85ef261efe90a495cbdbf16a6465369cbc076f8eebfe5c0c4585727e1d9edb32. Complete/lastcheckpointresults preserved onprojectand scratch. Canonicalevidence is ignored ops/review_20261023/p9_collected_01,p9_collection_manifest.json,p9_gate_01.json,p9_analysis_01.json,p9_live_01.json. No metric/HV edits or dependencyupdates. CurrentP10focused suite119passed13.17s;historical5dense_votes failures disclosed,not a fullsuitepassclaim.

## Next study

Separately frozen [P10](../superpowers/specs/2026-10-23-proposal-p10-design.md) compares originallast12encoderblock adaptation C with last4 S. Sameinitialmodel/head,data/loss/augmentation/optimizer/decode/gate,fourfresh130epochruns. Earlierfeaturetaps remain frozen inS,so this is an adaptation-scope intervention including featurepaths andtrainablecount,not a purecapacitymechanismtest. Noautomaticpromotion;realGPUvalidation and strictready precede formalarray.
