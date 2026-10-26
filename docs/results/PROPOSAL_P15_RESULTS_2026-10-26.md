# P15 fixed training translation: final closeout, 2026-10-26

P15 fails the original admission in BOTH seeds. Close the fixed training translation recipe; do not add a translation-range/seed/epoch menu or start Phase B/masks. The result is point-proposal development evidence on repeatedly reused fold1 images, not PQ, test or independent generalization performance.

Array55278753, tasks0=C19/1=J19/2=C1/3=J1, all COMPLETED/0:0. Each completed130epochs and last-checkpoint fit/audit evaluation. Elapsed4:10:58/4:12:13/4:17:28/4:12:26 on exp-3-58/exp-2-58/exp-8-57/exp-4-60. Original P15 training commit is cf9777be1914884f48d2d8cb142a9b09d7408fc6; manifest5654147fa8ba972d012e0bee07c1ff3dfc6d1013c4dfcefe17f27dfcd06995a5. Fresh C/J use original architecture,last12/encoderLR1e-5/head3e-4, original full image permutation,130epochs,BF16 and original decoding. J alone translates RGB/centers by dx/dy[-16,16] with white fill, removes out-of-frame centers, rebuilds original Gaussian and uses retained-object denominator before original augmentation. Fit/audit evaluation is unaugmented.

|Seed|Arm|Fit all recall|Fit internal Dead recall|Audit all recall|Audit unique precision|Audit internal Dead recall|
|---|---|---:|---:|---:|---:|---:|
|19|C|.977790|.952744|.591102|.691661|.489655|
|19|J|.470315|.445122|.388408|.865406|.296552|
|1|C|.977613|.951220|.594286|.698120|.400000|
|1|J|.470589|.451220|.380163|.862887|.310345|

Original F1=C/T1=J admission recomputed without modification. J−C internal Dead −.193103 with original tissue-image bootstrap95%CI[−.292453,−.124138] for seed19; −.089655[−.200000,−.021739] for seed1. Support11images145objects and nondeg1 pass; precision rises above.75, but absolute Dead recall<.60, improvement<+.05 and CI lower>0 fail, and overall recall changes−.202694/−.214122 violate C−.02. Both conjunctions fail; high precision cannot override recall clauses.

The detector emits10469→5498 and10428→5397 audit candidates. Unique matches fall7241→4758 and7280→4657, while unmatched candidates fall3228→740 and3148→740. All-object GT count12250; no image reaches512candidate cap. Small-object audit support is only32objects, hits7→6 both seeds, so it does not support a separate small-object claim. Higher precision here accompanies fewer proposals and fewer matched objects; this is not a quality gain under the original rule.

The failure extends to unaugmented fit images: all recall falls from~.978 to~.470 and internal Dead~.952 to~.45. Last training focal loss is.05557/.05538 for C versus.82139/.81409 for J, although augmented targets and exposure differ and losses should not be interpreted as a matched causal estimand. Pure GPU-training time is4.0243/4.0829/4.0367/4.0146hours; each training peak~4.0474GiB. The observed profile supports difficulty learning this recipe in the fixed budget; it does not isolate absolute-position sensitivity, white margins, partial nuclei, dropped targets or regularization as the cause.

Independent deterministic replay of all520per-epoch translation records matches seed/epoch/original image ID/namespace150025,2134image exposures,2134unique members, offset histograms and exact retained/dropped object totals. Original fit contains50968objects; J drops on average9.9377%/9.9406% per epoch (ranges9.4059–10.4026%/9.5629–10.4732%). Targets for centers outside frame are removed while partial appearance can remain, and white fill changes context. These protocol features limit a mechanistic interpretation; do not claim that all spatial augmentation fails.

Descriptive tissue breakdown of internal Dead counts: Lung60,Colon56,Cervix15,Esophagus8,Stomach4,Breast1,Uterus1. Lung recall C→J .650→.383/.600→.400; Colon .304→.179/.214→.214. Esophagus improves .500→.625/.375→.500, while remaining tissues have tiny counts and mixed results. Lung+Colon hold116/145objects (80%); this is a descriptive concentration, not a new selection rule or evidence of tissue transfer.

Verification:45smallfiles,40preservationpayloads pass exact byte length and SHA256; all four manifests/configs/UNI/input/cohort identities match frozen expectations;130ordered finite loss records each; exact audit/fit members; original summary and admission recomputed. All395actual Expanse snapshot files independently match the sealed manifest. Nine inherited historical documentation files differ from current LM1 committed text, while scientific scripts/src/tests match exactly; missing third_party files in bare worktree are inherited in deployment. The sealed deployed manifest is the byte identity, rather than a claim every ancillary file equals Git. P16 refreshes those historical docs only in its own new snapshot, preserving P15.

Limits: checkpoint hashes in the writer preservation manifest were not independently rehashed in this closeout. No new inference or fold2/fold3 model evaluation. The11Dead-positive audit images provide weak conditional uncertainty; image-level split is not patient/slide separation, seed bootstrap is not population uncertainty, and sequential hypothesis selection repeatedly reuses audit522. No biological or causal mechanism conclusion, no official three-split result. Lustre scratch/project are replicas, not an external archive; external archival remains outstanding. Earlier negative recipes and unavailable human expert review remain unchanged.

Evidence: ignored LM1 ops/review_20261026/p15_collected_01, p15_analysis.json,p15_additional_verification.json,p15_document_identity.json; original source/control protocol in [P15 design](../superpowers/specs/2026-10-25-proposal-p15-design.md). New [P16 design](../superpowers/specs/2026-10-26-proposal-p16-design.md) tests direct Gaussian squared-error supervision as a separate objective hypothesis; it does not reopen translation.

![P15 verified point-metric decomposition](figures/PROPOSAL_P15_REVIEW_2026-10-26.png)
