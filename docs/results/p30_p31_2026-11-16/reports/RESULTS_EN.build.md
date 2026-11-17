
The eleven comparisons identify encoder adaptation as a productive part of the nucleus-center detector. Full-depth updates in P22 recovered 18/20 additional interior Dead nuclei; the equal-capacity first-six plus last-six configuration in P29 recovered 16/13. P30's middle-twelve configuration recovered 7/15, added 648/516 total matches and removed 215/254 unmatched predictions. These within-round improvements coexist with tissue-specific losses and substantial remaining rare-object misses. P31 now tests a different component: fixed group normalization in the point decoder, with the original last-twelve encoder adaptation. Its four-run Expanse array **55792463** was submitted on 16 November 2026 at 22:05 EST after successful hardware acceptance.

## Experimental scope and measurement

Each comparison uses fresh initialization from the same pretrained UNI encoder and seeds 19/1. Training uses 2,134 PanNuke fold-1 images; development evaluation uses the same held-out 522 images and 12,250 annotated nuclei. All reported final values use epoch 130. Physical batch four, effective batch sixteen, augmentation, object-normalized focal loss and point decoding remain fixed unless the intervention explicitly changes them. P20 changes input size; P25 changes encoder depth and learning rates together.

The detector emits a single class-agnostic center map. A one-to-one assignment matches decoded points to annotated centers within three native pixels. Overall recall divides matches by 12,250; precision divides matches by every decoded prediction. Unmatched predictions are the difference between prediction count and matched count. Ground-truth categories provide evaluation strata. Interior Dead denotes apoptotic/necrotic nuclei whose annotated outlines clear the image boundary: 145 objects on eleven images. The predefined small-Dead subset has 32 objects with native area below 67 pixels.

These are sequential development experiments on one reused fold. Each new design responds to accumulated development evidence. Within-round paired differences are the primary comparisons because separate rounds retrain their own controls. Seed repeats describe optimization variation on the same images. Image uncertainty uses the original tissue-stratified paired bootstrap, with 1,000 draws. Official three-split instance-segmentation evaluation follows the project's separate last-checkpoint protocol.

## The eleven comparisons


\begin{figure}[H]\centering
\input{tex_figures/stage_net_dead.tex}
\caption{Change in interior Dead matches relative to each round's own control}
\end{figure}



\begin{figure}[H]\centering
\input{tex_figures/stage_net_all.tex}
\caption{Change in all matched nuclei relative to each round's own control}
\end{figure}



\begin{figure}[H]\centering
\input{tex_figures/stage_net_extra.tex}
\caption{Change in unmatched predictions relative to each round's own control}
\end{figure}


| Intervention | Dead matches, seeds 19 / 1 | All matches | Unmatched predictions |
|---|---:|---:|---:|
| Larger input, P20 | +13 / +6 | +3 / −244 | −302 / −497 |
| Class-center auxiliary supervision, P21 | +1 / −5 | −168 / −148 | −180 / −81 |
| All 24 encoder blocks, P22 | +18 / +20 | +738 / +753 | −250 / −211 |
| Parameter averaging, P23 | +1 / +3 | −51 / +87 | −52 / +85 |
| Higher late-block learning rate, P24 | +11 / −2 | +29 / +38 | −618 / −509 |
| All-block layerwise learning rates, P25 | +7 / +10 | +741 / +512 | −510 / −544 |
| Pretrained-parameter constraint, P26 | +2 / +2 | −157 / −331 | +208 / −82 |
| First twelve blocks, P27 | +6 / +15 | +521 / +213 | −502 / −696 |
| Alternating twelve blocks, P28 | +2 / −1 | +528 / +273 | −80 / −391 |
| First six plus last six blocks, P29 | +16 / +13 | +283 / +165 | −65 / −64 |
| Middle twelve blocks, P30 | +7 / +15 | +648 / +516 | −215 / −254 |

Larger input in P20 improved target detection while increasing pure training cost approximately 4.3-fold. Its overall coverage changed little in seed 19 and decreased in seed 1. P21's auxiliary class-center targets reduced true matches and unmatched points together. Parameter averaging in P23 produced only small target changes and mixed overall effects. The higher late-block learning rate in P24 removed many unmatched points but reversed its Dead benefit across seeds. These experiments distinguish recovering missed nuclei from simply reducing the output count.

P22, P25 and the encoder-position comparisons improved overall coverage in both seeds. Their rare-object outcomes vary considerably. P26's pretrained-weight constraint adds two target matches per seed while losing 157/331 overall matches. P28 restores high training coverage after P27, yet its Dead change is +2/−1. Training coverage, parameter count and the number of trainable layers consequently provide incomplete explanations of development detection. The image-level and object-level evidence below shows which parts of those aggregate changes repeat.

## P22: full-depth adaptation and substantial rare-object recovery

P22 updates all 24 UNI blocks at the original encoder learning rate; its control updates the last twelve. It adds 18/20 Dead matches and 738/753 total matches, while reducing unmatched predictions by 250/211. Final Dead coverage is 85/145 and 82/145, or 58.62%/56.55%; precision is 72.84%/73.36%. This is the strongest two-seed rare-object gain among these eleven fixed comparisons. Pure training costs about 1.43/1.42 times the control, and training memory rises from approximately 4.05 to 6.75 GiB. The four completed scheduler tasks used 21.48 GPU hours.



\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p22_image_1037_seed_19.png}
\captionof{figure}{P22, lung image 1037, seed 19: nine additional target matches}
\raggedright

The lung image above changes from 8/20 to 17/20 interior Dead matches. Object 75 at row,column (202,29), native area 44, is a small nucleus in the dense lower-left tissue region. Its nearest prediction changes from 21.40 to 1.41 pixels. Object 67 at (98,180), area 65, changes 20.62→2.24 pixels. Both predefined small target nuclei in this image are recovered. Objects 66, 71, 76 and 79 also gain nearby predictions, covering a range of native sizes and spatial contexts. Objects 56 and 62 instead cross the three-pixel boundary, each from 3.16 to a distance within the radius. Thus the nine gains combine coverage of previously empty neighborhoods with improved positioning of existing nearby points.

\end{minipage}





\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p22_image_776_seed_19.png}
\captionof{figure}{P22, colon image 776, seed 19: recovery accompanied by two losses}
\raggedright

Colon image 776 changes 5/7→4/7. Object 9 at (205,201), area 191, gains a prediction as distance drops 38.42→1.41 pixels. Object 7 at (148,26) loses its nearby prediction, 2.83→30.02 pixels; object 12 in the lower strip at (238,110) changes 3.00→62.39. The colon example therefore supplies a concrete cost of the full-depth configuration alongside the pronounced lung recovery. Its fixed image was chosen by annotated strata, together with the other examples, before evaluating model effects.

\end{minipage}



Seed 1 independently recovers small object 75 and objects 66/71 in lung 1037, but its image count changes 10/20→15/20 through six gains and one loss. Colon object 7 is lost in both seeds. In seed 19, lung 1037 contributes nine net target matches and cervix 1756 contributes seven; their combined sixteen account for most of the total eighteen. This concentration makes the individual tissue images central to the interpretation of the aggregate result.

## P25: stronger broad detection with a lung–colon tradeoff

P25 combines all 24 trainable blocks with layerwise rates: block b starts at 3e-5 × 0.95^(23−b). It increases total matches by 741/512 and reduces unmatched predictions by 510/544. Precision reaches 74.44%/74.24%, close to the fixed 75% requirement. Dead gains are smaller than P22's: +7/+10, ending at 76/145 and 75/145. Lung contributes +9/+8, while colon contributes −7/−6. Colon image 1482, with 49 target objects, changes 15→9 and 14→10 matches. The combined depth-and-rate intervention favors some tissue contexts while losing target detections in the most concentrated colon image.



\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p25_image_1037_seed_19.png}
\captionof{figure}{P25, lung image 1037, seed 19: recovery of both small target objects}
\raggedright

Lung 1037 changes 10/20→15/20. Objects 67 (98,180), area 65, and 75 (202,29), area 44, gain predictions at distances 20.62→2.24 and 15.00→1.00 pixels. Object 66 changes 20.62→1.00. These are the same annotated neighborhoods improved by P22. At the same time, object 63 at (67,158) loses a point, 2.24→19.80 pixels, and object 78 at (222,125) changes 1.41→23.26. New coverage and lost coverage coexist within one image.

\end{minipage}





\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p25_image_776_seed_19.png}
\captionof{figure}{P25, colon image 776, seed 19: the same lost object reappears across configurations}
\raggedright

Colon 776 changes 4/7→3/7. Object 7 again loses a nearby prediction, 2.24→28.84 pixels, while object 9 is gained, 39.05→1.41. Object 8 at (179,54) changes 1.00→3.16 pixels, crossing the matching boundary. Seed 1 loses object 7 as well and additionally loses object 11 in the lower-left strip. This repeated object-level pattern connects the lung–colon tradeoff to actual annotations rather than a tissue label alone.

\end{minipage}



P25's training Dead coverage reaches 649/656 in both seeds, whereas development small-Dead coverage is 10/32 and 12/32. Pure training takes approximately 1.41/1.42 times the control and peaks at 6.75 GiB. The four completed tasks use 19.99 GPU hours. Stronger training coverage and higher overall precision therefore leave a specific transfer problem around rare target nuclei.

## P27–P29: update position at fixed capacity

P27 updates the first twelve blocks, P28 alternates twelve blocks, and P29 updates the first six plus last six. All retain the original trainable encoder count of 151,179,264 parameters and head count of 3,294,657. Early trainable blocks receive gradients through the frozen suffix. Their same-round controls update blocks 12–23. Each configuration changes the trainable path through the representation hierarchy.

P27 adds 6/15 Dead matches and removes 502/696 unmatched predictions, despite training recall falling to 88.80%/87.82%. P28 raises training recall to approximately 96.3%, adds 528/273 total matches, and produces only +2/−1 target matches. P29 adds 16/13 targets, with positive original paired-bootstrap lower bounds in both seeds. Its total-match gain, +283/+165, is smaller than P27 or P30. These fixed comparisons show separate responses in broad coverage, precision and rare-object recovery.



\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p27_image_1037_seed_19.png}
\captionof{figure}{P27, lung image 1037, seed 19: five gains and three losses}
\raggedright

The first-twelve configuration changes lung 1037 from 12/20 to 14/20. Objects 71 (170,23), 75 (202,29) and 76 (211,14) recover predictions at 21.02→1.00, 20.81→1.41 and 12.00→1.00 pixels. All three are gained in seed 1 as well. Object 56 near the top of the image is lost in both seeds: seed 19 changes 2.24→21.00. Object 73 at (187,68) also loses its match in both seeds, with distance 1.41→3.61. The shared lower-left recoveries appear alongside a repeated loss at the matching boundary.

\end{minipage}





\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p29_image_776_seed_19.png}
\captionof{figure}{P29, colon image 776, seed 19: recovery in the lower tissue strip}
\raggedright

P29 changes colon 776 from 3/7 to 6/7. Objects 10 (231,172), 11 (230,19) and 12 (238,110) gain nearby points, with distances 41.23→1.00, 28.02→1.00 and 80.66→1.00 pixels. Object 12 is recovered in seed 1 too, 61.52→2.00; its native area is 104, above the predefined small-object bin. This same object was lost in P22 seed 19, illustrating how update position can reverse the detection outcome around an identical annotation.

\end{minipage}





\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p29_image_1037_seed_19.png}
\captionof{figure}{P29, lung image 1037, seed 19: a small-object gain and a positioning loss}
\raggedright

The split-depth configuration changes lung 1037 from 12/20 to 13/20. Small object 75 gains a match, 15.00→2.83 pixels. Object 62 changes 3.16→1.00. Object 56 changes 2.83→3.61 and loses its match in both seeds. The recovered small nucleus therefore repeats across P22, P25, P27 and P29 seed 19, while the nearby localization losses vary with the update recipe. P29's seed-19 pure training cost rises by 21.24% and memory from approximately 4.05 to 4.84 GiB; seed-1 timing includes long recorded Lustre delays. The preserved timing records separate scientific outcomes from scheduler and storage costs.

\end{minipage}



## P30: middle-twelve adaptation

P30 updates blocks 6–17 instead of 12–23, retaining the same encoder and head parameter counts, rates and training schedule. All four tasks in array 55746187 completed successfully, each with 130 consecutive epochs and final development evaluation. Their combined scheduler duration is 17.4389 GPU hours.


\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}\hline
Run & Interior Dead & Small Dead & All matches & Predictions & Recall & Precision \\\hline
C19 & 71/145 & 9/32 & 7,246 & 10,539 & 59.151\% & 68.754\% \\
M19 & 78/145 & 14/32 & 7,894 & 10,972 & 64.441\% & 71.947\% \\
C1 & 64/145 & 8/32 & 7,338 & 10,647 & 59.902\% & 68.921\% \\
M1 & 79/145 & 13/32 & 7,854 & 10,909 & 64.114\% & 71.996\% \\
\hline\end{tabular}
\caption{P30 final development results: C updates the last twelve blocks; M updates the middle twelve. Suffixes denote training seeds.}
\end{table}

The intervention emits 433/262 more predictions but finds 648/516 more nuclei, reducing unmatched predictions by 215/254. Overall recall improves by 5.2898/4.2122 percentage points and precision by 3.1926/3.0748 points. Small-Dead coverage gains five objects in each seed. Interior Dead recall improves by 4.8276/10.3448 points; its original 95% intervals are [−0.9091,+9.6552] and [+1.8987,+18.3673] points.

The original decision requires all clauses in both seeds: adequate support, final Dead recall at least .60, a gain at least .05, a strictly positive paired-bootstrap lower bound, precision at least .75, relative quality and nondegeneracy. Both P30 seeds satisfy relative-quality and support checks. Seed 19's gain is .048276 and its lower bound crosses zero; final target recalls of 53.79%/54.48% and precisions of 71.95%/72.00% remain below the absolute requirements. The independently recomputed full decision is false in both seeds, closing this fixed middle-twelve recipe under its original rule.

## P30 tissue distribution and training cost


\begin{figure}[H]\centering
\input{tex_figures/p30_dead_by_image_seed_19.tex}
\caption{P30, every target-positive development image, seed 19}
\end{figure}



\begin{figure}[H]\centering
\input{tex_figures/p30_dead_by_image_seed_1.tex}
\caption{P30, every target-positive development image, seed 1}
\end{figure}


Lung contributes +5/+9 and colon +3/+6; cervix loses one match in each seed. Lung 1037 gains four/six, lung 1038 gains two/four and colon 1482 gains three/five. Lung 1025 loses one in each seed. Colon 1482 ends at 17/49 and 16/49 target matches, leaving 32/33 target objects uncovered. Its positive change reduces a concentrated miss burden while retaining substantial remaining work.

Seed 19 has three positive, two negative and six unchanged target-image counts; seed 1 has five positive, two negative and four unchanged. Removing one positive image at a time as a descriptive sensitivity calculation yields gain ranges of 2.40–6.67 and 7.20–13.33 percentage points. Both signs survive any single-image omission, while the original bootstrap reflects resampling uncertainty over the full fixed image set.


\begin{figure}[H]\centering
\input{tex_figures/p30_training_seed_19.tex}
\caption{P30, seed 19: complete loss trajectory and final coverage}
\end{figure}



\begin{figure}[H]\centering
\input{tex_figures/p30_training_seed_1.tex}
\caption{P30, seed 1: complete loss trajectory and final coverage}
\end{figure}


Training all-nucleus recall changes 97.85%→97.10% and 98.05%→97.12%. Pure training lasts 14,457.50 versus 15,957.59 seconds in seed 19, and 14,509.86 versus 15,957.90 in seed 1: increases of approximately 10.4% and 10.0%. Peak allocated training memory rises from 4.05 to 4.44 GiB. Middle-depth adaptation thus delivers broad development gains at a moderate measured cost without increasing trainable capacity. The changed path through frozen blocks affects computation even when parameter counts match.

## P30 fixed objects: recovery, loss and seed variation

The eight predetermined development examples contain 53 interior Dead annotations. Seed 19 retains 29, gains eight, loses five and misses eleven in both arms; seed 1 retains 26, gains eleven, loses five and misses eleven in both. Their net gains are +3/+6. Green rings mark intervention-only matches, orange rings control-only matches, white shared matches and pink shared misses; cyan crosses show decoded predictions. All object indices and row,column positions accompany the report in CSV and coordinate JSON files.



\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p30_image_1037_seed_19.png}
\captionof{figure}{P30, lung image 1037, seed 19: six gains and two losses}
\raggedright

Lung 1037 changes 11/20→15/20. Object 75 at (202,29), area 44, gains a point, 13.60→1.41 pixels, in the same lower-left neighborhood recovered by several earlier configurations. Object 57 at (43,37), a pale rounded nucleus beside darker neighboring tissue, changes 11.66→1.00. Objects 62, 66, 71 and 76 also gain matches. Object 56 (16,74) loses its nearby prediction, 2.24→22.02; object 58 (50,166) changes 2.24→3.61. The latter is a boundary crossing, whereas the former leaves a much larger gap to the nearest point.

\end{minipage}





\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p30_image_1025_seed_1.png}
\captionof{figure}{P30, lung image 1025, seed 1: a recovered small nucleus alongside four losses}
\raggedright

Lung 1025 changes 17/25→16/25 through three gains and four losses. Object 14 at (43,212), area 31, is recovered at 41.98→1.00 pixels. Object 15 (54,13) changes 13.45→2.83 and object 32 (217,245) crosses 3.16→1.00. Objects 20 (98,163), 27 (165,32) and 35 (246,77) lose nearby outputs, leaving distances 15.62, 21.21 and 15.65 pixels. Object 25 (129,55) changes 2.00→3.16 and narrowly crosses the matching radius. In seed 19, the same small object 14 reverses: 2.00→41.98 pixels. This annotation gives a concrete example of optimization-seed sensitivity within a tissue image whose net loss repeats.

\end{minipage}





\noindent\begin{minipage}{\linewidth}\centering
\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../figures/p30_image_776_seed_19.png}
\captionof{figure}{P30, colon image 776, seed 19: unchanged count with exchanged objects}
\raggedright

Colon 776 remains at 4/7 while object 8 (179,54) gains, 3.16→2.24, and object 9 (205,201) loses, 3.00→3.16. Both changes lie near the matching boundary. Seed 1 gains object 9 and improves to 5/7. The unchanged seed-19 count therefore conceals an exchange between annotations. The preserved coordinates measure decoded-point presence and position; probability maps and intermediate feature statistics would allow a direct test of score strength or peak selection in future mechanism work.

\end{minipage}



## P31: fixed normalization in the point decoder

The position comparisons have now covered early, late, middle, alternating and split-depth updates. Their repeated recoveries show useful representational flexibility; their differing tissue losses leave the decoder's handling of feature scale as a separate research question. P31 retains last-twelve encoder updates and adds eight affine GroupNorm modules to the point decoder. Group normalization computes feature statistics within each sample, which suits the fixed physical batch of four. Its groups and placement are chosen before the run.

Four normalization modules follow the four 1024→192 feature projections. One follows the 768→256 stem convolution before ReLU. Three follow the 256→192→128→64 upsampling convolutions before ReLU. All use 32 groups and epsilon 1e-5; the channel counts sum to 1,408, adding exactly 2,816 scale/bias parameters. Convolution tensors remain shared at initialization. Initialization preserves the original random-number state and the initial constant center probability of .01.

The scientific question is whether this fixed normalization improves optimization and center coverage under unchanged encoder adaptation, loss, rates, decoding and evaluation. It changes both the normalization transform and its affine parameters. P31 compares original C with normalized G, seeds 19/1, four fresh 130-epoch runs. The last-checkpoint rule and original two-seed decision remain frozen.

Hardware acceptance 55792375 completed in 369 seconds on a V100. The node ran 131 focused checks in 73.96 seconds. Both seeds passed shared initialization and RNG checks, original-control two-step equivalence, real data-gradient updates at all eight normalization modules and blocks 12/23, unchanged frozen tensors, exact optimizer membership, BF16/FP32 behavior and checkpoint reloads. Independent normalization-formula errors were at most 1.43e-6 over 64 production calls per seed. Conservative C/G runtime estimates are 7.0457/7.0848 hours; memory peaks are 4.0943/4.1193 GiB. Four one-epoch training previews are preserved with their preliminary scope.

Formal array **55792463**, submitted at **2026-11-17 03:05:58 UTC**, maps tasks 0–3 to C19/G19/C1/G1, concurrency two. Each requests one V100, four CPUs, 32 GiB and twelve hours; total formal envelope 48 GPU hours. Acceptance plus the formal envelope is 48.1025 GPU hours within the 48.5-hour combined budget. Fresh source hashes, quota, filesystem and OST headroom, durable write/readback, allocation and scheduler admission passed before submission. All four tasks were pending at the submission check. The actual receipt is retained in [P31 submission evidence](../p31_submission_receipt.json).

## Results, reproducibility and follow-up

P30 collection verifies 117 small artifacts, including 48 final PNGs, with source and file hashes; its inherited deployment contains 514 verified source files. This report adds thirteen P30 analysis images and eight reconstructed large P22/P25 overlays, with the original matching coordinates and pixel provenance. Eight P30 checkpoint sizes and their original preservation records are retained; independent large-checkpoint rehashing and external archival are the next preservation step. P31's isolated deployment seals 520 files to scientific commit 26ce1f26da5326364f39f3d61c225d615876ad37.

The English report, Chinese report and Chinese reader are compiled from LaTeX. Statistical plots use PGFPlots; tissue overlays use real preserved image pixels and predictions. Source, build commands, rendered-page quality checks, raw results, object CSVs and scheduler receipts accompany the PDFs. Completion of P31 requires checking all four 130-epoch histories, final-checkpoint identity, preservation hashes and the unchanged full decision before interpreting its development outcomes. The experiment has been submitted; its final scientific results will follow those checks.