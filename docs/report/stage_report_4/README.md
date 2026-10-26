# Stage report IV

**Mechanism controls for 2× scale, the Dead specialist hypothesis and inference cost**

Experiments cover October 6–8, 2026. The Chinese report and digest were revised on October 10 to incorporate the existing October 9 corrections. No new experimental performance was added.

## Reading guide

- [English report](stage_report_4.pdf), 12 pages; historical version, subject to the corrections below · [TeX](stage_report_4.tex).
- [Chinese report](stage_report_4_zh.pdf), 11 pages · [TeX](stage_report_4_zh.tex).
- [Plain-language Chinese digest](summary_plain_zh.pdf), 3 pages · [TeX](summary_plain_zh.tex).
- Language revision and build verification (archived verification record): source/PDF hashes, fresh log checks, table comparisons and regression results.
- Original numerical verification snapshot (archived verification record): frozen October 8 values and historical cost aggregation.
- Historical build record (archived verification record): earlier English/Chinese report builds, not the revised Chinese PDF hashes/page counts.
- [October 9 research proposal](../../proposals/RESEARCH_NEXT_2026-10-09.md): historical proposal for fixed-geometry typing probes and independent candidate review. Consult [current results](../../results/README.md) for later outcomes.

The full report explains settings, rules and evidence limits; the digest introduces the work to new readers. Earlier report bodies were not rewritten in that revision, and the English PDF was not recompiled.

## Corrections and revision scope

The Chinese abstract, transitions and digest were rewritten to explain terminology and reduce awkward language mixing. All **7 numerical tables, figure paths, labels and bibliography keys match the earlier version**. Figures and experimental snapshots were preserved.

The [October 9 audit](../../audits/RESEARCH_AUDIT_2026-10-09.md) corrections incorporated into the Chinese report are:

1. **Weight cost by image count.** The old `mean_ms` gave tail chunks and full chunks equal weight. Recovering image-weighted means from saved total throughput in the same `cost.json`, then selecting the highest-throughput repetition again, gives **6.08447×** for two-model/x1 and **4.66316×** for TTA/x1. Historical tables retain the labelled old values 6.08687× / 4.66653×. This is an arithmetic correction, not a new GPU measurement; the approximate 6.1× conclusion is unchanged.
2. **The forward pass used bf16 autocast.** Historical fp32 labels are inaccurate. Chunk-level p95 is not single-request latency p95.
3. **State test-data access precisely.** New E2a, DSB and E0 mechanism experiments used train/validation folds. Audits checked historical test summaries; two test evaluation artifacts were also reconstructed from existing predictions on October 8. The entire stage cannot be described as having no test-data access.
4. **Limit the DSB inference.** Proxy losses at the final checkpoint did not exhibit the preregistered gradient conflict, so the recipe stopped. The specialist branch was not trained and its efficacy was not tested. This does not rule out conflict throughout training or all specialist designs.
5. **Retain reproduction gaps.** Verifiable main values do not mean every historical experiment was rerun. Complete P0–P3 provenance checking remains blocked by a conflict between the old seed manifest and later same-name directories; the check was not bypassed.

## Main conclusions

- **E2a:** Four configurations × two seeds on split1 validation are complete. Lowering the x1 HV threshold recovers 9.59% of the original x2 Dead gain, below the baseline-switch rule. This is a decision ratio, not a causal contribution estimate. After matching the threshold in original-image area units, the x2 bPQ cost remains and strict Dead decreases in all four seed-pair contrasts. Default `hv_min_size=30` is unchanged.
- **DSB:** Gates A, C and D permit continuation. After the invalid first Gate B measurement was corrected under the registered protocol, 64 valid batches had mean decoder composite-gradient cosine +0.749, minimum +0.594, and no negative values. This fails the continuation condition. The specialist recipe stopped without specialist training. The probe itself used a GPU; the stage did not have zero GPU activity.
- **E0:** The two-model system takes approximately 6.1× the x1 inference time. Shared-device load and timing boundaries limit absolute latency extrapolation; M/S and other stages were not all included. EF-P2 remains mechanism evidence and a teacher candidate, without a deployment advantage claim. Teacher effectiveness remains untested.

E2a uses one split and two seeds; all four Dead contrast confidence intervals cross zero. Gate B uses final-checkpoint proxy losses and batches that reuse images. This stage adds no new three-split test result.

## Numerical verification

The journal is not a substitute for original artifacts. The recorded revision reran:

- `verify_report3.py`: all 108 endpoint differences passed verification; P2 fails its joint rule and EF-P2 passes.
- `verify_report4.py`: output matches frozen `verified_numbers.json` exactly. E2a includes 48 run-level metrics, 24 configuration means and 24 contrast means: 96 values total. Gate statistics are recomputed from batch/row records.
- Cost correction: independently recover image-weighted means using `1000 / img_per_s`; see the [cost audit snapshot](../../results/inference_cost_audit_20261009.json).
- All 7 numerical tables match the earlier report item by item. Corrected values added to the Chinese text match the cost audit. Bootstrap intervals reuse frozen results without resampling.
- `python -m pytest -q`: **277 passed, 64 warnings** in that historical run. Warnings chiefly concern dependency deprecations and existing tensor conversions; dependencies were not upgraded to suppress them.

Historical `verify_report4.py` intentionally reproduces the old cost snapshot. Passing it does not validate the old weighting method; image-weighted results are reported separately. Figure CSVs were archived outside this delivery: `e2a_contrasts.csv`, `gate_b_cosines.csv`, `e0_cost.csv`.

## Reproduction and builds

Run from the repository root in the pinned project environment:

```bash
python scripts/report4_numbers.py
python scripts/verify_report4.py
python -m pytest -q
```

`make_report_figures4.py` can regenerate historical figures from frozen artifacts; the language revision did not regenerate figures. Python 3.10, torch 2.5.1+cu124 and numpy 1.23.5 remain pinned.

In this directory, use TeX Live with ctex, Fandol fonts and xeCJK; run each compiler twice:

```bash
xelatex -interaction=nonstopmode -halt-on-error stage_report_4_zh.tex
xelatex -interaction=nonstopmode -halt-on-error stage_report_4_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
```

The recorded Chinese PDF builds had no TeX errors, missing glyphs, undefined references or overflowing text boxes in fresh logs. Source hashes, extracted text and key values were checked, with spot checks of titles, digest pagination and the report appendix. No independent page-by-page visual review was claimed. `figs/` is self-contained; machine configuration and operational records remain in ignored `ops/`. The 2026-10-26 Markdown translation does not rebuild or alter the PDFs, TeX sources, figures or historical verification snapshots; historical test counts are not fresh results for this edit.

## Delivery layout (2026-10-26)

This directory contains the report PDFs, LaTeX sources and every referenced figure/input, plus build instructions. Verification JSON, figure CSVs, unused raster previews and montage source tiles were moved to the project owner's ignored operations archive; historical verification descriptions above refer to those archived records. The report PDFs and TeX content are unchanged by this cleanup. Research evidence and current outcomes are indexed in [the project results](../../results/README.md).

Build the English source with `pdflatex` and Chinese sources with `xelatex`, twice from this directory, unless the supplied build script specifies otherwise. Keep temporary compiler output in a separate build directory. No datasets or checkpoints are required to compile these reports.
