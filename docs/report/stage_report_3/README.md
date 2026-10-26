# Stage report III

**Cross-scale candidate selection and area filtering for nuclei instance segmentation**
Revised 2026-10-06; experiments cover the October 2–5 code audits, paired validation and EF-P2 round.

## Reading guide

- [English report](stage_report_3.pdf) · [TeX source](stage_report_3.tex)
- [Chinese report](stage_report_3_zh.pdf) · [TeX source](stage_report_3_zh.tex)
- [Plain-language Chinese digest](summary_plain_zh.pdf) · [TeX source](summary_plain_zh.tex)
- [Verified numerical snapshot](verified_numbers.json)
- [Historical build and test verification](build_verification.json): Chinese report 11 pages, English report 12 pages, digest 3 pages; includes file hashes.
- Figure data: [nine paired differences](figs/paired_deltas.csv), [split-level intervals](figs/split_intervals.csv).

The report explains methods, controls, corrections and statistical limits. The digest introduces the work to readers unfamiliar with the project. Earlier report bodies remain historical records. This report describes the October 2–5 milestone; consult the [current results index](../../results/README.md) for later work.

## October 6 review and revisions

- Distinguished class-agnostic candidate matching from the same-class matching used by mPQ and Dead PQ; added the PQ definition.
- Clarified that the Dead typing proportion uses **matched predictions labelled Dead** as its denominator. It is not classification recall over true Dead objects.
- Corrected rounded Dead PQ for x2+du2 in the historical seed19 scale table: 0.18312 → **0.18311**. The main table and decision are unchanged.
- Checked 18 model configurations, 108 paired differences and 18 split-level bootstrap intervals across two rounds. No retraining, inference or threshold search was performed.
- Revised the Chinese title, abstract, terminology, paragraphs and references; added primary sources for PanNuke, UNI, CellViT and PQ. Updated the corresponding English and digest explanations.
- Regenerated English and Chinese figures. The Chinese PDFs use valid embedded TrueType fonts, covered by regression checks. Figure data match the original CSV files.

## Conclusions and limits

Across nine genuine `(split, seed)` pairs, the three-split mean of seed means for EF-P2 versus matched-seed x1+M/S is:

- Dead PQ: 0.17255 → 0.17954, an absolute gain of **+0.00699**.
- mPQ: 0.49760 → 0.49828, an absolute gain of **+0.00068**.
- bPQ: 0.66530 → 0.66529, a change of approximately **−0.000009**.
- Dead improves in eight pairs; one identity pair is unchanged. The filter retains **94.3%** of the unfiltered P2 Dead gain.

EF-P2 passes the preregistered rule; this is not a statistical non-inferiority proof. Split-level mPQ/bPQ intervals cross zero, strict mPQ declines slightly, and test folds were reused across rounds. Full dual-scale inference cost had not been measured at this milestone; the later [stage IV cost correction](../stage_report_4/README.md) takes precedence for cost. Test folds are 3/3/1 across the three splits, so they are not three disjoint patient cohorts.

## Numerical verification

The journal is not used as the final numerical source. Main-table values were re-extracted from raw evaluation summaries:

1. Read the fair baseline from `matched_seed_20261004/p0/<pair>/test/base/ms/eval/summary.json`.
2. Read `matched_seed_20261004/p2/<pair>/test/eval/summary.json` and `existence_ef_20261005/p2ef/<pair>/test/eval/summary.json`.
3. Recalculate **108 differences**: nine pairs × six endpoints × two rounds. Compare with `stats/stats.json` at tolerance `1e-12`.
4. Independently recalculate split-level seed means, population standard deviations of paired differences (`ddof=0`), sign counts and preregistered decisions.
5. Check all five selection quantities in each of the nine EF `off` rows against the frozen P2 rows.
6. Recalculate area and border groups from `existence_audit_20261005_v2/records.npz` and compare with the v2 summary.

These paths are relative to `runs/analysis/`. Local run artifacts are not distributed with Git. `verified_numbers.json` is a portable numerical snapshot without machine configuration; it retains SHA-256 hashes of statistical files.

Historical seed19 P0–P3 controls come from `p0p3_20261002_v2/results_verified.json`; typing audits come from `typing_audit_20261003/`; split-level intervals come from each round's `stats/bootstrap_split*.json`. Intervals condition on the three trained seeds and resample images only; they do not represent all training uncertainty. This revision reran numerical verification, figure generation and tests, without training, inference or threshold search.

### Correction precedence

- Use **`existence_audit_20261005_v2`** for border statistics. The old all-zero border table is invalid.
- Earlier checks found non-border records/groups consistent with v2; the existing `scripts/report3_numbers.py` was not overwritten.
- The EF identity fallback bug was fixed. None of the nine existing pairs triggered it, so their scores are unchanged.
- Preserve historical selection source fingerprints. Re-evaluation must use the matching code snapshot or explicitly record a new review run in a new output directory. Do not disable provenance checks.

## Reproduction

Run from the repository root in the pinned project environment:

```bash
# Extract existing table values; print without changing experiment artifacts.
python scripts/report3_numbers.py

# Independently verify raw summaries and v2 records; emit JSON to stdout.
python scripts/verify_report3.py

# Generate English/Chinese figures and CSVs from frozen statistics.
python scripts/make_report_figures3.py

python -m pytest -q
```

Figures use DejaVu Sans for Latin/Greek text and the TrueType Droid Sans Fallback font for Chinese text. Use `--zh-font /path/to/chinese-font.ttf` for another font location. Choose a font with TrueType outlines; do not embed OpenType/CFF fonts as Type 42. The figure script accepts `--analysis-root` and `--out`. Plotting does not evaluate models or change selection rules; its CSV outputs were checked against the earlier version.

Historical verification on 2026-10-06: **157 passed, 63 warnings** (Python 3.10.21, torch 2.5.1+cu124, numpy 1.23.5; no dependency installation or upgrade). The added Chinese figure PDF font/text extraction regression failed before the fix and passed afterward. Other training, inference and evaluation code was unchanged. Passing tests do not establish experimental independence or remove provenance limits. This historical count is not a fresh test result for a later documentation edit.

## PDF builds

In this directory, use TeX Live with ctex, Fandol fonts and xeCJK. Run each compiler twice:

```bash
pdflatex -interaction=nonstopmode -halt-on-error stage_report_3.tex
pdflatex -interaction=nonstopmode -halt-on-error stage_report_3.tex
xelatex -interaction=nonstopmode -halt-on-error stage_report_3_zh.tex
xelatex -interaction=nonstopmode -halt-on-error stage_report_3_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
xelatex -interaction=nonstopmode -halt-on-error summary_plain_zh.tex
```

`figs/` is self-contained. Inspect fresh build logs for errors, missing figures/glyphs and overflow; old logs cannot verify a new build. Machine addresses, remote build commands and operational records belong only in ignored `ops/`. The 2026-10-26 Markdown translation does not rebuild or alter the report PDFs, TeX sources, figures or historical verification snapshots.
