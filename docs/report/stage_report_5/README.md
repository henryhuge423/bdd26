# Stage report V

**Learnability of the typing gap and evidence about candidate label quality**

Experiments cover October 9–12, 2026: the typing-oracle diagnostic; T0–T3 probes that fail preregistered admission; the E1 no-expert amendment (E1-C detector consensus and E1-M model review); E1-T typing consensus; and E3 candidate-quality development, closed by its stop rule.

## Reading guide

- [English report](stage_report_5.pdf) · [TeX](stage_report_5.tex).
- [Chinese report](stage_report_5_zh.pdf) · [TeX](stage_report_5_zh.tex).
- [Plain-language Chinese digest](summary_plain_zh.pdf) for readers new to the project · [TeX](summary_plain_zh.tex).
- Independent numerical verification (archived verification record): `verify_report5.py` recomputation of oracle/evaluator agreement, admission booleans, E1-C histogram/matching joins, E1-M agreement/kappa, E1-T group counts and interval containment, and complete E3 selection replay.
- Table-value snapshot: `runs/analysis/report5_numbers.txt` (local, not distributed with Git).
- Related documents: [research plan](../../RESEARCH_PLAN.md), [October 9 proposal](../../proposals/RESEARCH_NEXT_2026-10-09.md), [model structure guide](../../background/STRUCTURE_AND_STRATEGY_2026-10-09.md), and [preregistered specifications](../../superpowers/specs/) for T / E1 / E1-T / E3.

## Main conclusions

- **Oracle (October 9, read-only audit):** With instance geometry fixed, replacing types of IoU>0.5 matched objects by GT types raises three-split mPQ .4995 → **.6632**. Restricting correction to objects whose GT class is Inf/Conn gives **.6141**, about seven tenths of the total gain. Selection is by GT class, not an additive confusion decomposition. This diagnostic reads historical test predictions and motivates hypotheses only.
- **T0–T3 (October 10, preregistered):** Admission fails. The best T2 gain is +.0002 with a CI crossing zero; T3 changes −.0017. The unweighted linear head lowers Dead PQ by −.0053, beyond the guard. GT-region accuracy 85.3% is close to T2 matched typing 85.8% and majority-vote typing 85.6%; the simple linear readout adds no useful information over the TP head. Frozen-feature post-hoc typing closes alongside the earlier CONCH negatives.
- **E1 amendment (October 11):** Human expert resources are unavailable. Human review closes unexecuted, with the packet and obligation retained. **E1-C:** a 13-seat cross-fold jury gives synthetic null candidates ≥3 support at 0.75%, whereas 66–73% of zero-GT-overlap candidates have ≥3 support across folds/architectures (380/751 versus 518/1146). This is detector-consensus evidence motivating label review, not a biological judgment. Matching increases with support: 5–12% at 0–2 votes and 40–60% at ≥5 votes. **E1-M:** two model-review rounds yield kappa −.09 to .35; model review is unusable and establishes process validation only.
- **E1-T (October 12, preregistered):** Among 6,858 matched objects mistyped by the base model, the 12-seat jury's majority agrees with the base type in **72.5%** [.706,.742], versus the GT type in 18.2% [.167,.197]; intervals are separated. The reversal rate among objects correctly typed by the base model is 1.9%. The main Inf↔Conn groups show 74.8%/78.5%; GT=Epi/base=Neo is an exception favouring GT. Dead is reported separately. The preregistered interpretation supports detector-consensus evidence of an Inf/Conn label-boundary ambiguity; it does not establish biological truth.
- **E3 (October 12, preregistered):** Candidate matching is predictable (CV AUC .721/.737), but the EF strict-Dead guard rejects all 16 candidate configurations: maximum .1245 < .12957, versus A1 .1345. All three arms fall back to identity, adding no candidates, and both gates fail. Some official-metric improvements accompany lower strict Dead (a4_p0.5: −.0130). The line closes under its stop rule. A reopened proposal would need image-level class-absence-risk features.

Data boundaries: T/E1-C/E1-M/E1-T/E3 do not read test folds. They fit linear classifiers/logistic regressions without new segmentation-model training. E1-C includes validation-fold inference by two HoVer-Net jury seats. The oracle alone accesses historical test artifacts read-only. This stage adds no new three-split test performance.

## Numerical verification

- `python scripts/report5_numbers.py`: extracts tables from five frozen artifact sets, including joined zero-overlap strata whose counts are only aggregated in `consensus.json`.
- `python scripts/verify_report5.py`: independently recomputes from the lowest-level artifacts. Oracle values match the official baseline evaluation exactly; T admission booleans are recomputed from stored quantities; T0 majority-vote accuracy 85.6% is recalculated using official pairing because `results.json` stores only T1/T2. E1-C histograms/matching use `per_candidate.npz` joined to frozen candidate records. E1-M agreement/kappa use both CSV rounds. E1-T group totals, interval containment and input SHA are checked. E3 selection is replayed on every row against the frozen choice; discrepancies cause failure.
- `python scripts/make_report_figures5.py`: produces three English/Chinese figures (E1-C support, E1-T label agreement, E3 strict metrics) and frozen CSVs.
- Historical `python -m pytest -q`: **335 passed**, a net increase of 58 tests, rerun locally on 2026-10-13.
- Final E1-T/E3 artifacts were rerun after independent review and match the initial artifacts exactly, except that E3 bootstrap mPQ was corrected to official NaN exclusion. All 56 document/artifact audit checks passed.
- Bootstrap intervals reuse frozen results without new resampling.

Compiled figure PDFs remain in `figs/`; the following CSVs were archived: `e1c_support.csv`, `e1t_pairs.csv`, `e3_strict.csv`.

## Reproduction and builds

Run from the repository root in the pinned environment (Python 3.10, torch 2.5.1+cu124, numpy 1.23.5):

```bash
python scripts/report5_numbers.py
python scripts/verify_report5.py
python scripts/make_report_figures5.py
python -m pytest -q
```

`figs/` is self-contained. PDFs use TeX Live with ctex/Fandol/xeCJK: two `pdflatex` passes for English and two `xelatex` passes for Chinese. Historical builds are recorded in `build_verification.json`; machine details remain in ignored `ops/`. The 2026-10-26 Markdown translation does not rebuild or alter the PDFs, TeX sources, figures or verification snapshots. Historical test counts are not fresh results for this edit. See the [current results index](../../results/README.md) for later experiments.

## Delivery layout (2026-10-26)

This directory contains the report PDFs, LaTeX sources and every referenced figure/input, plus build instructions. Verification JSON, figure CSVs, unused raster previews and montage source tiles were moved to the project owner's ignored operations archive; historical verification descriptions above refer to those archived records. The report PDFs and TeX content are unchanged by this cleanup. Research evidence and current outcomes are indexed in [the project results](../../results/README.md).

Build the English source with `pdflatex` and Chinese sources with `xelatex`, twice from this directory, unless the supplied build script specifies otherwise. Keep temporary compiler output in a separate build directory. No datasets or checkpoints are required to compile these reports.
