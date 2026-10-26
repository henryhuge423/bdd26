# Research documentation

Start with the [October 26 documentation and evidence audit](audits/DOCUMENTATION_RESULTS_GIT_AUDIT_2026-10-26.md),
the [latest experiment results](results/README.md), then the
[research protocol and dated progress](RESEARCH_PLAN.md) and [experiment journal](findings.md).
The [October 14 closeout checklist](audits/RESEARCH_STATUS_2026-10-14.md) is a historical milestone. Dates in older documents describe their own historical
state; a proposal or diagnostic gate is not a training result or continuing authorization.
The current experiment's approved spec and plan govern execution.

| Location | Purpose |
|---|---|
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | Canonical scientific protocol, research direction and dated progress |
| [findings.md](findings.md) | Dated experiment journal, numbers, decisions and artifact provenance |
| [results/](results/README.md) | Detailed experiment results and portable verification snapshots |
| [audits/](audits/README.md) | Evidence checks, corrections, research status and closeout tasks |
| [proposals/](proposals/README.md) | Dated research proposals; historical hypotheses and execution order |
| [background/](background/README.md) | Model explanations and interpretation of research evidence |
| [literature/](literature/README.md) | Literature survey and source material |
| [reference/](reference/README.md) | Course brief and stable external requirements |
| [report/](report/README.md) | Stage report bundles, sources, figures and numerical verification |
| [superpowers/specs/](superpowers/specs/) | Experiment designs, approval records and frozen decision rules |
| [superpowers/plans/](superpowers/plans/) | Implementation/execution plans and outcomes |

## Writing and maintenance conventions

- Put new dated results, audits and proposals in their corresponding folder. Keep existing
  filename dates and append dated corrections or a superseding-result notice; preserve the
  original evidence and explain which version is valid.
- Use English for the main README, research plan, findings journal, navigation pages and new scientific results.
  Explicitly labelled Chinese reports/digests and historical source documents may remain Chinese.
  Preserve experiment identifiers, hashes, timestamps, signs, units and links when editing prose.
  Flag corrupted text for source verification; do not guess missing scientific content.
- Keep the journal chronological and the navigation concise. Update current status in the
  status document and project README when an experiment changes state; do not rewrite an
  old dated proposal as though it already knew a later outcome.
- Freeze splits, seeds, menus, checkpoint-selection rules and gates in approved specs/plans
  before execution. Separate proposals, approvals, measured results and interpretations.
- Tie reported numbers to artifact paths, code/config hashes, split/seed, evaluation protocol,
  uncertainty and limitations. Preserve verification JSON/CSV alongside report sources.
  Machine-local run artifacts remain under ignored `runs/`; report bundles carry portable
  evidence needed to understand the report.
- Keep each stage report self-contained: English and Chinese reports plus a plain-language
  Chinese digest, TeX/PDF, figures, source links and reproduction/verification notes. A source
  edit does not imply that an existing PDF or build-verification snapshot was rebuilt.
- Use document-relative Markdown links; use repository-root paths for commands and artifact
  citations. Check incoming/outgoing links whenever moving a file. Keep report folders and
  frozen workflow filenames stable.
- Put machine routing, resource inventories, job PIDs/logs, SSH/sync notes and credentials in
  ignored `ops/` or `CLAUDE.local.md`; portable documentation must not embed secrets.

Repository entry points: [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md).
