# Documentation, experimental evidence and Git audit — 2026-10-26

## Outcome and scope

The main documentation is being reconciled against the latest P16 source snapshot,
`527a0893eac23f98f0b6c9a6b7a173a01237290d`. The primary checkout's master is an older,
published P12 snapshot; this explains why opening different worktrees gives different
status summaries. Three independent Luna agents checked language, experimental evidence
and Git state. Their work was reviewed and integrated by the main agent.

This audit inventories all **153 tracked Markdown files in the P16 snapshot**. The older
master contains 136. Inventory, encoding and relative-link checks cover the whole
Markdown set; detailed result reconciliation covers P13–P16 and their current summaries.
This is not a new reproduction of every historical experiment or a full translation of
every archived design, proposal and literature document.

The main README, research plan, findings journal, result navigation, stage-report III–V
navigation and current P13–P16 result/submission prose use English. Explicitly labelled
Chinese reports and digests retain their language. Older technical documents may retain
Chinese text; the post-edit inventory records that remaining scope. Original Markdown
bytes and SHA-256 hashes were backed up in ignored operational storage before editing.
No experimental source, frozen rule, raw result, checkpoint, PDF, TeX source or figure
was altered by this documentation revision.

## Result reconciliation

| Round | Treatment minus matched control: internal-Dead recall, seed 19 / seed 1 | Original gate | Evidence |
|---|---|---|---|
| P13 | −.137931 / −.048276 | Both fail | [Final result](../results/PROPOSAL_P13_RESULTS_2026-10-25.md) |
| P14 | −.068966 / −.089655 | Both fail | [Final result](../results/PROPOSAL_P14_RESULTS_2026-10-25.md) |
| P15 | −.193103 / −.089655 | Both fail | [Final result](../results/PROPOSAL_P15_RESULTS_2026-10-26.md) |
| P16 | No final scientific result verified in this audit | Submission only | [Submission record](../results/PROPOSAL_P16_SUBMISSION_2026-10-26.md) |

P13/P14/P15 values and gate booleans were compared with their returned analysis JSON,
not inferred from the prose. Their collection manifests record 45/40, 41/36 and 45/40
collected files/payloads, respectively. This comparison checks stored values and rounded
reporting; it does not independently regenerate every bootstrap interval from predictions.
P15 additional verification supports replay of 520 translation records; the source audit
records 395 deployed-file hashes. Large model checkpoints were not independently rehashed
in this documentation audit, and external archival remains outstanding.

The [P3–P15 synthesis](../results/PROPOSAL_PROGRESS_2026-10-26.md) reports that no tested
recipe passed its original two-seed admission. Higher precision in P14/P15 does not
override lower recall or a failed gate. These experiments repeatedly reuse a fold1
development point-detection audit: 522 images, 11 internal-Dead-positive images and
145 internal-Dead objects. They are not instance PQ, held-out test, independent-patient
validation or official three-split performance. The chronology is not an unbiased
hyperparameter ranking or evidence against every member of a method family.

P16 array **55287551** has a single submission receipt at
`2026-10-26T05:29:54.529735Z`. The stored scheduler snapshot at
`2026-10-26T05:30:48.240388+00:00` shows four pending tasks. This audit did not query the
current scheduler or collect later P16 results; pending is a historical observation,
and submission is not training completion. No new training, inference or evaluation
was launched during this documentation task.

## Language, encoding and navigation repairs

The language pass replaces concatenated English phrases with readable prose, translates
the Chinese portions of the main scientific plan/journal and labels old execution states
as historical. Scientific names, complete hashes, timestamps, code identifiers, numerical
values and link targets are checked separately from prose. A raw digit-count match alone
is insufficient: splitting a hash or scientific notation would still damage provenance.

The E6 journal contained corrupted multiplication/minus/arrow text and an obscured
paired-recall endpoint. The [E6 result](../results/SET_MASK_E6_DEV_2026-10-16.md) supplies
the authoritative Q−B endpoint, latency ratios, gamma .85→.99 and decoded recall
**18.73%→36.86%** versus terminal-learning-rate control **20.24%**. The restoration uses
that existing table; it is not a new measurement.

The P7/P8 result title contained non-UTF-8 bytes `A1 AA`; the rest of its body was valid
UTF-8. The repaired title explicitly marks the undecodable fragment instead of guessing
its original punctuation. Its numerical body is preserved.

The initial repository-relative link check found 19 missing targets. One journal link
omitted `results/`. The other 18 links pointed to local-only result/log files or an ignored
runtime amendment. These references are now labelled local artifacts, retaining filenames
and provenance context without implying that Git distributes them. Navigation now leads
to the October 26 synthesis and audit, rather than presenting October 14/18 as current.
The historical October 15 synthesis also now reflects the completed C2 mechanism failure,
superseding its old “unexecuted” row.

## Git map and preservation

The 16 registered worktrees were clean at the initial inventory. There was no observed dirty-index problem
to repair. Relative to the existing local remote-tracking refs:

| Ref | Initial tip | Relation to master |
|---|---|---|
| `master` / local `origin/master` | `e546d01` | Published P12 snapshot; same local ref value |
| `research/proposal-p13` | `6aeaf83` | 3 commits ahead |
| `research/proposal-p14` | `7811207` | 6 commits ahead |
| `research/proposal-p15` | `b0d806e` | 9 commits ahead |
| `research/proposal-p16` | `527a089` | 12 commits ahead |

P13→P14→P15→P16 form a linear ancestor chain. This maintenance work uses
`docs/language-results-audit-2026-10-26`, based on P16, in its existing isolated worktree.
Experiment branch tips and master remain unchanged. No extra worktree was created,
and no branch, job, data, checkpoint or historical commit was deleted.

No fetch, merge, push or live GitHub divergence check was performed. Equality with a
local `origin/master` ref does not establish current GitHub equality. Before any future
publication, inspect live divergence and the documented teammate `Presentations/`
exception; do not blindly pull master or rewrite history. Operational files and
credentials remain ignored; they are not part of the documentation commit.

## Verification and remaining work

Validation is specific to Markdown: UTF-8 decoding, main-document language scans,
hash/timestamp/code-token checks, numerical-value comparisons, repository-relative links,
whitespace checks and the final Git diff. Software code and metrics were unchanged, so
no training smoke or full pytest run is claimed. Historical test counts in the reports
remain dated historical evidence. PDFs were not rebuilt by changing their README pages.

Machine-local validation receipts and the exact final branch/commit are recorded in
ignored handoff notes. Remaining scientific work includes checking the exact P16 array
without resubmission, collecting completed results only when ready, and retaining the
original admission rule. Human expert review remains unavailable. Older mixed-language
technical archives are listed in the post-edit inventory rather than silently treated
as translated or newly verified.
