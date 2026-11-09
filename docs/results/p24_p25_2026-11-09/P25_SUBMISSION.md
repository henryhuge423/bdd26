# P25 validation and submission — V100 validation running

Updated9November2026. Final completeCPU **1086passed/3skipped/19073existingwarnings** in679.66s;wrapper684.21s. Focused87passed. Memoryguard regression1RED→6guardPASS; direct CLI refusal/full24 checkpoint round-trip included. Reviewed deployment wrappers had no Important/Critical finding. Code/source seal remains unchanged after later documentation edits.

Smoke **55635543** submitted once at about08:51UTC and RUNNING onexp-7-59. Node66tests passed63.00s; actual two-seed update and final readiness are still running. No formal job has been submitted at this checkpoint. **Do not replay** deployment or submission. Sourcecommit93fea3661baf6d9b7187bc442789fec95917b981,485files,manifestb419ac061fffc7d36f0b87614d4ec3f7cb1ad93994a35947bd0a9f6e71153354. Source implementationf8ccb8f. C/D×19/1,fresh130epochs,lastcheckpoint,originalgate;formalmax48GPUh.

Initial fullCPU before the review fix also passed1086/3 in669.20s; final committed-source run above is authoritative. Both CPU runs and all logs preserved. Initial fullstate/RNG equality is hashed onCPU; exact updates/gradients/AdamW/RNG parity are tested on the same actualGPU. No cross-device bit-equivalence claim.

Project/scratch本人bdd26/p25_20261109_01,smokesiblingp25_smoke_20261109_01;onlynewdirectoriesOST43+150GiBreserve. Fresh three-scope quotas>=100GiB/1000files,fsync/readback andallocation passed. Allocation1900/spent1120,accountqueuefull-walltime60SU,smokeadditional.5;formal48requiresfreshadmission. [Official user guide](https://www.sdsc.edu/systems/expanse/user_guide.html) and livepartition/QoS checked;compute throughSLURM.

See[frozen spec](../../superpowers/specs/2026-11-09-proposal-p25-design.md). Botharms safeprojection<12h/peak<=24GiB andsource-boundCOMPLETED0:0 readiness required before formal submission. Submission is not scientific admission.
