# P21 validation and formal submission — 30 October 2026

Formal array **55416129** was submitted once at **2026-10-30T07:44:30.665848Z UTC**. Mapping0=C19, 1=A19, 2=C1, 3=A1; four fresh130-epoch runs, maximum concurrency2. Each requests1V100, 4CPU, 32GiB, 12hours in gpu-shared/gpu-shared-normal, jhu159, Lustre, no-requeue:48GPUhours total. Submission is not completed training or method admission. [Machine-readable receipt](P21_RECEIPT.json).

## Fixed scientific contrast

Original256-input UNI-last12/single-center detector C versus A with a training-only five-class Gaussian center auxiliary head. Extra325parameters, coefficient1, all five negative fields retained, full effective-batch object denominator. Auxiliary parameters use the original decoderLR3e-4/cosine/decay.01. Both arms retain effective16/physical4, 134updates and tail6, original encoderLR1e-5, AdamW/globalclip/BF16, cohort/input/weights, 130epochs, last checkpoint and native main-map decoding. Auxiliary maps are never used to create proposal points. [Complete frozen protocol](../../superpowers/specs/2026-10-30-proposal-p21-design.md).

This design was chosen after reading P20 scores. It is adaptive exploratory development, not an independent confirmation or a continuation of failed512scale. No coefficient, class-balance, seed, epoch, threshold or scale menu. Original two-seed F1=C/T1=A conjunction remains unchanged. No masks, PhaseB, fold2/fold3 evaluation. Repeated fold1 point-development results are notPQ/test or independent patient evidence.

## CPU and actual V100 validation

Focused49checks passed. Runner tests first failed17cases because the module was missing; implementation passed16with one incorrect handwritten focal derivative expectation. The correct positive derivative atlogit0 is−1/8−ln2/4; correcting that test yielded17passes without a production change. Verifier evidence-rejection test failed before implementation, then passed. This history is retained, not counted as a scientific failure.

Full CPU command was the project interpreter with `python -m pytest -q --basetemp=<fresh ignored directory>`, explicit worktree src/scripts imports, authoritative compact data, CUDA disabled and OMP/OpenBLAS/MKL4. Actual result: **968 passed, 3 skipped, 19073 warnings in 1426.74s (0:23:46)**, wrapper1434.43seconds, exit0. Temporary storage used a short fresh project-owned path, with current free space checked. No dependency upgrade, test plugin, omitted tests, metric/HV change or shared environment modification. Documentation and the independently executed analysis CLI were added while CPU ran; the model/verifier/test sources remained identical to the sealed tested implementation.

V100 smoke **55416052 COMPLETED/0:0**, **746seconds**, empty stderr. One30minuteGPU request; actual elapsed is within both requested1800s and predeclared2100s cleanup acceptance. No failed/retried GPU validation occurred. Node18newchecks passed in204.03s. Both seeds proved original P3/C parameter/RNG initialization and two-effective-step full trainable gradient, parameter, AdamW-state and CPU/CUDA-RNG equality. A original parameter/RNG initialization matches C; the extra zero-initialized convolution does not perturb its initial main parameters.

Both seeds A have second-step finite nonzero block12/block23/main-output/auxiliary gradients and parameter updates beyond pure decay. Auxiliary-only loss produces a nonzero shared-stem gradient; combined loss alone is not used to establish that claim. All frozen encoder tensors remain byte-identical, optimizer membership includes every and only trainable parameter, and auxiliary parameters are in the decoder group. Actual BF16 linear/convolution internals and FP32 interpolated outputs/loss were observed. Inference skips the auxiliary head, and mutating it leaves main predictions unchanged. C/A134-image smoke training, nine optimizer steps/tail6, cross-arm rejection and strict last/final reloading all passed.

The fixed134-image fit-only validation includes the first118fit IDs plus the first16later distinct annotatedDead fit IDs; all five annotated classes are represented. Full formal2134-image membership and original per-epoch permutation are unchanged. No audit predictions were used in readiness.

|Arm|Training peak GiB|1.5× safety estimate h|Task limit h|
|---|---:|---:|---:|
|C|4.0709|7.2367|12|
|A|4.0807|7.5015|12|

The unchanged formula includes measured134-image training, 130checkpoint saves, a final save and1800seconds overhead before multiplying1.5. These are estimates, not completion guarantees. Training memory bound24GiB. 29small evidence/preview files passed byte/SHA checks. Source commit **911a3718c157ceb51b2924feb18c50a3165a1a73**, **438sealed files**, manifest **c750d7d059d8837e7dea17d9a37794e6c3f526c7d3aa13d17d5f220de2fe53a3**. All sources and three job/readiness scripts match the smoke-accepted hashes and original submission receipt.

## Point-example output

Each final run saves8audit and4fit PNGs plus coordinate JSON: RGB, GT centers, predicted main-map points, and unique radius3match/miss/extra panels. GT interiorDead is magenta; otherGT cyan; matched predictions green, missedGT red, extra predictions orange. Selection uses GT strata and ID order only, never prediction quality. Both arms use the same deterministic selection. The smoke created twofit-only one-epochpreliminary images perarm; these validate rendering/output provenance and do not describe130-epoch quality. No audit samples were inferred during smoke.

Images/coordinates are preserved recursively with final results. One minor review limitation remains: when a requestedGTstratum is unavailable, the implementation expands the other-image quota before sorted fallback, whereas the frozen prose specifies fixed quotas then fallback. Current completefit/audit strata have sufficient support and current selections are unaffected; future deficient strata can yield a different deterministic selection. This does not change predictions, metrics or admission.

## Operational gates and review

Fresh design and code reviews found no blocking model issue. Controller findings were corrected before formal submission: allocation reservations now cover account-wide GPUjobs, not only this user; smoke-ready script hashes bind to original receipt/provenance; resource output is evaluated, not merely collected; short scheduler time formats are accepted. The earlier smoke textjob-ID file contains literal backslash-n; historical evidence is preserved and the correctJSONreceipt is authoritative. Formal textreceipt uses an actualnewline. No submission was replayed.

Before formal submission, we rechecked exact completedsmoke accounting, full CPU exit0, all source/script hashes, currentallocation and account-wide conservative full-walltime reservations plus48newGPUhours, filesystem free space and user/groupquota100GiBheadroom, and scheduler admission. The current[SDSC guide](https://www.sdsc.edu/systems/expanse/user_guide.html) was checked; computations run throughSLURM. A successful admission-only `sbatch --test-only` was followed by one real `sbatch --parsable`, exclusive local/remote guards and immediateJSONreceipts. Existing unrelated jobs/connections were preserved. Only small evidence returned toLM1; runtime/data/UNI were reused without upgrades or bulk transfers. Large checkpoints remain onLustre and independent externalarchive is outstanding.

After all four runs finish130epochs, verify preservation/source/input/config/cohort, collect small outputs and apply the original conjunction. A failed/inconclusive result closes this fixedauxiliary recipe without automatic menus or expansion. Reports retain nominal conditional bootstrap intervals without family-wise correction across historical development rounds; neither positive descriptive intervals nor example images establish independent confirmation.
