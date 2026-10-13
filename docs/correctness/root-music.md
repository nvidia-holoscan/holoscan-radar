# RootMusic correctness investigation

Investigation date: 2026-10-13 (America/New_York). Branch: `deustice/root-music-correctness`; isolated worktree: `.worktrees/correctness/root-music`, based on `ffa63b3a`. No public contract, precision, source-count algorithm, polynomial, smoothing algorithm, unsupported-mode rejection, or deployment clock/power setting changes. The validated fix is landing on `deustice/optimization-skills` with user approval. No pushes or external issues.

## Root cause

The operator intentionally creates an all-NaN companion when the leading polynomial coefficient fails its existing norm floor (`1e-20` in float, `1e-40` in double). It then calls `cusolverDnXgeev`, assuming a positive per-item `info` will report non-convergence and allow that item's roots to be blanked. In the installed CUDA 13.0/cuSolver runtime, an all-NaN **float** companion instead returns API status 7 (`CUSOLVER_STATUS_INTERNAL_ERROR`). The wrapper throws before checking `info`, aborting the entire tick. This is an invalid numerical input produced by the operator, not evidence of an HSDK defect.

The recipe's IQ fixture repeats the same finite waveform on every channel. Its covariance is analytically rank one. Floating-point roundoff in its null-space eigenvalues affects the existing EFT source count and the selected noise-space basis. On the failing float frame, the resulting projector has an exactly zero extreme-lag coefficient. Covariance, eigenvectors, projector, and polynomial coefficients remain finite; the companion kernel itself introduces the NaNs. Neither input lifetime nor cuSolver workspace reuse is needed to reproduce the failure.

| Reproduction | Failing operation | Exact location | Leading coefficient / result |
| --- | --- | --- | --- |
| Unchanged native `baseline-small`, float | `cusolverDnXgeev`, degree 6 | Frame 2, item 0 (zero-based frame 1; input amplitude 1.125) | `0 + 0i`; all 36 companion entries non-finite; status 7 |
| Unchanged warmed `spatial-smoothing-medium`, float | `cusolverDnXgeev`, degree 12 | Measured graph frame 56, item 0 (zero-based 55; first live override after 5 warmup + 50 measured frames) | `0 + 0i`; all 144 companion entries non-finite; status 7 |
| Native `baseline-small`, double | Same solver, degree 6 | Frame 2, item 0 | Approximately `4.9773e-8 - 4.4355e-5i`; finite companion; status 0, info 0 |
| Direct cuSolver replay, float | Same solver without Holoscan | Fresh handle; captured NaN matrix, three calls | Status 7 each time; info remains 0 |
| Direct cuSolver replay, float | Same solver without Holoscan | Captured finite degree-14 matrix, three calls | Status 0, info 0 each time |

The recorded degree-6 float workspace is 265216 device bytes / 1024 host bytes; degree-12 float is 266240 / 2560. Allocation and API arguments use the sizes returned by `cusolverDnXgeev_bufferSize`, column-major companions, `lda = degree`, matching complex storage/compute precision, and `NOVECTOR` for both eigenvector modes. The isolated replay uses the same requirements and reproduces the failure independently of graph scheduling, item offsets, streams, or prior frames. The [CUDA 13.0 cuSolver documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cusolver/index.html#cusolverdnxgeev) distinguishes API errors from numerical `info`; a non-finite matrix cannot be assumed to yield only positive `info`.

Initial unchanged/diagnostic builds used the default CMake configuration. Final fix validation uses **Release**, C++20, native CUDA architecture, CUDA 13.0.88, GCC 13.3.0, HSDK 5.0.0, and MatX 1.0.0 in immutable image `sha256:d4ccdf30d119f555320cb629704a50614dab7cd0eecd1f7513481d52c0b7ddb2`. The SDK testing/core/schema/logger library hashes match the prior campaign. Final executable, source, build-file, and actual loader dependency hashes are recorded separately. The isolated container mounts only the worktree; embedded Git discovery cannot traverse its external Git directory, so host-verified base commit and source hashes are the provenance authority.

## Focused fix

Before calling `geev`, a stream-ordered kernel classifies each companion for finite real and imaginary entries. The wrapper downloads the flags, calls the existing solver only for finite companions, and fills the other items' roots with NaNs. It resets their per-item info slots so cached state cannot reuse an earlier error. This preserves the existing documented zero-source / NaN-DOA behavior for an unrootable polynomial. It does not discard a valid item or suppress an API error from a finite matrix. Positive solver `info` retains its original blanking policy; negative `info` and API failures still fail loudly.

The validity buffer belongs to the existing solver state and follows its degree/precision/item-capacity allocation lifecycle. Output pools, publication, retained inputs, and precision are unchanged. The additional matrix classification, host transfer, and synchronization are correctness costs; no performance improvement or timing acceptance is claimed. Temporary diagnostic instrumentation was removed from production code.

## Regression evidence

- The unchanged float native and warmed smoothing failures are preserved, alongside double captures, upstream binary dumps, workspace sizes, and standalone replay logs.
- A graph regression reproduces three scaled equal-channel float frames on one operator: the degenerate second frame yields zero sources and NaN DOAs, and the following frame completes. Eigenvalues remain finite on every frame.
- Two deterministic solver regressions use the analytic polynomial `z² - 1/4` (roots `±1/2`), a zero-leading polynomial, and an empty item. Across three frames, valid and degenerate items swap positions on the same solver state. Float and double both preserve valid roots, blank only the degenerate item, and process empty items without failure.
- **96/96 focused unit tests pass:** the complete RootMusic suite, both new solver tests, analytic point-target simulator, source-count references, spatial smoothing, and native owning-snapshot/staging tests. Existing analytic DOA and precision-agreement tolerances are unchanged.

- **Complete native matrix:** all 40 recipe cases × both precisions × two isolated captures = **160/160 successful captures**. All 80 case/precision repeat pairs match exactly, including output bytes, shapes, dtypes, memory kinds, emission metadata/counts, diagnostics, and termination. Intentionally unwired optional ports are checked for zero emissions; connected ports each emit all three frames.
- **Historical equivalence:** all 49 successful captures from the prior first sweep (9 float, 40 double) match the final Release build exactly. Previously failed captures remain preserved as failures; they are not represented as before/after equivalence passes.

- **Affected warmed/cached timing contexts:** all four cases (`spatial-smoothing-medium`, `spatial-smoothing-large`, and both explicitly cached variants) × both precisions × three isolated repeats = **24/24 completed runs**. Each retains five warmup frames, fifty measured samples, the live override, and Params reversion; no cases or failures were filtered. These runs validate execution, not performance acceptance.

- **Fresh pre-fix Release confirmation:** the complete 40-case float matrix reproduces exactly the same 31 status-7 failures as the prior campaign; the nine other float cases complete. The companion double `baseline-small` capture completes. All four affected warmed/cached float cases reproduce status 7. The three new regressions (graph, float solver, double solver) fail specifically with status 7 before the fix and pass after it. The degree-2 double regression also demonstrates that double is not generally immune to invalid-matrix API errors.
- **Pinned checks:** focused checks for every changed/new file and the complete `pre-commit run --all-files` suite pass using the repository configuration and a compatible cached Node 22 runtime for pinned markdownlint. No hook versions or repository configuration were changed.

The evidence directory was relocated from temporary `artifacts/root-music/` to the ignored build tree after collection. Raw command records retain their original output prefix; `artifact-relocation.json` records that mapping. No historical reports were moved or rewritten. At landing, the entire fresh build/evidence tree was preserved as `build-correctness/root-music/` in the main checkout before retiring the isolated worktree. Raw commands and build files retain their original worktree paths.

## Evidence and campaign decision

Fresh logs, binaries, binary diagnostic dumps, executable/runtime identities, per-case commands, native outputs, and chronological timing outputs are retained under the main checkout’s `build-correctness/root-music/artifacts/root-music/`. Historical evidence in the original checkout was read only; old reports were not rewritten. The original checkout's user edits and `.agents/` were preserved.

All 40 cases in both precisions can clear the runtime/native blocker on the fixed source, including all 31 previously failing float cases. The four affected warmed/cached cases also clear runtime completion in both precisions. Rebaseline the fixed source before any performance optimization; retain the original failed reports as historical evidence. That does not establish performance acceptance: the campaign's A/A stability limits and performance gates remain independent. These noise-free rank-one automatic-source fixtures are degeneracy stress cases. Existing analytic tests provide the accuracy checks for the supported DOA scenarios. A zero-source output census is retained without concealing blank results.

**Separate contract question, left unresolved:** STATUS decision S2 says “rank-deficient covariance → fall back to 1 + warn.” The existing implementation gates that fallback on `num_snapshots < num_channels`, rather than algebraic covariance rank. With sufficient snapshots but identical channel signals, EFT can select three sources in the four-channel fixture, and its result varies with amplitude/precision. This investigation does not change EFT, add an algebraic-rank test, or reinterpret S2. Developer review must decide the intended rank-deficiency scope before any algorithm or documented-behavior change. The runtime failure can be cleared independently; this report does not certify correct automatic source counts on singular covariances.

Native owning snapshots validate copied output ownership and frame completion. They do **not** prove that consumers can retain borrowed RootMusic output pool loans across arbitrary later ticks; the prior campaign's retained-output coverage gap remains. The fix does not change output allocation or ownership. GPU memcheck was attempted but `compute-sanitizer` is absent from the immutable image and host; no sanitizer pass is claimed. Python binding execution remains unavailable in this SDK image. No SDK/MatX source revision is inferred from installed package versions.

## Evidence index

All paths below are relative to the main checkout’s `build-correctness/root-music/artifacts/root-music/`:

| Record | Contents |
| --- | --- |
| `diagnosis.json`, `diag-native/`, `diag-timing/` | Finite input/covariance/projector checks, eigen residuals, exact leading coefficients, matrices and failing frame/item logs |
| `replay.cu`, `replay-*.log` | Standalone cuSolver reproduction without Holoscan |
| `before-release/native-runs.json`, `before-release/all-runs.json` | Fresh unchanged-source float matrix and warmed/cached failures, including commands |
| `before-release-graph-regression.log`, `before-release-solver-regressions.log` | All three new tests fail with status 7 against pre-fix production |
| `matrix/inventory.json`, `matrix/runs.json`, `matrix/repeats.json` | All 40 cases, 160 complete fixed-source captures and 80 exact repeats |
| `matrix/timing-runs.json`, `matrix/timing-*.json` | 24 complete isolated warmed/cached runs, both precisions and all chronological samples |
| `historical-comparison.json` | Exact comparison to every previously successful first-sweep capture, with historical hashes |
| `zero-source-output-census.json` | Explicit zero-source item counts by case/precision/emission; optional wiring retained |
| `focused-tests.log`, `pinned-final-*.log` | Focused C++ and final pinned checks |
| `final-build-runtime.json`, `final-source-hashes.json`, `final-runtime-links.log` | Final binaries, actual runtime dependencies, compiler and build/source identities |
| `deployment-power.log`, `deployment-sysfs.json`, `deployment-sysfs-final.json` | Read-only deployment mode and clock configuration snapshots |
| `campaign-decision.json`, `artifact-sha256.json` | Per-case unblock scope and retained evidence integrity inventory |
