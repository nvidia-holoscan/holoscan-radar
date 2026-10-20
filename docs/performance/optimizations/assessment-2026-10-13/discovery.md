# Fresh deployment baseline assessment: discovery

Assessment-only campaign on `deustice/optimization-skills`, starting 2026-10-13.
The user authorized four hours of measurement qualification (14,400 seconds).
Build and preparation durations are recorded separately. No full-baseline
collection, operator optimization, worktrees, commits, pushes, deployment-setting
changes or installation are included in this phase.

Discovery found 32 public operators, of which 26 have existing performance
harnesses: **760 declared case IDs / 1,520 precision-case combinations**.
Declaration coverage is not execution or qualification evidence. Executable
inventories and verification will be checked on fresh builds before timing.
Only the four qualification representatives are built in this phase; building
and measuring the complete matrix follows a separate forecast and scope decision.

## Qualification configuration

| Family | Operator | Existing case | Precision |
| --- | --- | --- | --- |
| Reduction | clutter_filter | baseline-medium | Float and double |
| FFT | fmcw_range_processor | baseline-small | Float and double |
| Eigensolver | music_estimator | baseline-small | Float and double |
| Irregular clustering | cluster_dbscan | baseline-small | Float and double |

Each family receives at most one hour, totaling the confirmed four-hour budget.
Explore 50/500/2,000 measured frames and 3/7 isolated repeats with ten calibration
pairs per affordable configuration. Keep median aggregation, five warmup frames,
the existing two-frame process warmup and the first-50 protected window. Freeze
selection before fresh held-out validation, with a matrix-derived pair proposal
and a 30-pair floor. A representative qualifies only its measured case.

Protected metrics are full-window and first-50 median/p90/p99, setup, cold and
applicable live transitions. Proposed P90-delta/IQR limits are steady 5%/10%,
setup 15%/25%, cold 20%/30% and live transition 25%/40%. Confidence width and
+/-1% precision remain diagnostics. No threshold relaxation is authorized.
These static pilot cases do not qualify live/cached transitions in other cases.

An inventory-time tooling finding corrected an accidental second multiplication
of the operator-family count in qualification. A host regression checks equality
with the pair-budget proposal. New evidence uses the corrected formula;
historical reports are preserved without relabeling. No candidate is being
accepted under a changed gate.

## Deployment and provenance

The available compatible HSDK 5 image is verified by immutable ID
`sha256:d4ccdf30d119f555320cb629704a50614dab7cd0eecd1f7513481d52c0b7ddb2`.
Host telemetry reports NVIDIA driver 580.00, CUDA 13.0 and 120W deployment mode.
Container preflight finds GCC 13.3.0, CUDA tooling and Nsight Systems 2026.4.1;
Nsight Compute is unavailable. Host Holoscan CLI is unavailable, so commands
use an explicit Docker wrapper and matching absolute mount/work directories.
Clock/power configuration is recorded and preserved. Missing observations stay
unknown. The cached SDK source is at `5fed2c24acf5b0bc1e538deeaae311efd8d256e1`;
that is not by itself proof of the installed SDK source revision.

Fresh Release builds, archived runtime libraries, source/recipe hashes,
correctness captures, forecasts, raw frames and budget ledgers are stored under
`build-optimization/operator-perf/assessment-2026-10-13/`.

## Requested operator inventory

| Operator | Declared cases | Declaration precision pairs | Verification |
| --- | --- | --- | --- |
| beamscan_estimator | 40 | 80 | Fresh verification pending |
| esprit_estimator | 40 | 80 | Fresh verification pending |
| music_estimator | 52 | 104 | Fresh verification pending |
| mvdr_estimator | 46 | 92 | Fresh verification pending |
| phase_shift_beamformer | 36 | 72 | Fresh verification pending |
| rank_one_residual_screen | 17 | 34 | Fresh verification pending |
| root_music_estimator | 40 | 80 | Fresh verification pending |
| snapshot_gather | 16 | 32 | Fresh verification pending |
| subarray_selection | 42 | 84 | Fresh verification pending |
| cluster_dbscan | 29 | 58 | Fresh verification pending |
| cfar_detector | 64 | 128 | Fresh verification pending |
| detection_reduction | 30 | 60 | Fresh verification pending |
| feature_assembly | 15 | 30 | Fresh verification pending |
| multi_source_expansion | 8 | 16 | Fresh verification pending |
| clutter_filter | 21 | 42 | Fresh verification pending |
| ddm_dense_demultiplex | 11 | 22 | Fresh verification pending |
| ddm_power_fold | 3 | 6 | Fresh verification pending |
| ddm_sparse_demultiplex | 22 | 44 | Fresh verification pending |
| doppler_notch | 26 | 52 | Fresh verification pending |
| fmcw_doppler_processor | 43 | 86 | Fresh verification pending |
| fmcw_range_doppler_processor | 66 | 132 | Fresh verification pending |
| fmcw_range_processor | 43 | 86 | Fresh verification pending |
| mimo_deinterleave | 3 | 6 | Fresh verification pending |
| tdm_dense_compensation | 12 | 24 | Fresh verification pending |
| tdm_sparse_compensation | 23 | 46 | Fresh verification pending |
| virtual_array_assembly | 12 | 24 | Fresh verification pending |

## Exclusions and visible limits

The six operators without harnesses are outside the requested timing scope:
CPHDReader, TensorDump, ApertureManager, Backprojection, PhaseCorrection and
SARRangeCompression. Recipe creation is not authorized.

DetectionReduction will use its explicitly authorized input-derived semantic
policy with fresh witnesses. Custom weighted scorers and consumer equivalence
remain separate coverage questions. Other outputs retain exact comparison.
RootMusic's runtime fix is included in the current source, but its singular
covariance source-count interpretation and borrowed-output retention remain
visible review gaps. Native snapshot ownership does not close borrowed-loan
coverage. Missing CUDA sanitizer and Python-binding capabilities are not passes.

## Next milestone

Publish representative qualification results, absolute A/A measurements, limits,
confidence intervals, early/full-window drift, phase durations and budget status.
Then forecast a baseline collected in rounds: small/medium/large static cases
across all operators first, followed by curated variants, live overrides,
cached/unseen transitions and optional outputs. Complete matrices still need
correctness and casewise calibration; no pilot certifies unmeasured workloads.
