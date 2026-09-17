# Operators — Open Items

Deferred, agreed-upon work that spans the operator libraries. Each item
records the intent and reason it was deferred so it can be picked up without
re-litigating the design. Remove items when they land and note the commit in
the PR.

## 1. Add automatic C++ formatting checks

**Status:** open; the current pre-commit configuration runs `cpplint` but not
`clang-format`.

Add a checked-in `.clang-format` policy and a pinned `clang-format`
pre-commit hook, then run the same formatter check in CI. This would make
equivalent formatting choices deterministic across operator sources instead
of relying on manual consistency.

Before landing, choose and document the formatter version and verify the
initial repository-wide formatting diff is acceptable. Avoid mixing the
bootstrap formatting sweep with behavioral changes.

## 2. Support staggered and non-uniform TDM timing

**Status:** deferred 2026-07-22; uniform repeating TDM is the v1 scope.

`TXSchedule`, `MIMODeinterleaveOp`, `TDMDenseCompensationOp`, and
`TDMSparseCompensationOp` currently require a repeating cycle in which each TX
appears once per cycle with uniform per-TX spacing. A permutation of the TX
order is supported; varying dwell, count, or inter-firing interval is rejected.

Supporting staggered TDM requires a per-TX timing representation, generalized
deinterleave indexing, Doppler/phase correction based on actual firing times,
simulator support, and dense/sparse parity tests. Do not reinterpret
`FMCWWaveformDescriptor.chirp_repetition_interval`: it remains the effective
per-TX slow-time sample period for the current uniform workflow. The future
contract must define how timing is represented and kept coherent across both
TDM operators before relaxing `TXSchedule` validation.

## 3. Support M+1 empty-band DDM ambiguity resolution

**Status:** research item deferred 2026-09-09; current DDM uses M occupied
bands in an M-band packed layout.

Packed M-band DDM provides no empty-band observation from which one frame can
identify the cyclic Doppler ambiguity. `DDMSparseDemultiplexOp.fold_idx`
therefore remains an external escape hatch: missing input uses
`SafeDefault{0}`, which is correct only in the principal reduced velocity
interval. This is a DDM-wide waveform limitation, not a sparse-demultiplex
implementation gap; dense demultiplexing does not inherently resolve it.

Investigate an M+1 layout in which M TX comb lines occupy M of M+1 Doppler
bands and the empty band's location supplies ambiguity information. Before
locking a contract, establish behavior for multiple targets and comb
collisions in one range bin, guard-band loss, noise/clutter, and band-boundary
crossings. Compare the usable velocity and detection performance against
packed M-band DDM and external staggered-CRI or code-rotation observations.

Implementation would affect `DDMCodeDescriptor`, dense and sparse demultiplex
tables, `DDMPowerFoldOp` shape/index semantics, simulators, applications, and
cross-path tests. Do not introduce a generic `DDMFoldResolverOp` until a
specific observable and resolution algorithm are selected. Keep `fold_idx`
available for applications that resolve the ambiguity externally.

## 4. Add code-aware DDM clutter suppression

**Status:** corrected and deferred 2026-09-16; dense and sparse DDM v1
explicitly do not support clutter filtering.

The existing `ClutterFilterOp` is not valid after `DDMDenseDemultiplexOp`:
demultiplexed outputs are already Doppler-domain maps, so mean subtraction acts
across Doppler bins and creates a range-aligned pedestal. Applying that filter
directly to coded pre-demux chirps is not equivalent either, while power folding
concentrates zero-Doppler clutter into the folded zero bin. Both DDM workflows
therefore currently require scenes where omitting clutter suppression is
acceptable.

Design a code-aware pre-demux or folded-domain suppression method and validate
its Pfa/Pd impact before adding it to either workflow, or define a dedicated
post-demux operator that notches physical zero velocity in the RD domain. Any
new operator or parameter is a contract change.

## 5. Extend sparse DDM beyond linear integer-bin codes

**Status:** deferred 2026-09-09; dense DDM retains broader code support.

`DDMPowerFoldOp` and `DDMSparseDemultiplexOp` currently require a linear
M-band code whose offsets map exactly to Doppler bins. Fractional-bin offsets
and nonlinear codes reject. `DDMDenseDemultiplexOp` continues to support
nonlinear full-axis correlation.

Fractional offsets require an interpolation/leakage model rather than the
current exact gather table. Nonlinear codes require detection and sparse
recovery definitions based on correlation, not band folding. Treat these as
separate algorithm and contract extensions; retain dense-versus-sparse
snapshot parity as the acceptance oracle wherever both paths are defined.

## 6. Support frame-to-frame DDM code rotation

**Status:** deferred 2026-09-09; `ddm_code` is static in all current DDM
operators.

Code rotation across frames can mitigate comb collisions between targets in
the same range bin and can contribute an external observation for fold
resolution. One operator instance cannot currently consume a new
`DDMCodeDescriptor` each tick; callers must use separate statically configured
instances.

A future live-code contract must keep `DDMPowerFoldOp`,
`DDMDenseDemultiplexOp`, and `DDMSparseDemultiplexOp` temporally coherent,
invalidate/reuse mapping and phase caches safely, and warn once per unseen
configuration. It must also define how detections and `fold_idx` stay paired
with the code that produced them.

## 7. Add a sparse DDM application workflow

**Status:** deferred 2026-09-09; intentionally excluded from the first
operator implementation.

Add an `applications/fmcw/mimo_ddm` sparse-DDM graph/config path that wires
`DDMPowerFoldOp → CFARDetectorOp → DDMSparseDemultiplexOp → angle estimator`,
including `preintegrated_looks = M_tx * N_rx`, geometry and validity sidebands,
and performance/reporting integration. The example must label
`SafeDefault{0}` fold handling as principal-interval-only; it must not present
unresolved full-velocity operation as generally correct. Add deterministic
smoke coverage and compare its recovered snapshots against the dense DDM
pipeline for supported linear codes.
