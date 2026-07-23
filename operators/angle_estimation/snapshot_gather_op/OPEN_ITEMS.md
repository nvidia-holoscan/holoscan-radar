# SnapshotGatherOp open items

## 1. `is_valid` bool dtype vs Holoscan 4.3 transport

**Context.** `OPERATOR_API.md` specifies `is_valid` as `bool`. The operator
emits `kDLBool` to match `CFARDetectorOp` `detection_mask` (see
`operators/detection/cfar_detector_op/OPEN_ITEMS.md` §1). On HSDK 4.3,
emitting the mask through the graph logs:

```text
[error] [tensor.cpp:820] Unsupported DLPack data type (code: 6, bits: 8, lanes: 1)
```

(`kDLBool`). GXF cannot map `kDLBool` to a numpy typestr on the C++ message
path; the error is logged and swallowed, so **C++-to-C++ port wiring and
host capture via raw `cudaMemcpy` still work**.

**Impact.** Any **Python** consumer of `is_valid` is at risk: wrapping the
tensor as a `PyTensor` calls `numpyTypestr()` and **throws**
(`Unable to determine NumPy dtype from DLPack tensor` in
`python/holoscan/core/dl_converter.cpp`). A Python pipeline that reads
`is_valid` can fail on HSDK 4.3.

**Questions for the team** (team-wide policy — defer until full team present):

- Keep `bool` in the contract and treat HSDK 5 as the validation target?
- Add a compatibility `uint8` mask port or change the contract dtype?
- Document a hard "C++ only until HSDK 5" caveat for bool mask outputs?

**Status:** open — aligned with CFAR `detection_mask` policy; not validated on
a finalized HSDK 5 build in this environment (only 4.3.0 installed).

## 2. Multi-look snapshot axis transposition (fixed)

**Symptom.** Any sparse angle estimator (Beamscan, MVDR, MUSIC, Root-MUSIC,
ESPRIT) fed a multi-look gather (`snapshot_window` with a non-zero half-extent,
so `num_snapshots > 1`) recovered wildly wrong angles — a clean single target at
boresight scattered across the full FOV. First surfaced in
`applications/fmcw_aoa_estimators` (`snapshot_window: [2, 1]`, K = 15) as "no
clusters for Root-MUSIC/ESPRIT" and grid estimators pinned to their scan-grid
edges. Single-look apps (`fmcw_simo`, `snapshot_window: [0, 0]`) were unaffected.

**Cause.** `gather_snapshots_kernel` allocated `snapshot` with the contract
shape `(num_detections, num_channels, num_snapshots)` but *wrote* it in
`(num_detections, num_snapshots, num_channels)` order (`snap_slot *
num_channels + c`). Every consumer indexes per the contract as
`xi[c * num_snapshots + s]` (see `mvdr_stage.hpp` / `music_stage.hpp`
covariance kernels), so the channel and snapshot axes were transposed for
K > 1. For K = 1 the two layouts coincide, which is why single-look pipelines
and the K = 1 tests never caught it.

**Why tests missed it.** The gather's own C++/Python tests asserted the
`(N, C, K)` *shape* but indexed the data as `slot * num_channels + c` — i.e.
they encoded the same transposition, so they agreed with the buggy kernel. The
estimator unit tests build their snapshots directly in `(N, C, K)` (never via
the gather), so they were correct and unaffected.

**Fix.** Write `out[c * num_snapshots + snap_slot]` (channel-major,
snapshot-minor), matching the documented contract and the consumers. The gather
tests were corrected to index `(N, C, K)`.

**Status:** fixed. Aligns code to `OPERATOR_API.md`
(`snapshot` = `(num_detections, num_channels, num_snapshots)`); no contract
change.

## 3. Performance harness (landed)

`performance/cpp/perf_snapshot_gather.cpp` characterizes steady-state gather
cost and inline `[live] snapshot_window` override latency (window grows one
cell per axis → output re-shape + gather re-plan). It reuses the shared harness
(`perf_driver` / `perf_runner` / `perf_source` / `perf_sink`) via a `SnapTraits`
struct — no bespoke helpers — and sweeps channels x RD-map size x detections x
window. The `perf_snapshot_gather_smoke` ctest (label `performance`) runs a tiny
subset and verifies a report is emitted.
