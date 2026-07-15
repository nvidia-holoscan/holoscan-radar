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

## 2. Performance harness (landed)

`performance/cpp/perf_snapshot_gather.cpp` characterizes steady-state gather
cost and inline `[live] snapshot_window` override latency (window grows one
cell per axis → output re-shape + gather re-plan). It reuses the shared harness
(`perf_driver` / `perf_runner` / `perf_source` / `perf_sink`) via a `SnapTraits`
struct — no bespoke helpers — and sweeps channels x RD-map size x detections x
window. The `perf_snapshot_gather_smoke` ctest (label `performance`) runs a tiny
subset and verifies a report is emitted.
