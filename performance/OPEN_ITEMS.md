# Performance Harness — Open Items

Deferred / unresolved questions for `performance/`. Remove items when they
land or a decision is recorded in `docs/planning/STATUS.md`.

## 1. C++ / Python scenario-matrix sync

**Open question.** How should Python `perf_*` scripts stay aligned with the
C++ matrices in `performance/cpp/perf_*.cpp`?

Candidate approaches (not chosen):

- Manual mirror (current interim for all ops)
- Shared declarative matrix (JSON/YAML) loaded by both drivers
- Codegen from a single source into C++ and Python

There is no binary-inspection path that can derive the C++ matrix.

**Interim:** Every C++ `perf_*.cpp` has a Python counterpart under
`performance/python/`. Matrices (default / `--full` / `--sizes`) are
hand-mirrored. Default CLI matches C++ (`--full` off). CFAR mid/large
GO/SO/OS Monte Carlo cold builds remain `--full`-only on both sides.

**Status:** open — decide later; do not invent a sync mechanism in the meantime.

## 2. `perf_mvdr_estimator` Python abort

**Fixed.** CuPy was imported before `holoscan.radar` via an eager
`performance/python/utils/__init__.py`. Empty that `__init__` so bootstrap
cannot pull CuPy early. Details in
`operators/angle_estimation/mvdr_estimator_op/OPEN_ITEMS.md`.

## 3. Python frame clock starts after source `emit`

**Fixed.** Python sources used to stamp `starts` *before* `emit(dict(cupy))`,
so first-tick `cold_ms` on multi-GiB cubes included CuPy→Holoscan wrap, not
just the DUT. Sources now stamp after `emit` (setup `t_first_tick` still
marks first compute entry). Sink sync is unchanged.

## 4. Python xl cold ≫ C++ — host ``cp.asarray`` inputs

**Fixed.** Building large IQ/power cubes on the host and ``cp.asarray``-ing
them produced ~150 ms first-tick cold on xl compensation/virtual (median
matched C++). Sink sync was ~0.01 ms of that; cost was post-emit→receive.
Device-native fills (last-axis tone broadcast / sparse peak scatter) bring
Python cold in line with C++ (xl virtual ~3 ms, xl TDM ~9 ms).
`PerfTerminalSinkOp` now also stamps `ends_receive` before sync for future
split diagnostics.
