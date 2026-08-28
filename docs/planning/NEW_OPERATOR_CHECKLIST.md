# New Operator Checklist

Checklist for landing a new operator in **holoscan-radar**. The locked contract
lives in [`OPERATOR_API.md`](OPERATOR_API.md); tests in
[`TESTING.md`](TESTING.md); module conventions in
[`AGENTS.md`](../../AGENTS.md) and
[`operators/common/README.md`](../../operators/common/README.md).

For *how* to implement well (hot-path discipline, diagnosis, `[live]` /
calibration pitfalls, performance characterization — lessons from landed
operators), see [`OPERATOR_DEVELOPMENT.md`](OPERATOR_DEVELOPMENT.md).

Use **Before you're done** as the final gate; the sections below are the
detailed how-to if you need them.

**Vertical slices are fine; they do not clear this checklist.** Implement and
land intermediate slices (one policy, one CFAR variant, …) to make progress —
that is expected. A slice is **never** enough to mark the checklist done or
call the operator ready. Clear this checklist only when the **full**
`OPERATOR_API.md` contract for the operator is implemented (every port,
parameter, and documented mode), not when a useful subset works and the rest
throws / sits in `OPEN_ITEMS.md`.

---

## Before you're done

- [ ] Matches **`OPERATOR_API.md` in full** — every port, shape, dtype,
      parameter, `[live]` tag, and documented mode; no staged contract
      features left unimplemented. Contract changes documented in the same PR.
- [ ] **`STATUS.md`** updated (claim in flight → done or refreshed). Do not
      label the operator checklist-complete / ready while contract surface
      remains unfinished.
- [ ] **Build wired** — group + operator `CMakeLists.txt`, `metadata.json`,
      group `README.md`, SPDX on all new files.
- [ ] **`setup()` complete** — single-entry `TensorMap` ports, `initialize()`
      arg setters, `[live]` override ports where required.
- [ ] **`compute()` correct** — Exact `[live]` resolve (wiring /
      retention), rank-agnostic shapes,
      stream-ordered CUDA, clear errors on unsupported paths.
- [ ] **Tests** — `tests/cpp/<group>/test_<operator>.cpp` covers `TESTING.md`
      unit checks; all output ports wired in harness tests; float/double +
      error paths where applicable.
- [ ] **Performance** — `performance/cpp/perf_<operator>.cpp` harness + CMake
      smoke test when a hot path exists (see **Performance** below).
- [ ] **`ctest -L unit`** and **`pre-commit run --all-files`** pass.
- [ ] **Performance** — `performance/cpp/perf_<operator>` consumer when the
      operator is GPU/hot-path; **includes inline `[live]` override
      measurement** whenever the contract has live params (see
      `OPERATOR_DEVELOPMENT.md` → Performance characterization).
- [ ] **Open items** — unresolved *team* questions (contract ambiguity, SDK
      gaps, math decisions) recorded in
      `operators/<group>/<operator>/OPEN_ITEMS.md`. Missing contract
      features are not open items — they are unfinished work that blocks
      checklist completion.

---

## Before you start

- [ ] Read the operator's section in **`OPERATOR_API.md`** (ports, shapes,
      dtypes, parameters, `[live]` tags, static vs overridable).
- [ ] Check **`STATUS.md`** — claim the work in **In-flight work**; do not
      duplicate an active branch.
- [ ] Read **`WORKFLOWS.md`** if the operator sits on a modality chain —
      confirm port wiring and upstream/downstream dtypes.
- [ ] Check for an operator-local **`OPEN_ITEMS.md`** if this operator already
      has one; do not silently pick answers to documented ambiguities.

---

## Contract & file layout

- [ ] **Directory:** `operators/<group>/<operator_snake>/` (group =
      `fmcw`, `detection`, `clustering`, `angle_estimation`, …).
- [ ] **Three-file split** (GPU operators):
  - `<op>.hpp` — contract reference: ports, parameters, class shell
  - `<op>_stage.hpp` — kernels / MatX stage (CUDA-TU only; includes
    `matx_utils.hpp`)
  - `<op>.cu` — `setup()`, `compute()`, explicit template instantiations
- [ ] **Header** documents every port/shape/dtype and flags implementation
      gaps honestly (what throws vs what works).
- [ ] **No contract drift** — if the implementation differs from
      `OPERATOR_API.md`, change the doc in the same PR and call it out in
      the PR description.

---

## `setup()` — ports and parameters

- [ ] **Data ports:** single-entry `holoscan::TensorMap` in/out; port name is
      the contract (see `operators/common/README.md`).
- [ ] **Precision:** `template <typename T>` with `T ∈ {float, double}` only;
      `static_assert` in the header.
- [ ] **Parameters** match `OPERATOR_API.md` names, types, and defaults.
- [ ] **Schema payloads:** any `[live]` value that owns a container follows
      [`SCHEMA_PAYLOADS.md`](SCHEMA_PAYLOADS.md) — one committed canonical
      `.fbs`, generated outputs untracked, bounded Tensor fields, validated
      bridges, evolution discipline, install surface, and drift/Graph tests.
- [ ] **`[live]` parameters:** declare with `spec.param(...)`, register the
      `Input<>` in `setup()`, then add the `contract()` entry
      unconditionally via `add_live_override_to_contract` — an unwired port
      is legal and reads as "no override", under `OnAligned` as much as
      under `OnEach`. No `expect_*` parameter: wiring is the app's only
      statement.
- [ ] **Static parameters** (kernel-selecting enums, structural modes): no
      override port; document that callers instantiate separate operators.
- [ ] **`initialize()`:** call `ensure_radar_arg_setters()` and register any
      operator-local enum argument setters before `Operator::initialize()`.

---

## `compute()` — implementation

- [ ] **Receive / emit** via single-entry `TensorMap` helpers; reject
      multi-entry maps with a clear error.
- [ ] **Sticky overrides:** each tick call `resolve_live_override()` (or
      `resolve_live_override_via` when the payload needs conversion); seed
      from Params on miss / until first match. Pass `changed` when
      costly rebuilds use `ConfigCache`.
- [ ] **Named axes / no opaque batch:** infer trailing (and any named
      leading) axes from input tensors; use `collapse_batch` /
      `full_rank_shape` only as a MatX layout helper over **named** axes;
      reject unexpected extra leading dims. Fat outs take
      `MaxTensorExtents` at `graph.op()` (see `MAX_TENSOR_EXTENTS.md`).
- [ ] **Output shapes** follow the contract when `[live]` params change
      leading dimensions (downstream re-infers).
- [ ] **CUDA stream:** `receive_cuda_stream()` on the primary data input;
      enqueue on that stream; retain inputs until the next tick if needed
      for stream-ordered lifetime.
- [ ] **MatX policy:** use MatX where it fits (element-wise, FFTs); custom
      CUDA where MatX is a poor match (document if benchmarked).
- [ ] **Errors:** `std::invalid_argument` with operator name and port in the
      message; unimplemented paths throw rather than returning garbage.

---

## Build & metadata

- [ ] **`operators/<group>/CMakeLists.txt`** — `add_holohub_operator()` /
      group gate (`OP_<name>`).
- [ ] **Per-operator `CMakeLists.txt`** — library target, `holoscan::core` +
      `holoscan::radar_common`, MatX if used, install rules.
- [ ] **`metadata.json`** — schema `urn:holohub:operator:v1`, SDK version,
      tags, language.
- [ ] **`operators/<group>/README.md`** — table row + implementation status
      (implemented vs staged).
- [ ] **Root `operators/CMakeLists.txt`** — enable the group/operator option.
- [ ] **SPDX header** on every new source, CMake, and metadata file.

---

## Tests

- [ ] **C++ unit tests** in `tests/cpp/<group>/test_<operator>.cpp`, mirroring
      the operator family tree (`fmcw/`, `detection/`, …), label `unit`.
- [ ] Use **`create_operator_test`** + `tests/cpp/utils/sdk_harness.hpp`
      (`single_map`, `TensorMapCapture`, `wire_leading_override` for sticky
      `[live]` tests).
- [ ] **Wire every output port** in harness/Application tests — operators
      with multiple outputs will not tick if a port is left unconnected.
- [ ] Cover **`TESTING.md`** high-level checks for this operator (unit tier
      first; integration/statistical when dependencies exist).
- [ ] **`[live]` overrides:** at least one mid-stream Exact match + a
      following miss tick (sticky and/or ephemeral).
      per functional override param (not just build-time `Arg`).
- [ ] **Float and double** instantiation agreement where applicable.
- [ ] **Error paths:** invalid args, unsupported modes, bad TensorMap shape.
- [ ] Register the test in **`tests/cpp/CMakeLists.txt`** with the
      family-relative source path (e.g. `radar_add_test(detection/test_<op> …)`).
      The three helpers pick the label: `radar_add_test` → `unit`,
      `radar_add_integration_test` → `integration` (cross-family chains),
      `radar_add_statistical_test` → `statistical`. The target name stays the
      flat basename, so `ctest -R test_<operator>` is unaffected by the path.
- [ ] Run **`ctest -L unit`** and **`pre-commit run --all-files`** before
      push.

---

## Performance

Measurement harness (not pass/fail thresholds). See
[`OPERATOR_DEVELOPMENT.md`](OPERATOR_DEVELOPMENT.md) → Performance
characterization. Skip only if the operator has no GPU/hot path yet; say so
in `OPEN_ITEMS.md`.

- [ ] **`performance/cpp/perf_<operator>.cpp`** — second (or Nth) consumer of
      `radar_perf_utils`; scenario matrix covering representative sizes and
      the operator's meaningful axes (variants, batch, sparse counts, …).
- [ ] Wire **every output port** (terminal sink on the primary sparse/dense
      result; null sinks on the rest) so the operator ticks.
- [ ] **Process warmup** — discarded graph session before the measured matrix
      (`performance/cpp/utils/perf_process_warmup.hpp`) so scenario #1
      `cold_ms` is not inflated by CUDA/Holoscan first-touch.
- [ ] Report **setup**, **cold** (first timed frame), and **steady**
      (median / p90) separately; annotate platform / provenance.
- [ ] **`[live]` overrides in the harness** — if the operator has any live
      params, every default/`--full` compute run (or an explicit override
      scenario matrix) must inline a mid-session override and report
      `override_ms` / `override_delta_ms`. Cover each functionally distinct
      live param at least once, and exercise any `[live]` fast paths (named
      slow/fast pairs if applicable); do not claim a fast path while timing
      the slow one. See `OPERATOR_DEVELOPMENT.md` → Performance
      characterization → Required coverage.
- [ ] **CMake** — `add_executable(perf_<operator> …)` linked to the operator
      lib; optional `add_test(… LABELS performance)` `*_smoke`
      (`--sizes xs --warmup 1 --frames 2 --quiet --output …`) in
      **`performance/cpp/CMakeLists.txt`**.
- [ ] Smoke passes: `ctest --test-dir build -R perf_<operator>_smoke` (or via
      `ctest -L performance`); skim cold vs steady for per-tick alloc / D2H /
      host-scan red flags before calling the operator performance-ready.

Exemplar: `performance/cpp/perf_cfar_detector.cpp`.

---

## Land the PR

- [ ] Update **`STATUS.md`** — move claim to done or refresh state.
- [ ] Add any new **team open items** to `operators/<group>/<operator>/OPEN_ITEMS.md`
      (create it if needed) **and** index them in
      [`docs/planning/TEAM_BLOCKED.md`](TEAM_BLOCKED.md).
- [ ] PR description notes **contract changes** explicitly if `OPERATOR_API.md`
      changed.
- [ ] Python bindings — **deferred** module-wide unless this PR is explicitly
      part of a bindings wave (`STATUS.md` settled decision).

---

## Quick reference — shared helpers

| Need | Header |
| --- | --- |
| Override ports / Exact resolve | `radar_common/sdk/override_param.hpp` |
| Warn-once artifact cache | `radar_common/sdk/config_cache.hpp` |
| Custom parameter Arg setters | `radar_common/sdk/arg_setters.hpp` |
| Batching / dtype / device checks | `radar_common/tensor/tensor_utils.hpp` |
| Device tensor allocation | `radar_common/tensor/dlpack_buffer.hpp` |
| MatX views (`.cu` / stage only) | `radar_common/tensor/matx_utils.hpp` |
| Test TensorMap / overrides | `tests/cpp/utils/sdk_harness.hpp` |
| Host ↔ device test tensors | `tests/cpp/utils/tensor_fixtures.hpp` |
