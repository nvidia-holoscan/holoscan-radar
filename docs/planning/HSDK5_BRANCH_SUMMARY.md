# HSDK 5 Branch Summary (`dpc/hs5.0-fixes`)

[EA HSDK5 ONLY] Executive summary of this branch’s plan, what we hit on
Holoscan 5 EA (`v5.0.0.1`), how we worked around it, and what is still
blocked on EA / team decisions. Detail, per-operator progress, and
repro recipes live in [`HSDK5_PORT.md`](HSDK5_PORT.md). Coordination in
[`STATUS.md`](STATUS.md). PR: <https://github.com/nvidia-holoscan/holoscan-radar/pull/32>

Remove this file with the other `[EA HSDK5 ONLY]` fences before promoting
the port to `main` for public HSDK 5.

---

## Plan (what this branch set out to do)

Migrate the radar module off Holoscan 4.x Application/`InputContext` onto
HSDK 5 Graph + `Operator<Params>` with typed Tensor ports — **no 4.x shim**.

| Phase | Intent | Branch outcome |
| --- | --- | --- |
| 0 — Foundation | EA image, CMake/MatX, Holoviz probe, Python forced off | Landed |
| 1 — Shared glue | `radar_common/sdk/` Tensor I/O + sticky helpers; Graph test harness | Landed |
| 2 — Operator wave | All 19 ops → Graph/`Operator<Params>` + unit tests (allow-list) | Landed (intentional live skips OK) |
| 3 — Perf wave | Graph `perf_*` drivers + cold/steady smokes | Landed (POD `override_ms` restored; enum/§9 lives still deferred) |
| 4 — Apps / Python | Demos + bindings when Holoviz / Python EA surface exists | **Partial:** C++ Graph demos + X11 RD viewer; Holoviz/Python still off |
| 5 — Close deferred | Re-probe EA gaps; ratify durable API deltas; unskip live Graph | **Open** — see blockers below |
| 6 — Cleanup | Drop allow-lists, `[EA HSDK5 ONLY]`, this file / `HSDK5_PORT.md` | Blocked on phase 5 |

**Also on this branch:** `MaxTensorExtents` ctor-capture for HSDK 5 Tensor
pool sizing and dropping opaque leading-batch (see
[`MAX_TENSOR_EXTENTS.md`](MAX_TENSOR_EXTENTS.md)).

**Out of scope:** non-SDK algorithm-contract changes, Pulse-Doppler / SAR /
CDM, new shared descriptor types without an `OPERATOR_API.md` change.

**Environment:** `dpc/hs5.0-fixes` against the temporary HSDK 5 EA
devcontainer (pin `v5.0.0.1`). Preview docs are the SDK source of truth —
do not infer Graph behavior from 4.x.

---

## What landed (high level)

- **Operators:** all 19 FMCW / detection / clustering / angle / MIMO
  marshaling ops on Graph; CFAR GO/SO/OS/alpha_lut + statistical Graph
  suites; allow-lists for ops and perf.
- **Perf:** shared `perf_graph.hpp`; all drivers cold/steady via
  `ctest -L performance`; POD `override_ms` on CFAR, DBSCAN, MUSIC /
  Root-MUSIC / ESPRIT, MVDR, SnapshotGather, and VirtualArrayAssembly.
- **Pool sizing:** `MaxTensorExtents` through fat ops, Graph units, apps,
  and Python binding constructors (no empty extents).
- **Demos:** `fmcw_siso`, `fmcw_simo`, `fmcw_dense_simo`, `fmcw_mimo`
  (TDM/DDM), `fmcw_aoa_estimators` under `HOLOSCAN_RADAR_HAS_RD_LIVE`.
- **Viz interim:** Holoviz-free X11 RD viewer with Holoviz-parity overlays
  (axes, detections/clusters, legend, rolling plots, boresight, estimator
  name).
- **AoA without `set_dynamic_flows`:** `EstimatorRouterOp` + `AoaMergeOp`
  (`OnAnyInput`) replace Application dynamic flows.

---

## Issues encountered and how we overcame them

Grouped by theme. Section numbers match [`HSDK5_PORT.md`](HSDK5_PORT.md)
**Open — SDK / EA**.

### Graph / partition / binding

| Issue | What we did |
| --- | --- |
| Mid-stream sticky `[live]` Graph (scalar/enum beside device Tensor) fails compile or runtime (`PARTITION_MANUAL_AUTHORING_REQUIRED`, `RUNTIME_PORT_BINDING_UNAVAILABLE`, `RUNTIME_EXECUTION_LANE_PLAN_INVALID`, …) | Gate typed live Inputs in `contract()` with `expect_*`; keep remaining scalar/enum Graph skips and the minimal repro in `tests/cpp/utils/test_live_override_repro.cpp` (`ctest -L ea_repro`). FMCW schema-backed window/waveform Graph tests run (§1 / §9). |
| Device Tensor outs need explicit pool bind; foreign-handle / zero-copy republish rejected | `bind_*_device_pools` in harness; apps/perf `allocate_tensor` / loan fill each tick; `PerfPublishMode`; accept VAA lg/xl vs 4.3 gap as publish-path artifact (§3) |
| Raw `DLManagedTensor*` imports are read-only | Writable path via `DLManagedTensorContext` + owned device memory (`dlpack_buffer.hpp`) (§4) |
| Operator with **no** wired port breaks auto-partition (`PARTITION_MANUAL_AUTHORING_REQUIRED`, unnamed vertex) | Always drain fan-outs (e.g. `cfar_power` → viewer **and** `NullSinkOp`); never leave a constructed sink unwired (§6) |
| Registered-but-unwired ports vs wired-not-in-contract | `contract()` matches exactly the wired set (X11 viewer overlays, report optional ports) |
| Optional Tensor Inputs hard to wire on Graph (e.g. FeatureAssembly bins, Beamscan `is_valid`) | Omit from contract for unconnected apps; stage/kernel coverage; app-local `PhysicalRangeVelocity*PointsOp` stand-ins for demos |
| No Graph equivalent of Application `set_dynamic_flows` (AoA estimator rotate) | `EstimatorRouterOp` (one of N snapshot outs) + `AoaMergeOp` (`OnAnyInput`); `estimator_idx` Tensor for UI label (scalar mid-stream still EA-open) |
| Multi-in `OnAligned`+`Exact` needs co-timed sources | Dual/triple Tensor sources on one OnClock tick in `sdk_harness.hpp` |
| Harness “hang” on reject paths | Poll `session.diagnostics()` — capture CV alone never wakes |
| Perf `run_async` has no `CountCondition`; fixed wall budgets abort xl | Stall-based stop (`kPerfStall`); optional `--stop-after-s` |

### Contract / API shape (durable HSDK 5)

| Issue | What we did |
| --- | --- |
| In/out port names share one namespace | Rename physical outs (`range_profiles_out`, `adc_cube_out`, `rd_maps_out`, …); note in `OPERATOR_API.md` (§2) |
| Live ports cannot carry `std::vector` / owning descriptors | POD ChannelPod for fixed shapes (e.g. SnapshotGather window); generated descriptor-Tensor schemas for FMCW windows/waveform (interim pending ratification); Params-only for other unbounded vectors / owning descriptors (§9) |
| `Output<ArrayGeometry>` not a `PublishedPayload` | Pack geometry as float64 Tensor `(N+1, 3)` (`array_geometry_tensor.hpp`) on SubarraySelection / VAA (§10) |
| Hardcoded fat Tensor pool ceilings OOM or under-size | `MaxTensorExtents` at `graph.op()`; fail-closed extent checks where required |

### Product surface on EA

| Issue | What we did |
| --- | --- |
| Holoviz absent; Python operator surface Tensor-only | Force Python off; X11 RD viewer + Graph demos under `HOLOSCAN_RADAR_HAS_RD_LIVE` / `HOLOSCAN_RADAR_HAS_X11` |
| Legend / status flicker from DBSCAN label ranking and digit width | Range-rank legend order; fixed-width / right-padded status fields |
| EA pin / “is there EA2?” | Stayed on `v5.0.0.1`; no newer Graph EA tag found; `pb6.5` is a different (Application) track |

### Process

| Issue | What we did |
| --- | --- |
| Lint CI: cpplint `indent_namespace` on ctor init-lists; `tested_versions: "5.0.0-ea"` fails schema | Body `extents_ = extents`; schema-legal `"5.0.0"` |

---

## Blocked on EA gaps (and related durable decisions)

Until these close, phase 5 / cleanup cannot finish. Detail and next steps:
[`HSDK5_PORT.md`](HSDK5_PORT.md) §§1–4, 8–10.

### Need newer EA / GA or Holoscan maintainer answer

| ID | Gap | Impact if still broken |
| --- | --- | --- |
| §1 | Mid-stream sticky Graph `[live]` (device Tensor path + scalar/enum override) | POD perf timing works with an exact contract and one explicit partition; one-shot sticky unit tests, enums, and general wired/unwired production contracts remain open |
| §6 (Holoviz / Python) | Holoviz package + usable Python / pybind11 operator surface | Holoviz demos and pytest remain forced off; X11 is interim only |
| §6 (partition) | Auto-partition rejects graphs with an unreachable / unwired operator | Keep always-drain pattern; want named diagnostics or tolerance |
| §3 (re-probe) | Foreign-handle / loan-reuse Tensor publish | Keep per-tick pool fill; 4.x→5.x e2e medians include publish cost |

### Need team contract vote (durable HSDK 5 — interim already in code)

| ID | Question | Interim today |
| --- | --- | --- |
| §2 | Keep module-wide `*_out` unique names as locked contract? | Physical `*_out` names + doc notes |
| §9 | Params-only vs schema/Tensor/POD for vector / descriptor `[live]` overrides? | Generated descriptor-Tensor schemas for FMCW windows/waveform pending ratification; Params-only for other non-POD / unbounded values; POD where fixed-shape |
| §10 | Restore typed `Output<ArrayGeometry>` when SDK admits it, or lock Tensor pack? | float64 Tensor pack |
| §3 / §4 | Confirm GA keeps explicit bind + writable DLPack rules | Documented patterns in harness / `radar_common` |

### Not EA — do not confuse with the above

Radar algorithm / estimator contract questions stay in
[`TEAM_BLOCKED.md`](TEAM_BLOCKED.md) and per-op `OPEN_ITEMS.md` (e.g.
covariance multi-snapshot form, source-count policy, ULA selection on
non-uniform virtual arrays). Phase 5 does **not** silently answer those.

---

## How to verify on this branch

```bash
# EA image — see .devcontainer/README.md
cmake -S . -B build \
  -DHOLOSCAN_RADAR_BUILD_TESTING=ON \
  -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON
cmake --build build -j"$(nproc)"

ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L performance --output-on-failure
ctest --test-dir build -L ea_repro --output-on-failure   # sticky live Graph repro

# Demos (smoke YAMLs are log-only; default YAML needs DISPLAY/Xvfb for X11)
./build/applications/fmcw_siso/fmcw_siso applications/fmcw_siso/fmcw_siso_smoke.yaml
```

---

## Pointers

| Doc | Role |
| --- | --- |
| [`HSDK5_PORT.md`](HSDK5_PORT.md) | Full playbook, open items, progress table, cleanup gate |
| [`SCHEMA_PAYLOADS.md`](SCHEMA_PAYLOADS.md) | Interim schema-payload ownership, wire shape, evolution, and verification rules |
| [`MAX_TENSOR_EXTENTS.md`](MAX_TENSOR_EXTENTS.md) | Pool-sizing design |
| [`OPERATOR_API.md`](OPERATOR_API.md) | Locked operator contracts (incl. EA physical renames) |
| [`STATUS.md`](STATUS.md) | Claim / land log |
| [`.devcontainer/README.md`](../../.devcontainer/README.md) | EA image build / run |
