# Project Status & Work Coordination

The coordination point for in-flight work (see [`AGENTS.md`](../../AGENTS.md) → Team workflow). Before starting non-trivial work, check the claims table and add a row; update it in the PR that lands the work.

## Where the project stands

- **2026-07-01** — Planning docs (`OPERATOR_API.md`, `WORKFLOWS.md`, `TESTING.md`) written; FMCW API lockdown specified. No operator code yet beyond the `radar_op` template placeholder.

## Settled decisions

Decisions made outside the planning docs, recorded so they aren't re-litigated:

| Decision | Choice | Date |
| --- | --- | --- |
| Operator directory layout | Per functional group under `operators/`: `fmcw/` (signal processing + MIMO marshaling), `detection/`, `clustering/`, `angle_estimation/`, plus a shared `common/` library for the six supporting types | 2026-07-06 |
| Python bindings for new operators | Deferred to a later phase — operators land C++-first with tests; bindings for the full operator set follow as a dedicated effort | 2026-07-06 |
| `radar_op` template placeholder | Removed on `deustice/sigproc-ops` once the first real operators landed with tests; app, tests, and package rewired to the FMCW operators | 2026-07-10 |
| MatX + CUDA build wiring | Wire into the build from the start (not deferred to first kernel) | 2026-07-01 |
| GPU as a hard dependency | A CUDA-capable NVIDIA GPU is required for operators, apps, and the C++ test suite; no CPU fallbacks, GPU-optional builds, or skip-if-no-GPU test guards (the `RADAR_TEST_SKIP_IF_NO_GPU` machinery was removed). Host-only code exists only where host-only by design (simulator, window generation) | 2026-07-10 |
| Uniform CUDA-accessible input requirement | All operators reject pageable-host (`kDLCPU`) input with a clear error; the range and fused operators' H2D staging was removed (originally kept because the range processor is potentially first in the chain — ingest operators upload instead). Amends SIGPROC_PLAN decision 7 and the OPERATOR_API memory-space convention | 2026-07-10 |

## In-flight work

| Work item | Owner | Branch | State |
| --- | --- | --- | --- |
| Fill out `AGENTS.md` + this status file | deustice | `deustice/planning` | in review |
| Signal-processing operators (`common/` + range, clutter, Doppler, fused) with MatX/CUDA wiring, simulator + test harness | deustice | `deustice/sigproc-ops` | **merged** (PR #8) |
| Detection operators (`CFARDetectorOp` + FeatureAssembly + DetectionReduction) | dpc | `dpc/detection-full` | **merged** (PR #10) |
| Wire `CFARDetectorOp` into `applications/fmcw_siso` | dpc | `dpc/fmcw-siso-cfar` | in review |
| Clustering operator (`operators/clustering/` + `ClusterDBSCANOp`) | deustice | `deustice/clustering` | **checklist-complete** — GPU DBSCAN (brute-force neighbor graph + iterative CC), `epsilon` (scalar/vector) + `min_points` + `max_clusters` sticky live params, `labels` + `sizes`, float/double, unit + `perf_cluster_dbscan`; staged perf follow-ups in OPEN_ITEMS |
| Wire `ClusterDBSCANOp` into `applications/fmcw_siso` | deustice | `deustice/clustering` | in progress — full chain (CFAR, FeatureAssembly, DBSCAN, DetectionReduction, ClusterReport) with a live Holoviz GUI: the RD map draws every detection as CROSSES colored by DBSCAN cluster, and two rolling panels plot each cluster's reduction representative vs time (`headless`/`plots` YAML opts). Sustainability pass: the cfar_detector / cluster_dbscan / reduction blocks flow natively via `from_config()` (dropped ~220 lines of app-side parsing) and the reusable report + viz operators moved into a header-only `applications/common/radar_app` library, slimming the app from ~1460 to ~510 lines. Verified via the debug-sink harness. |
| `SnapshotGatherOp` (sparse angle bridge) | dpc | `dpc/integration` | **checklist-complete** — GPU RD-neighborhood gather for the sparse angle path; `snapshot_window` sticky live param; static edge modes; float/double; unit tests (edge modes, preserved batch, error paths, live override) + `perf_snapshot_gather`; HSDK 4.3 `is_valid` bool transport caveat in OPEN_ITEMS §1 (CFAR-aligned) |
| `BeamscanEstimatorOp` + `ArrayGeometry` / `AngleConvention` descriptors | dpc | `dpc/integration` | **checklist-complete** — conventional sparse-path Bartlett estimator; descriptors land in `operators/common/`; sticky `[live]` geometry/grid/taper/peak_interp; float/double; unit tests + `perf_beamscan_estimator`; steering weights built on GPU (shared `radar_common_steering`); `custom` preset + bool `is_valid` transport in OPEN_ITEMS |
| Wire sparse SIMO path into `applications/fmcw_simo` | dpc | `dpc/integration` | **done** — sparse angle-then-cluster graph (`SimoAdcSourceOp`, CFAR NCI over RX, SnapshotGather → Beamscan → 4D FeatureAssembly → DBSCAN → reduction); 8×8 URA, three targets with 3D velocity; Holoviz RD (ch-0 slice) + range/velocity rolling plots + boresight az/el panel; `RdChannelSliceOp`, `BoresightViewOp`, and 4-column `ClusterReportOp` in `radar_app` |
| `MIMODeinterleaveOp` + `TXSchedule` descriptor | dpc | `dpc/integration` | **checklist-complete** — TDM slow-time gather into explicit `(M_tx, N_rx, L, samples)` axis; `TXSchedule` lands in `operators/common/`; round-robin + custom `tx_order`; rejects non-uniform schedules; complex/real IF dtype preserved; float/double; unit tests + `perf_mimo_deinterleave` |
| `VirtualArrayAssemblyOp` | dpc | `dpc/integration` | **checklist-complete** — flatten `(M_tx, N_rx)` RD maps into virtual channel axis; emit runtime `array_geometry`; `overlap_policy` {dedup, average, keep_all}; `[live]` `wavelength`; float/double; unit tests + `perf_virtual_array_assembly` |
| `MIMOCoherentCompensationOp` (TDM + DDM) | dpc | `dpc/integration` | **checklist-complete** — TDM per-Doppler-bin inter-TX phase correction for separated `(M_tx, N_rx)` RD maps, and DDM Doppler-band demux into an explicit TX axis (standard linear-phase: `doppler_offsets` or linear-phase `phase_codes`; pure Doppler-bin gather, waveform-independent). `[live]` `waveform`; float/double; unit tests (48) + `perf_mimo_coherent_compensation` (TDM + DDM variants). Imposed layout / underspecified contract pins in operator `OPEN_ITEMS.md` §A; arbitrary/non-linear DDM `phase_codes` deferred (§B.1) |
| `applications/fmcw_mimo` (MIMO TDM + DDM demo) | dpc | `dpc/integration` | **done** — one binary; `fmcw_mimo.yaml` (TDM) and `fmcw_mimo_ddm.yaml` (DDM) drive `compose()` via `multiplexing_scheme`; interleaved / coded ADC simulators; Holoviz GUI |
| `PhaseShiftBeamformerOp` (dense angle path) | dpc | `dpc/integration` | **done** — conventional (Bartlett) dense beamformer: `rd_maps (…, num_channels, D, R)` → `rda_cube (…, num_az, num_el, D, R)` for CFAR's dense path; one broadcast complex GEMM per tick; reuses `ArrayGeometry` / `AngleConvention` / `array_taper` steering machinery from `BeamscanEstimatorOp`; sticky `[live]` geometry/scan-grid/taper, static `angle_convention`; float/double; 17 unit tests + `perf_phase_shift_beamformer`; steering weights built on GPU via shared `build_steering_weights_device` (`radar_common_steering`), removing the serial host trig loop from the `[live]` override path (host build alone ≈0.97 ms at the `lg` override grid, larger than the whole post-move override delta ≈0.70 ms) |
| `applications/fmcw_dense_simo` (dense angle demo) | dpc | `dpc/integration` | **done** — SIMO dense-path demo of `PhaseShiftBeamformerOp`: Range→Clutter→Doppler→beamformer→CFAR-on-cube (`nci_axes=[]`)→FeatureAssembly→DBSCAN→reduction; Taylor taper + tuned SNR/epsilon give 3 targets → 3 clusters. Optional Holoviz (`radar_app/viz`: RD map + cluster crosses, rolling range/velocity plots, boresight az/el centroids) or headless. Kept separate from `fmcw_simo`: dense path diverges in the detect+angle middle and reporting (4-col `detection_idx` / rank-4 score vs. sparse 2-col report ops). App-local ops: `AnglePeakPickOp` (score-peak (az,el) beam per RD cell), `CubeAoaLookupOp` (az/el-bin → `aoa_estimates`; OPEN_ITEMS §6), `TrailingIdxColumnsOp`, `DenseClusterReportOp`, `EntityDrainOp` |
| `MVDREstimatorOp` (adaptive sparse-path DOA) | dpc | `dpc/integration` | **NOT checklist-complete — one blocked param** — Capon core (per-item covariance + `is_valid` mask, diagonal loading `auto`/fixed δ, batched Hermitian inverse, Capon spectrum, dominant DOA) **plus** the multi-source surface: `max_sources` **[live]** + `doa_multi`/`source_count` (pass 0 = Capon argmax; further peaks = strict 2-D local maxima) and `eigenvalues` (raw-covariance spectrum via batched `matx::eig`). Reuses `build_steering_weights_device` + `ArrayGeometry`/`AngleConvention`; sticky `[live]` geometry/scan-grid/loading/`max_sources`; float/double; the 4 estimator outputs are `kNone`; unit tests + perf. **Only `spatial_smoothing` remains — BLOCKED on a team decision**; see `mvdr_estimator_op/OPEN_ITEMS.md` |

### Landed on `dpc/detection`

- **`CFARDetectorOp` CA slice** — complex/real power paths, border-correct CA
  stencil, `[live]` overrides with sticky semantics, `nci_axes` non-coherent
  integration (Gamma(`K`) threshold via host incomplete-beta LUT), batch-widened
  `detection_idx`, unit + Monte Carlo statistical tests. Team open items in
  [`CFARDetectorOp open items`](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md).
- **`CFARDetectorOp` GO slice** — 2-D GOGO greatest-of stencil (range GO ×
  Doppler GO), host Monte Carlo `alpha` LUT (CA-seeded bracket), full GO
  contract test matrix + `GoMonteCarloPfaK1` / `GoMonteCarloPfaNciK8`
  statistical tests.
- **`CFARDetectorOp` SO slice** — 2-D SOSO smallest-of stencil (shared GOGO
  partition), host Monte Carlo `alpha` LUT, full SO contract test matrix +
  `SoMonteCarloPfaK1` / `SoMonteCarloPfaNciK8` statistical tests.
- **`CFARDetectorOp` OS slice** — per-cell order-statistic stencil with
  `[live] os_rank` (0-based ascending), host Monte Carlo `alpha` LUT keyed by
  rank, OS-specific validation/live-override tests, full contract test matrix
  plus `OsMonteCarloPfaK1` / `OsMonteCarloPfaNciK8` statistical tests.

## Up next (unclaimed)

Roughly in dependency order — claim by adding a row above:

1. **C++ API skeletons** — supporting data structures + operator headers with `setup()` port/parameter declarations per `OPERATOR_API.md`, stub `compute()`, compilable; MatX/CUDA wired into CMake.
2. **Simulator + test harness** — the analytic point-target simulator and test scaffolding per `TESTING.md`.
3. **Operator implementations** — kernels against the locked contracts, per functional group.
4. **Operator implementations, remaining groups** — detection & clustering, angle estimation, and MIMO marshaling kernels against the locked contracts (`operators/common/` supporting types `ArrayGeometry`, `AngleConvention`, `DDMCodeDescriptor`, `TXSchedule` land with their first consumer). The signal-processing group, the simulator + test harness, and the MatX/CUDA build wiring landed on `deustice/sigproc-ops`.
5. **Python bindings** — pybind11 bindings for the implemented operators (restores the Python example app). The YAML converters for the shared descriptor types landed on `deustice/sigproc-ops` (`radar_common/sdk/yaml_converters.hpp`; `from_config()` works for C++ apps).

## Open items for team review

Contract, SDK, and signal-processing questions that need a team decision are
indexed in [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md). Per-operator detail stays in
each operator’s `OPEN_ITEMS.md` (for example,
[`CFARDetectorOp open items`](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md)).
New operators should follow [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md)
and [`OPERATOR_DEVELOPMENT.md`](OPERATOR_DEVELOPMENT.md).
