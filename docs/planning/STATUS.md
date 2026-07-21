# Project Status & Work Coordination

The coordination point for in-flight work (see [`AGENTS.md`](../../AGENTS.md) → Team workflow). Before starting non-trivial work, check the claims table and add a row; update it in the PR that lands the work.

## Where the project stands

- **2026-07-01** — Planning docs (`OPERATOR_API.md`, `WORKFLOWS.md`, `TESTING.md`) written; FMCW API lockdown specified. No operator code yet beyond the `radar_op` template placeholder.

## Settled decisions

Decisions made outside the planning docs, recorded so they aren't re-litigated:

| Decision | Choice | Date |
| --- | --- | --- |
| Operator directory layout | Per functional group under `operators/`: `fmcw/` (signal processing + MIMO marshaling), `detection/`, `clustering/`, `angle_estimation/`, plus a shared `common/` library for the six supporting types | 2026-07-06 |
| Python bindings for new operators | Dedicated bindings wave on `dpc/python-bindings` — full operator set + radar_app viz + C++-parity tests/perf + Holoviz Python demo apps (was deferred 2026-07-06). Checklist-complete pending PR; bool-mask NumPy inspection remains HSDK-4.3 limited | 2026-07-19 |
| `radar_op` template placeholder | Removed on `deustice/sigproc-ops` once the first real operators landed with tests; app, tests, and package rewired to the FMCW operators | 2026-07-10 |
| MatX + CUDA build wiring | Wire into the build from the start (not deferred to first kernel) | 2026-07-01 |
| GPU as a hard dependency | A CUDA-capable NVIDIA GPU is required for operators, apps, and the C++ test suite; no CPU fallbacks, GPU-optional builds, or skip-if-no-GPU test guards (the `RADAR_TEST_SKIP_IF_NO_GPU` machinery was removed). Host-only code exists only where host-only by design (simulator, window generation) | 2026-07-10 |
| Uniform CUDA-accessible input requirement | All operators reject pageable-host (`kDLCPU`) input with a clear error; the range and fused operators' H2D staging was removed (originally kept because the range processor is potentially first in the chain — ingest operators upload instead). Amends SIGPROC_PLAN decision 7 and the OPERATOR_API memory-space convention | 2026-07-10 |
| S1 — spatial-smoothing input form | Forward / forward-backward sub-array averaging on a filled ULA/URA (1-D **and** 2-D), parametrized by a per-aperture-axis vector of sub-array **counts** (`spatial_smoothing`) + a `forward_backward` bool; implemented as a front-end transform reducing to the existing covariance path (shared helper `radar_common/dsp/spatial_smoothing`). Non-uniform / holey arrays are rejected → reduced upstream (S3) or via a custom upstream smoother. Two forks left to ratify: count-vs-length; holey→S3 vs in-operator partitions (TEAM_BLOCKED L13) | 2026-07-20 |
| S3 — ULA/subarray selection | `SubarraySelectionOp` selects a uniform ULA/URA from a holey MIMO virtual array (auto = largest filled sub-block via holey-lattice detection + maximal-rectangle; or explicit indices) and emits a reduced `(snapshot, array_geometry)` pair; the channel gather reuses the S1 gather (`P=1`). Unblocks Root-MUSIC/ESPRIT to start and the holey-array path for spatial smoothing. Four forks left to ratify (TEAM_BLOCKED L14). Gridless consumers not yet built | 2026-07-20 |
| S2 — source-count provenance | `num_sources` for the subspace estimators (MUSIC / Root-MUSIC / ESPRIT): `int` (fixed order) or default `auto` = **EFT** (Exponential Fitting Test — robust at automotive `M≈N` snapshot counts, unlike MDL/AIC) on the **forward-backward-smoothed (S1)** covariance eigenvalues, clamped to `[1, max_sources]`; `source_count_pfa` sets the EFT rate; rank-deficient covariance → fall back to **1** + warn. **Rejected:** seeding from the CFAR/DBSCAN count (# RD detections, not the per-cell angular multiplicity `num_sources` denotes). **Deferred:** RMT/Kritchman–Nadler, AIC, upstream-seed port. Spectral-first; FBSS (S1) prerequisite for `auto`. Settled; EFT helper + operators are follow-on. Basis: Quinlan EFT 2007; TI arXiv:2501.07649 | 2026-07-21 |
| S4 — dense az/el-bin → angle features | Extend `FeatureAssemblyOp` (chosen over a standalone shared op): add `az_bin`/`el_bin` feature tokens + optional `az_angles`/`el_angles` grid inputs, resolving az/el from the preserved `(az_bin, el_bin)` `detection_idx` columns — symmetric with its existing range/velocity bin lookups. The dense path no longer needs the app-local `CubeAoaLookupOp` adapter (apps can migrate at leisure). Implemented + tested | 2026-07-21 |
| L2 — CFAR NCI `K > 1` α closed form | Adopt the **regularized incomplete-beta** relation as the pinned CA-CFAR calibration: `pfa = I_{1/(1 + α/N)}(N·K, K)`, inverted numerically for α per training-cell count `N` (host LUT), reducing to the locked `α = N·(pfa^(−1/N) − 1)` at `K = 1`. Ratifies the code-ahead-of-contract implementation (`cfar_alpha.cpp`); `OPERATOR_API.md` `pfa` / `nci_axes` bullets now state the relation. Validated by Monte Carlo (`CaMonteCarloPfaK1`, `CaMonteCarloPfaNciK8`). GO/SO/OS `K > 1` still calibrate by Monte Carlo (separate item, TEAM_BLOCKED L5). **Rejected:** reverting to the power-law placeholder `α = N·(pfa^(−1/(N·K)) − 1)` (misses target Pfa for `K > 1`) | 2026-07-21 |

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
| `MIMOCoherentCompensationOp` (TDM + DDM) | dpc | `dpc/integration` | **checklist-complete** — TDM per-Doppler-bin inter-TX phase correction; DDM Doppler-band demux (linear gather via `doppler_offsets` / linear `phase_codes`) **and** non-linear `phase_codes` conjugate-correlation demux (full Doppler out; `guard_bands` must be 0). `[live]` `waveform`; float/double; unit tests + `perf_mimo_coherent_compensation` (TDM / DDM / DDM-NL). Imposed layout pins in operator `OPEN_ITEMS.md` §A; §B.1 landed |
| `applications/fmcw_mimo` (MIMO TDM + DDM demo) | dpc | `dpc/integration` | **done** — one binary; `fmcw_mimo.yaml` (TDM), `fmcw_mimo_ddm.yaml` (linear DDM), `fmcw_mimo_ddm_phase_codes.yaml` (non-linear correlate); interleaved / coded ADC simulators; Holoviz GUI |
| `PhaseShiftBeamformerOp` (dense angle path) | dpc | `dpc/integration` | **done** — conventional (Bartlett) dense beamformer: `rd_maps (…, num_channels, D, R)` → `rda_cube (…, num_az, num_el, D, R)` for CFAR's dense path; one broadcast complex GEMM per tick; reuses `ArrayGeometry` / `AngleConvention` / `array_taper` steering machinery from `BeamscanEstimatorOp`; sticky `[live]` geometry/scan-grid/taper, static `angle_convention`; float/double; 17 unit tests + `perf_phase_shift_beamformer`; steering weights built on GPU via shared `build_steering_weights_device` (`radar_common_steering`), removing the serial host trig loop from the `[live]` override path (host build alone ≈0.97 ms at the `lg` override grid, larger than the whole post-move override delta ≈0.70 ms) |
| `applications/fmcw_dense_simo` (dense angle demo) | dpc | `dpc/integration` | **done** — SIMO dense-path demo of `PhaseShiftBeamformerOp`: Range→Clutter→Doppler→beamformer→CFAR-on-cube (`nci_axes=[]`)→FeatureAssembly→DBSCAN→reduction; Taylor taper + tuned SNR/epsilon give 3 targets → 3 clusters. Optional Holoviz (`radar_app/viz`: RD map + cluster crosses, rolling range/velocity plots, boresight az/el centroids) or headless. Kept separate from `fmcw_simo`: dense path diverges in the detect+angle middle and reporting (4-col `detection_idx` / rank-4 score vs. sparse 2-col report ops). App-local ops: `AnglePeakPickOp` (score-peak (az,el) beam per RD cell), `CubeAoaLookupOp` (az/el-bin → `aoa_estimates`; OPEN_ITEMS §6), `TrailingIdxColumnsOp`, `DenseClusterReportOp`, `EntityDrainOp` |
| `MVDREstimatorOp` (adaptive sparse-path DOA) | dpc | `dpc/integration` | **NOT checklist-complete — one blocked param** — Capon core (per-item covariance + `is_valid` mask, diagonal loading `auto`/fixed δ, batched Hermitian inverse, Capon spectrum, dominant DOA) **plus** the multi-source surface: `max_sources` **[live]** + `doa_multi`/`source_count` (pass 0 = Capon argmax; further peaks = strict 2-D local maxima) and `eigenvalues` (raw-covariance spectrum via batched `matx::eig`). Reuses `build_steering_weights_device` + `ArrayGeometry`/`AngleConvention`; sticky `[live]` geometry/scan-grid/loading/`max_sources`; float/double; the 4 estimator outputs are `kNone`; unit tests + perf. **Only `spatial_smoothing` remains — BLOCKED on a team decision**; see `mvdr_estimator_op/OPEN_ITEMS.md` |
| Python bindings wave (all landed ops + radar_app + tests/perf + demo apps) | dpc | `dpc/python-bindings` | **checklist-complete (pending PR)** — all 15 ops + descriptors + `radar_app` viz (F32/F64; bare `*Op` → F32); C++-parity pytest (bool `detection_mask`/`is_valid` drained via `NullSinkOp` on HSDK 4.3 — cannot inspect as NumPy until HSDK 5 / `uint8` contract); all 15 `performance/python/perf_*.py` drivers; Holoviz Python demos for SISO/SIMO/MIMO(TDM+DDM)/dense SIMO. CFAR Python covers CA leftovers + GO/SO/OS matrix/unique cases + `alpha_lut` + Monte Carlo (`pytest -m statistical`). **Accepted gap:** GO/SO/OS `detail::` α calibration white-box tests stay C++-only ([CFAR OPEN_ITEMS §9](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md)) |
| Fused vs split RD e2e perf harness | dpc | `dpc/python-bindings` | **landed (no keep/kill verdict)** — `perf_fmcw_split_vs_fused` (C++ + Python) times end-to-end Holoscan graphs (split range→[clutter]→Doppler vs fused) and prints comparison table for [OPEN_ITEMS §1](../../operators/fmcw/fmcw_range_doppler_processor_op/OPEN_ITEMS.md); team decides keep/retire from the table |
| DDM non-linear `phase_codes` (correlate demux) | dpc | `dpc/python-bindings` | **landed** — C++ correlate path + unit/perf; demo YAML `fmcw_mimo_ddm_phase_codes.yaml` (`generate_phase_codes: zadoff_chu` low-crosstalk NL codes; `sin` retained for crosstalk stress); Python bindings + pytest + perf `ddm_nl` + Python demo/smoke |

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
5. **Python bindings** — landed on `dpc/python-bindings` (see in-flight row). Remaining: land the PR; bool-mask NumPy inspection awaits HSDK 5 / TEAM_BLOCKED L1.

## Open items for team review

Contract, SDK, and signal-processing questions that need a team decision are
indexed in [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md). Per-operator detail stays in
each operator’s `OPEN_ITEMS.md` (for example,
[`CFARDetectorOp open items`](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md)).
New operators should follow [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md)
and [`OPERATOR_DEVELOPMENT.md`](OPERATOR_DEVELOPMENT.md).
