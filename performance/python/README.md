# Python performance harness

Measurement-only Holoscan operator benchmarks mirroring
[`performance/cpp/`](../cpp/). These scripts compose tiny
source → operator → sink graphs, time end-to-end frame latency with a device
sync at the terminal sink, and print a unified text report (optional JSON via
`--output`).

Most C++ performance tools have a Python counterpart. The current exceptions
are multi-source expansion, rank-one residual screening, and subarray
selection. Shared scenario matrices are still hand-mirrored where both forms
exist (see [`OPEN_ITEMS.md`](../../OPEN_ITEMS.md)).

The current `main-5x` source image does not export the SDK Python binding
target, so these scripts are retained but are not built by the default
container. See [`SDK_DEVELOPMENT.md`](../../SDK_DEVELOPMENT.md).

## Prerequisites

- CUDA-capable GPU
- Built holoscan-radar Python extensions (same as pytest)
- Holoscan SDK importable (`pip install holoscan` or dev container)
- CuPy (pulled in by holoscan in typical dev environments)

## Build path

The harness resolves `holoscan.radar` from the CMake build tree. Point
`HOLOSCAN_RADAR_BUILD_DIR` at your build directory if it is not `./build`:

```bash
export HOLOSCAN_RADAR_BUILD_DIR=/path/to/holoscan-radar/build
```

If unset, scripts default to `../../build` relative to this directory.

## Scripts

| Script | Operator | Default matrix notes |
| --- | --- | --- |
| `perf_cfar_detector.py` | `CFARDetectorOp` | Curated variants×sizes; mid/large MC only with `--full`; includes RD and rank-3/4 full N-D peak-mode rows |
| `perf_clutter_filter.py` | `ClutterFilterOp` | methods × sizes × channel batch |
| `perf_fmcw_range_processor.py` | `FMCWRangeProcessorOp` | windows × sizes × channels |
| `perf_fmcw_doppler_processor.py` | `FMCWDopplerProcessorOp` | windows × sizes × channels |
| `perf_fmcw_range_doppler_processor.py` | `FMCWRangeDopplerProcessorOp` | clutter modes × sizes × channels |
| `perf_fmcw_split_vs_fused.py` | split chain vs fused RD | noclt+msub pairs × sizes (e2e compare) |
| `perf_mimo_deinterleave.py` | `MIMODeinterleaveOp` | sizes + batch (no live override) |
| `perf_tdm_dense_compensation.py` | `TDMDenseCompensationOp` | TDM sizes × batch |
| `perf_ddm_dense_demultiplex.py` | `DDMDenseDemultiplexOp` | Linear/nonlinear DDM × sizes |
| `perf_ddm_power_fold.py` | `DDMPowerFoldOp` | Linear folded-power sizes |
| `perf_ddm_sparse_demultiplex.py` | `DDMSparseDemultiplexOp` | Detection-count × sizes |
| `perf_virtual_array_assembly.py` | `VirtualArrayAssemblyOp` | policies × sizes × batch |
| `perf_phase_shift_beamformer.py` | `PhaseShiftBeamformerOp` | size ladder (az override) |
| `perf_beamscan_estimator.py` | `BeamscanEstimatorOp` | size ladder (az override) |
| `perf_mvdr_estimator.py` | `MVDREstimatorOp` | size ladder (az override) |
| `perf_snapshot_gather.py` | `SnapshotGatherOp` | windows × sizes |
| `perf_feature_assembly.py` | `FeatureAssemblyOp` | packs × N × sizes (no live override) |
| `perf_cluster_dbscan.py` | `ClusterDBSCANOp` | (N,F) ladder × override kind |
| `perf_detection_reduction.py` | `DetectionReductionOp` | policies × sizes × override kind |

## Quick smoke

From the repo root (after configuring and building):

```bash
export HOLOSCAN_RADAR_BUILD_DIR="${HOLOSCAN_RADAR_BUILD_DIR:-./build}"

for s in performance/python/perf_*.py; do
  python "$s" --size xs --repeats 1 --warmup 1 --measured 2 --quiet
done
```

Default CLI (no flags): `--warmup 5`, `--measured 20`, `--repeats 1`,
**`--full` off** (C++ default matrix). Pass `--full` for the larger matrix.

- **`--size` / `--sizes`**: size-sweep subset (operator-specific default variant).
- **`perf_cfar_detector`**: mid/large GO/SO/OS footprints are Monte Carlo–heavy;
  only in `--full`. Curated pfa↔lut calibration pairs are included. Default and
  `--full` also includes peak-mode NMS characterization rows
  (`_peak_radius1` / `_peak_radius2` scenario ids).

## CLI

| Flag | Default | Meaning |
| --- | --- | --- |
| `--full` / `--no-full` | off | Full scenario matrix (match C++) |
| `--size` | — | Single ladder id (size sweep; overrides `--full`) |
| `--sizes` | — | Comma-separated ids or `all` (size sweep) |
| `--warmup` | `5` | Frames excluded from steady-state stats |
| `--measured` | `20` | Frames measured per scenario |
| `--repeats` | `1` | Independent graph sessions pooled |
| `--output PATH` | — | Write JSON report |
| `--text PATH` | — | Write text report to file |
| `--quiet` | off | Suppress stdout summary |

## Layout

| Path | Role |
| --- | --- |
| `utils/perf_app.py` | `TimingCollector` (setup + per-frame latency) |
| `utils/perf_source.py` | Timestamping source + live-override source |
| `utils/perf_sink.py` | Terminal sink (CuPy device sync) + Entity null drain helper |
| `utils/tensor_factory.py` | Deterministic CuPy tensors (incl. zero-size-safe upload) |
| `utils/perf_runner.py` | Run plan + sample bucketing (cold/steady/override) |
| `utils/perf_report.py` | Report rows + text/JSON printer |
| `utils/perf_driver.py` | Shared CLI (`run_perf_matrix_main`) |
| `perf_*.py` | Operator-specific scenario wiring |

## Notes

- No timing assertions — use for characterization and regression tracking, not CI pass/fail gates (see C++ `performance/cpp/CMakeLists.txt` smoke tests for the binary harness).
- Latencies are valid only on the machine/GPU they were captured on.
