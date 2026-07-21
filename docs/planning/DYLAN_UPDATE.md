# Progress update for Dylan

**Date:** 2026-07-19
**Author:** Dan (dpc)
**Scope:** Everything developed since you left — currently at tip of `dpc/python-bindings` (includes the full `dpc/integration` stack). ~42 commits ahead of `origin/main` (main is through detection-full, PR #10).
**Status:** Forward development **paused** pending your return. Most of the FMCW SISO→SIMO→MIMO→dense wave is checklist-complete; Python bindings checklist-complete pending PR; one MVDR param and the next subspace estimators wait on team decisions.

---

## 1. Where we are (one-paragraph)

Since you left we took the module from “detection on SISO” to a full sparse and dense angle-capable stack with MIMO TDM/DDM demos, adaptive DOA (MVDR almost done), a shared `radar_app` Holoviz/report library, and a complete Python bindings + pytest + perf + demo wave. Non-linear DDM `phase_codes` correlate demux is in on `MIMOCoherentCompensationOp`, and a fused-vs-split RD e2e harness is ready for a keep/kill discussion. Development is stopped here on purpose — next meaningful progress is either landing this work or resolving the TEAM_BLOCKED items below.

---

## 2. What landed (by capability)

Grouped by pipeline role, not by branch. All of this is on the current tip unless noted.

### Apps & Holoviz

| App | What it demonstrates |
| --- | --- |
| `fmcw_siso` | CFAR → FeatureAssembly → DBSCAN → DetectionReduction; RD map with cluster crosses; rolling range/velocity plots; `from_config()` operator blocks; shared `radar_app` |
| `fmcw_simo` | Sparse angle-then-cluster: SnapshotGather → Beamscan → 4D features → DBSCAN; 8×8 URA; boresight az/el panel |
| `fmcw_mimo` | TDM and linear DDM (and NL phase_codes) in one binary; coded ADC sims; Holoviz |
| `fmcw_dense_simo` | Dense path: PhaseShiftBeamformer → CFAR on RDA cube → cluster; separate from sparse SIMO on purpose |

`applications/common/radar_app` holds reusable report + viz (Detection/Cluster report, RdMapToImage, overlays, rolling plots, boresight, cadence meter). Python mirrors exist for all four demos.

### Operators & descriptors (checklist state)

| Item | State |
| --- | --- |
| `ClusterDBSCANOp` | Checklist-complete (GPU DBSCAN; live `epsilon` / `min_points` / `max_clusters`) |
| `SnapshotGatherOp` | Checklist-complete |
| `BeamscanEstimatorOp` + `ArrayGeometry` / `AngleConvention` | Checklist-complete |
| `MIMODeinterleaveOp` + `TXSchedule` | Checklist-complete (uniform TDM only; non-uniform rejected — L12) |
| `VirtualArrayAssemblyOp` | Checklist-complete |
| `MIMOCoherentCompensationOp` | Checklist-complete — TDM phase correction; DDM linear gather; **DDM-NL correlate** (full Doppler out) |
| `PhaseShiftBeamformerOp` | Done (dense Bartlett → RDA cube) |
| `MVDREstimatorOp` | Capon + multi-source surface + eigenvalues **done**; **`spatial_smoothing` not declared — blocked on S1** |

Shared GPU steering: `radar_common_steering` / `build_steering_weights_device` (Beamscan, PhaseShift, MVDR).

### Python bindings wave

- All 15 operators + descriptors + `radar_app` (F32/F64; bare `*Op` → F32)
- C++-parity pytest; all 15 `performance/python/perf_*.py` drivers
- Holoviz Python demos for SISO / SIMO / MIMO / dense SIMO
- **Accepted gaps:** bool masks not NumPy-inspectable on HSDK 4.3 (drain only — L1); GO/SO/OS α white-box tests stay C++-only

### Extra closed while you were out

- **DDM-NL correlate** — mode on existing compensation op, not a new op. Zadoff–Chu demo YAML; center-zero I/O; cuFFT tiled path; CFAR `threshold_scale` + Holoviz column-median display tuned so the ~−15 dB 2-TX ZC pedestal does not dominate the RD image. Tracking/clusters were already correct before the display fix.
- **Fused vs split RD harness** (`perf_fmcw_split_vs_fused` C++/Python) — measurement only; **no keep/kill verdict**.
- **`TEAM_BLOCKED.md`** — roll-up of stops-development vs needs-team-look (see §4).

---

## 3. Notable technical details (worth a skim)

**DDM-NL (`MIMOCoherentCompensationOp`):**
Non-linear `phase_codes` → conjugate correlation; `guard_bands == 0`; `num_chirps == num_doppler`; full Doppler axis out. Linear codes still gather. Imposed rules in operator `OPEN_ITEMS.md` §A.10. Perf: `ddm_nl` (~16 ms `xl` median on RTX PRO 6000 vs ~2 ms gather); remaining wall time mostly per-tick output alloc of large maps — annotated under module `operators/OPEN_ITEMS.md` §1 (**do not land ping-pong without revisit**).

**Try NL demo:**

```bash
./build/applications/fmcw_mimo/cpp/fmcw_mimo fmcw_mimo_ddm_phase_codes.yaml
```

**Dense path:** App-local `CubeAoaLookupOp` bridges az/el bins → `aoa_estimates` until S4 decides a shared design.

**Tests layout:** Operator tree mirrored under `tests/cpp/`; Python suites under `tests/python/`.

---

## 4. State of play — what is blocked vs what is next

### Done enough to land (review / PR)

- Full integration stack + bindings tip (however you want to slice PRs).
- No further implementation planned until you are back.

### Hard stops (TEAM_BLOCKED — “Stops development”)

| ID | Decision | What cannot proceed |
| --- | --- | --- |
| **S1** | Spatial-smoothing / multi-snapshot input form on (often non-uniform) virtual arrays | Finish MVDR checklist (`spatial_smoothing`) |
| **S2** | Source-count provenance (fixed / MDL·AIC / CFAR seed) | Start MUSIC / Root-MUSIC / ESPRIT |
| **S3** | ULA / subarray selection from holey MIMO virtual arrays | Root-MUSIC / ESPRIT on non-uniform geometry |
| **S4** | Dense az/el-bin → angle features (shared op vs extend FeatureAssembly) | Shared dense→clustering bridge |

Full text: [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md).

### Needs team look (interim already in code — does not stop landing)

L1–L12 in `TEAM_BLOCKED.md` (bool DLPack, CFAR NCI/α/`alpha_lut`/`detection_idx`, DBSCAN policies, `AngleConvention::custom`, non-uniform TDM rejection, etc.). Ratify or revise when you have bandwidth.

### Explicitly deferred (not asking you to decide to ship)

- Module-wide output buffer reuse / ping-pong (`operators/OPEN_ITEMS.md` §1)
- DDM unequal band widths / alternate guard placement (compensation OPEN_ITEMS §B.2–B.3)
- Fused operator keep vs retire — run harness, then team call (`fmcw_range_doppler_processor_op/OPEN_ITEMS.md` §1)

### Natural next development (after decisions)

1. Land this work on main (integration ± bindings PR strategy is your call).
2. Resolve **S1** → finish MVDR.
3. Resolve **S2/S3** → MUSIC / Root-MUSIC / ESPRIT wave.
4. Resolve **S4** when hardening dense→cluster.
5. Optional: fused keep/kill from harness numbers; ping-pong only if the team revisits §1.

---

## 5. Suggested first actions when you are back

1. Read this + skim [`STATUS.md`](STATUS.md) in-flight table and [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md).
2. Decide how to land: one stacked PR from current tip, or integration first then bindings.
3. Smoke what you care about (SISO Holoviz, MIMO TDM/DDM/NL, dense SIMO, Python import/pytest).
4. Short design sync on **S1–S4** before anyone starts the next angle estimators.

---

## 6. How to run unit and perf tests (C++ and Python)

Assumes repo root, a CUDA GPU, and Holoscan SDK discoverable (`pip install holoscan` or the holohub container). Prefer `ulimit -s 32768` (or the devcontainer default) to avoid Holoscan stack warnings/segfaults.

### One-time build

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
# Python extensions live under build/python/lib/holoscan/radar/
export HOLOSCAN_RADAR_BUILD_DIR="${HOLOSCAN_RADAR_BUILD_DIR:-./build}"
```

### Unit tests — C++

```bash
# All C++ unit tests
ctest --test-dir build --output-on-failure -L unit

# One binary / filter by name (examples)
ctest --test-dir build --output-on-failure -R MIMOCoherent
ctest --test-dir build --output-on-failure -R 'CFAR|MVDR|Beamscan|ClusterDBSCAN'
./build/tests/cpp/test_mimo_coherent_compensation --gtest_filter='*DdmNonLinear*'
```

Monte Carlo / long CFAR stats (separate label):

```bash
ctest --test-dir build --output-on-failure -L statistical
```

### Unit tests — Python

`conftest.py` prepends `$HOLOSCAN_RADAR_BUILD_DIR/python/lib/holoscan` onto `holoscan.__path__`. Do **not** put `build/python/lib` alone on `PYTHONPATH` (shadows the SDK package).

```bash
export HOLOSCAN_RADAR_BUILD_DIR="${HOLOSCAN_RADAR_BUILD_DIR:-./build}"

# Default suite (excludes pytest mark ``statistical`` — see pytest.ini)
pytest tests/python/ -v

# Or via CTest (same default suite)
ctest --test-dir build --output-on-failure -R '^pytest$'

# Import smoke only
pytest tests/python/test_import_smoke.py -v

# By area
pytest tests/python/fmcw/ -v
pytest tests/python/detection/ -v
pytest tests/python/angle_estimation/ -v
pytest tests/python/clustering/ -v

# CFAR Monte Carlo / statistical (opt-in; mirrors ctest -L statistical)
pytest tests/python/ -v -m statistical
ctest --test-dir build --output-on-failure -R pytest_statistical
```

Clean skip (exit 5) if Holoscan is not importable — expected outside a configured env.

### Perf — C++

Binaries under `build/performance/cpp/`. Default matrix is the “smoke/default” ladder; `--full` is the large matrix. Optional JSON/text under `perf_results/` (local; not committed).

```bash
# Build all perf tools (already included in a full build)
cmake --build build -j"$(nproc)" --target \
  perf_cfar_detector perf_clutter_filter \
  perf_fmcw_range_processor perf_fmcw_doppler_processor \
  perf_fmcw_range_doppler_processor perf_fmcw_split_vs_fused \
  perf_mimo_deinterleave perf_mimo_coherent_compensation \
  perf_virtual_array_assembly perf_snapshot_gather \
  perf_beamscan_estimator perf_mvdr_estimator \
  perf_phase_shift_beamformer perf_feature_assembly \
  perf_cluster_dbscan perf_detection_reduction

# Tiny CI-style smoke for every registered perf test
ctest --test-dir build --output-on-failure -L performance

# Run one operator (examples)
./build/performance/cpp/perf_mimo_coherent_compensation \
  --output perf_results/perf_mimo_coherent_compensation.json \
  --text   perf_results/perf_mimo_coherent_compensation.txt

./build/performance/cpp/perf_mimo_coherent_compensation --full \
  --output perf_results/perf_mimo_coherent_compensation.full.json \
  --text   perf_results/perf_mimo_coherent_compensation.full.txt

# Fused vs split keep/kill input
./build/performance/cpp/perf_fmcw_split_vs_fused \
  --output perf_results/perf_fmcw_split_vs_fused.json \
  --text   perf_results/perf_fmcw_split_vs_fused.txt
./build/performance/cpp/perf_fmcw_split_vs_fused --full \
  --output perf_results/perf_fmcw_split_vs_fused.full.json \
  --text   perf_results/perf_fmcw_split_vs_fused.full.txt

# Size / repeat knobs (common across tools)
./build/performance/cpp/perf_cfar_detector --sizes xs --warmup 1 --frames 2 --quiet
./build/performance/cpp/perf_beamscan_estimator --size lg
```

DDM-NL is a scenario inside `perf_mimo_coherent_compensation` (`ddm_nl`), not a separate binary.

### Perf — Python

Same matrices as C++ (hand-mirrored). See [`performance/python/README.md`](../../performance/python/README.md).

```bash
export HOLOSCAN_RADAR_BUILD_DIR="${HOLOSCAN_RADAR_BUILD_DIR:-./build}"

# Smoke every Python perf script (fast)
for s in performance/python/perf_*.py; do
  python "$s" --size xs --repeats 1 --warmup 1 --measured 2 --quiet
done

# One operator, default matrix
python performance/python/perf_mimo_coherent_compensation.py

# Full matrix + artifacts
python performance/python/perf_mimo_coherent_compensation.py --full \
  --output perf_results/perf_mimo_coherent_compensation.py.json

python performance/python/perf_fmcw_split_vs_fused.py --full \
  --output perf_results/perf_fmcw_split_vs_fused.py.json
```

Common flags: `--full` / `--no-full`, `--size` / `--sizes`, `--warmup`, `--measured`, `--repeats`, `--quiet`, `--output`.

### “Run everything” cheat sheet

```bash
export HOLOSCAN_RADAR_BUILD_DIR=./build
ulimit -s 32768

# Unit
ctest --test-dir build --output-on-failure -L unit
pytest tests/python/ -v
# optional long CFAR MC:
# ctest --test-dir build --output-on-failure -L statistical
# pytest tests/python/ -v -m statistical

# Perf smokes
ctest --test-dir build --output-on-failure -L performance
for s in performance/python/perf_*.py; do
  python "$s" --size xs --repeats 1 --warmup 1 --measured 2 --quiet
done
```

---

## 7. How to run the demo apps (C++ and Python)

Build the app targets once (or rely on a full build):

```bash
cmake --build build -j"$(nproc)" --target \
  fmcw_siso fmcw_simo fmcw_mimo fmcw_dense_simo
# Python demos need the operator + radar_app bindings from the same build.
export HOLOSCAN_RADAR_BUILD_DIR="${HOLOSCAN_RADAR_BUILD_DIR:-./build}"
ulimit -s 32768
```

**Config resolution (C++):** optional YAML path on the CLI. Bare names (e.g. `fmcw_mimo_ddm.yaml`) are tried relative to cwd, then next to the binary (where CMake stages copies). No arg → default YAML beside the binary.
**Config resolution (Python):** optional YAML path; default is the matching `applications/.../cpp/*.yaml`. Each app inserts `build/python/lib/holoscan` onto `holoscan.__path__` when `HOLOSCAN_RADAR_BUILD_DIR` is set.

**Display:** full demos use `visualize: true`. No display → warning and log-only (or set `visualize: false` / `headless: true`). `*_smoke.yaml` under `python/` are short headless-friendly configs for CI-style smoke.

### `fmcw_siso` — SISO detect → cluster

| Mode | YAML |
| --- | --- |
| Full Holoviz demo | `applications/fmcw_siso/cpp/fmcw_siso.yaml` |
| Python smoke (short / no GUI) | `applications/fmcw_siso/python/fmcw_siso_smoke.yaml` |

```bash
# C++
./build/applications/fmcw_siso/cpp/fmcw_siso
./build/applications/fmcw_siso/cpp/fmcw_siso fmcw_siso.yaml
./build/applications/fmcw_siso/cpp/fmcw_siso /path/to/custom.yaml

# Python
python applications/fmcw_siso/python/fmcw_siso.py
python applications/fmcw_siso/python/fmcw_siso.py \
  applications/fmcw_siso/cpp/fmcw_siso.yaml
python applications/fmcw_siso/python/fmcw_siso.py \
  applications/fmcw_siso/python/fmcw_siso_smoke.yaml
```

### `fmcw_simo` — sparse SIMO angle-then-cluster

| Mode | YAML |
| --- | --- |
| Full Holoviz demo | `applications/fmcw_simo/cpp/fmcw_simo.yaml` |
| Python smoke | `applications/fmcw_simo/python/fmcw_simo_smoke.yaml` |

```bash
# C++
./build/applications/fmcw_simo/cpp/fmcw_simo
./build/applications/fmcw_simo/cpp/fmcw_simo fmcw_simo.yaml

# Python
python applications/fmcw_simo/python/fmcw_simo.py
python applications/fmcw_simo/python/fmcw_simo.py \
  applications/fmcw_simo/cpp/fmcw_simo.yaml
python applications/fmcw_simo/python/fmcw_simo.py \
  applications/fmcw_simo/python/fmcw_simo_smoke.yaml
```

### `fmcw_mimo` — MIMO TDM / DDM / DDM-NL (one binary, YAML selects mode)

| Mode | YAML | Notes |
| --- | --- | --- |
| **TDM** (default) | `applications/fmcw_mimo/cpp/fmcw_mimo.yaml` | Deinterleave → range → clutter → Doppler → compensation |
| **DDM linear** | `applications/fmcw_mimo/cpp/fmcw_mimo_ddm.yaml` | Band-gather demux; `doppler_offsets` |
| **DDM non-linear** | `applications/fmcw_mimo/cpp/fmcw_mimo_ddm_phase_codes.yaml` | Correlate path; `generate_phase_codes: zadoff_chu` (or `sin` for crosstalk stress); `doppler_window: none`; `threshold_scale: 16` |
| Python smokes | `.../python/fmcw_mimo_smoke.yaml`, `fmcw_mimo_ddm_smoke.yaml`, `fmcw_mimo_ddm_phase_codes_smoke.yaml` | Short headless-oriented runs |

```bash
# C++ — TDM / DDM / DDM-NL
./build/applications/fmcw_mimo/cpp/fmcw_mimo
./build/applications/fmcw_mimo/cpp/fmcw_mimo fmcw_mimo_ddm.yaml
./build/applications/fmcw_mimo/cpp/fmcw_mimo fmcw_mimo_ddm_phase_codes.yaml

# Python — same YAMLs (full demos live under cpp/)
python applications/fmcw_mimo/python/fmcw_mimo.py
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/cpp/fmcw_mimo_ddm.yaml
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/cpp/fmcw_mimo_ddm_phase_codes.yaml

# Python smokes
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_smoke.yaml
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_ddm_smoke.yaml
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_ddm_phase_codes_smoke.yaml
```

### `fmcw_dense_simo` — dense RDA path (PhaseShiftBeamformer)

Sibling to sparse `fmcw_simo`; not a YAML mode of the same binary.

| Mode | YAML |
| --- | --- |
| Full Holoviz demo | `applications/fmcw_dense_simo/cpp/fmcw_dense_simo.yaml` |
| Python smoke | `applications/fmcw_dense_simo/python/fmcw_dense_simo_smoke.yaml` |

```bash
# C++
./build/applications/fmcw_dense_simo/cpp/fmcw_dense_simo
./build/applications/fmcw_dense_simo/cpp/fmcw_dense_simo fmcw_dense_simo.yaml

# Python
python applications/fmcw_dense_simo/python/fmcw_dense_simo.py
python applications/fmcw_dense_simo/python/fmcw_dense_simo.py \
  applications/fmcw_dense_simo/cpp/fmcw_dense_simo.yaml
python applications/fmcw_dense_simo/python/fmcw_dense_simo.py \
  applications/fmcw_dense_simo/python/fmcw_dense_simo_smoke.yaml
```

### Run all full demos (needs a display for Holoviz)

```bash
./build/applications/fmcw_siso/cpp/fmcw_siso &
./build/applications/fmcw_simo/cpp/fmcw_simo &
./build/applications/fmcw_mimo/cpp/fmcw_mimo &
./build/applications/fmcw_mimo/cpp/fmcw_mimo fmcw_mimo_ddm.yaml &
./build/applications/fmcw_mimo/cpp/fmcw_mimo fmcw_mimo_ddm_phase_codes.yaml &
./build/applications/fmcw_dense_simo/cpp/fmcw_dense_simo &
```

Or all Python smokes (no GUI expected):

```bash
export HOLOSCAN_RADAR_BUILD_DIR=./build
python applications/fmcw_siso/python/fmcw_siso.py \
  applications/fmcw_siso/python/fmcw_siso_smoke.yaml
python applications/fmcw_simo/python/fmcw_simo.py \
  applications/fmcw_simo/python/fmcw_simo_smoke.yaml
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_smoke.yaml
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_ddm_smoke.yaml
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_ddm_phase_codes_smoke.yaml
python applications/fmcw_dense_simo/python/fmcw_dense_simo.py \
  applications/fmcw_dense_simo/python/fmcw_dense_simo_smoke.yaml
```

---

## 8. Pointers

| Doc | Role |
| --- | --- |
| [`STATUS.md`](STATUS.md) | Claims / checklist state |
| [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md) | Team decision index |
| [`OPERATOR_API.md`](OPERATOR_API.md) | Locked contracts |
| [`DEVELOPER.md`](../../DEVELOPER.md) | Build / test quick reference |
| [`performance/python/README.md`](../../performance/python/README.md) | Python perf CLI details |
| `applications/*/README.md` | Per-app Holoviz / YAML detail |
| `operators/fmcw/mimo_coherent_compensation_op/OPEN_ITEMS.md` | DDM-NL §A.10 |
| `operators/angle_estimation/mvdr_estimator_op/OPEN_ITEMS.md` | S1 / MVDR open surface |
| `operators/fmcw/fmcw_range_doppler_processor_op/OPEN_ITEMS.md` | Fused keep/kill + harness |
| `operators/OPEN_ITEMS.md` | Module-wide alloc deferral |
