# FMCW MIMO — Python demo

Python Holoviz demo mirroring [`../cpp/fmcw_mimo.cpp`](../cpp/fmcw_mimo.cpp). Supports
TDM and DDM multiplexing via top-level `multiplexing_scheme` in the
YAML config.

## Prerequisites

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

## Run

```bash
export HOLOSCAN_RADAR_BUILD_DIR=build
# TDM (default)
python applications/fmcw_mimo/python/fmcw_mimo.py [yaml]
# DDM
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/cpp/fmcw_mimo_ddm.yaml
# DDM non-linear phase_codes
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/cpp/fmcw_mimo_ddm_phase_codes.yaml
```

Default TDM config: [`../cpp/fmcw_mimo.yaml`](../cpp/fmcw_mimo.yaml).

Headless smoke (2 frames, no Holoviz):

```bash
export HOLOSCAN_RADAR_BUILD_DIR=build
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_smoke.yaml
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_ddm_smoke.yaml
python applications/fmcw_mimo/python/fmcw_mimo.py \
  applications/fmcw_mimo/python/fmcw_mimo_ddm_phase_codes_smoke.yaml
```

## Notes

- DDM uses the app-local `DdmReportVelocityBinsOp` (Python port) for the post-demux
  velocity axis in reporting and plots.
- Non-linear DDM configs may use demo-only `generate_phase_codes: zadoff_chu`
  (or `zc`) under `ddm_code` for low-crosstalk correlate demux without listing
  every complex sample. `sin` remains available to stress high crosstalk.

## Layout

Same layering as the C++ side (see [`../README.md`](../README.md) and
[`../../common/README.md`](../../common/README.md)):

- `fmcw_mimo.py` — the graph only: `FmcwMimoApp.compose()` plus a `main()`
  that calls the shared `common.python.driver.run_fmcw_app()`. Every operator
  block flows through Holoscan's `from_config()` **except**
  the selected dense operator block, which must stay a manual kwargs dict (see
  the split dense parsers's docstring): the app
  injects an app-resolved `ddm_code` that shares a YAML key with that block,
  and Holoscan's Python `Operator.__init__` applies `from_config()`'s
  `ArgList` after all kwargs regardless of call-site order, so `from_config()`
  would silently discard the resolved override.
- `config.py` — colocated, app-owned: `AppConfig` (inherits
  `demo_config.CommonRunConfig`) + `load_config()` (TDM/DDM branch, virtual
  array geometry, targets).
- `custom_ops.py` — colocated, app-specific: `MimoTdmAdcSourceOp` /
  `MimoDdmAdcSourceOp` (MIMO's multiplexing needs a bespoke simulator, so it
  keeps its own sources rather than the shared `SimAdcSourceOp`) and
  `DdmReportVelocityBinsOp`.
- Reusable across all four demos, in `applications/common/python/`: the
  point-target simulator (`sim.py`), the MIMO TDM/DDM simulators
  (`mimo_sim.py`, co-located beside `sim.py` like the C++
  `radar_common/sim/mimo_sim.hpp`), the shared port-drain operator
  (`ops.TensorMapDiscardOp`), the entry-point scaffolding (`driver.py`), and
  the YAML-parsing helpers (`demo_config.py`).
