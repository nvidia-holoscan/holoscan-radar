# FMCW MIMO — Python demo

Python Holoviz demo mirroring [`../cpp/fmcw_mimo.cpp`](../cpp/fmcw_mimo.cpp). Supports
TDM and DDM multiplexing via `mimo_coherent_compensation.multiplexing_scheme` in the
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
