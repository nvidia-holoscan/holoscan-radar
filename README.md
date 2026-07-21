# Holoscan Radar

Accelerated radar signal processing operators for the Holoscan ecosystem.

Holoscan Radar is a **Holoscan Module** — a self-contained, redistributable library that extends
[Holoscan SDK](https://developer.nvidia.com/holoscan-sdk) with reusable operators under the
`holoscan.radar` namespace.

---

## Quick Start

```bash
# Run the C++ demo application (synthetic SISO FMCW chain)
./holohub run fmcw_siso
```

---

## Operators

All operators below are C++/CUDA (MatX) with **Python bindings for the full
set** (`holoscan.radar`, generated from the same pybind11-wrapped classes —
see `python/holoscan/radar/`). Each operator's header documents its full
port/parameter contract; precision is fixed per instance by the template
parameter (`float` → complex64, `double` → complex128; Python exposes both as
explicit `...OpF32` / `...OpF64` classes plus a bare `...Op` alias for `F32`).

### FMCW signal processing + MIMO marshaling

Shared conventions and the full table live in
[`operators/fmcw/README.md`](operators/fmcw/README.md).

| Operator | Function |
| --- | --- |
| `FMCWRangeProcessorOp<T>` | Range compression: optional software dechirp, windowed fast-time FFT; `range_bins` axis output |
| `ClutterFilterOp<T>` | Slow-time clutter suppression: mean subtraction or 1st/2nd-order MTI delay cancellers |
| `FMCWDopplerProcessorOp<T>` | Doppler compression: windowed slow-time FFT, zero-Doppler centering; `velocity_bins` axis output |
| `FMCWRangeDopplerProcessorOp<T>` | Fused range + optional inline clutter + Doppler in one node (bitwise parity with the split chain) |
| `MIMODeinterleaveOp<T>` | TDM marshaling: split interleaved slow-time into an explicit TX axis via `TXSchedule` gather |
| `VirtualArrayAssemblyOp<T>` | MIMO marshaling: flatten separated TX×RX channels into a virtual array axis |
| `MIMOCoherentCompensationOp<T>` | MIMO marshaling: TDM inter-TX phase correction, or DDM Doppler-band demux into an explicit TX axis |

### Detection

Full table in [`operators/detection/README.md`](operators/detection/README.md).

| Operator | Function |
| --- | --- |
| `CFARDetectorOp<T>` | 2-D CFAR (CA/GO/SO/OS) on the (Doppler, range) plane; adaptive per-cell threshold from the training-cell noise estimate |
| `FeatureAssemblyOp<T>` | Gather / bin→physical lookup into the clustering feature matrix |
| `DetectionReductionOp<T>` | One representative CFAR cell per cluster (`first` / `peak` / `medoid`) |

### Clustering

Full table in [`operators/clustering/README.md`](operators/clustering/README.md).

| Operator | Function |
| --- | --- |
| `ClusterDBSCANOp<T>` | Density clustering (DBSCAN) over an N-D feature matrix; noise labeled `-1` |

### Angle estimation

Full table in [`operators/angle_estimation/README.md`](operators/angle_estimation/README.md).

| Operator | Function |
| --- | --- |
| `SnapshotGatherOp<T>` | Per-detection channel-vector gather from the RD neighborhood around each CFAR cell |
| `BeamscanEstimatorOp<T>` | Conventional (Bartlett) sparse-path DOA on a scan grid |
| `MVDREstimatorOp<T>` | Adaptive (MVDR / Capon) sparse-path DOA; multi-source + eigenvalue diagnostics |
| `PhaseShiftBeamformerOp<T>` | Conventional (Bartlett) dense-path beamformer: a beam for every RD cell → Range-Doppler-Angle cube |

### Shared config-plane types

`ArrayGeometry`, `AngleConvention`, `WindowSpec`, `FMCWWaveformDescriptor`,
`DDMCodeDescriptor`, `TXSchedule` — see
[`operators/common/README.md`](operators/common/README.md). All are
constructible from Python and flow through Holoscan's `from_config()` on
either side.

### App-support library (`radar_app`)

Report + Holoviz visualization operators reused across the example apps
(`ClusterReportOp`, `DetectionReportOp`, `RollingPlotOp`, `BoresightViewOp`,
`ClusterOverlayOp`, `RdMapToImageOp`, `RdChannelSliceOp`, `NullSinkOp`,
`CadenceMeter`) — see
[`applications/common/README.md`](applications/common/README.md). Also bound
to Python; every `applications/fmcw_*` demo uses them from either language.

### Namespace

- C++: `holoscan::radar`

- Python: `holoscan.radar`

---

## Usage

### C++

```cpp
#include <holoscan/holoscan.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"
#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"

class MyApp : public holoscan::Application {
 public:
  void compose() override {
    using holoscan::Arg;
    namespace radar = holoscan::radar;

    radar::FMCWWaveformDescriptor waveform;
    waveform.carrier_frequency = 77e9;
    waveform.bandwidth = 150e6;
    waveform.chirp_duration = 20e-6;
    waveform.sample_rate = 12.8e6;
    waveform.chirp_repetition_interval = 25e-6;

    auto range = make_operator<radar::FMCWRangeProcessorOp<float>>(
        "range", Arg("waveform") = waveform);
    auto clutter = make_operator<radar::ClutterFilterOp<float>>("clutter");
    auto doppler = make_operator<radar::FMCWDopplerProcessorOp<float>>(
        "doppler", Arg("waveform") = waveform);

    // add_flow(source, range, {{"adc_cube", "adc_cube"}}); ...
    add_flow(range, clutter, {{"range_profiles", "range_profiles"}});
    add_flow(clutter, doppler, {{"range_profiles", "range_profiles"}});
  }
};

int main() { holoscan::make_application<MyApp>()->run(); }
```

See [`applications/fmcw_siso`](applications/fmcw_siso) for the
complete example, including the synthetic ADC source and peak reporting.

### Python

```python
from holoscan.radar import (
    ClutterFilterOp,
    FMCWDopplerProcessorOp,
    FMCWRangeProcessorOp,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
)

waveform = FMCWWaveformDescriptor(
    FMCWWaveformSpec(
        carrier_frequency=77e9,
        bandwidth=150e6,
        chirp_duration=20e-6,
        sample_rate=12.8e6,
        chirp_repetition_interval=25e-6,
    )
)

range_op = FMCWRangeProcessorOp(self, waveform=waveform, name="range")
clutter = ClutterFilterOp(self, name="clutter")
doppler = FMCWDopplerProcessorOp(self, waveform=waveform, name="doppler")

# self.add_flow(source, range_op, {("adc_cube", "adc_cube")}); ...
self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
self.add_flow(clutter, doppler, {("range_profiles", "range_profiles")})
```

See [`applications/fmcw_siso/python`](applications/fmcw_siso/python) for the
complete example. Every `applications/fmcw_*` app has a Python variant with
the same graph as its C++ counterpart.

---

## Building from Source (without HoloHub CLI)

| Requirement | Version |
| --- | --- |
| Holoscan SDK | ≥ 4.3.0 |
| CUDA Toolkit | 13.x |
| CMake | ≥ 3.31 |
| C++ compiler | ≥ C++17 (GCC 11+) |
| pybind11 | ≥ 2.11 |
| Python | 3.10 - 3.13 |

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

---

## Testing

Tests build with the code by default in a standalone checkout
(`HOLOSCAN_RADAR_BUILD_TESTING` defaults to ON; the explicit flag below
matters only when this module is built as a subproject):

```bash
./holohub run-container
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING:BOOL=ON
cmake --build build -j$(nproc)

# C++ (GTest via CTest)
ctest --test-dir build --output-on-failure -L unit

# Python (pytest via CTest, or directly)
ctest --test-dir build --output-on-failure -R pytest
HOLOSCAN_RADAR_BUILD_DIR=build pytest tests/python/ -v

# Optional long-running Monte Carlo statistical tests (CFAR Pfa, DOA vs. CRLB)
HOLOSCAN_RADAR_BUILD_DIR=build pytest tests/python/ -v -m statistical
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
