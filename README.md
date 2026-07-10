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

Each operator's header documents its full port/parameter contract; shared
conventions live in [`operators/common/README.md`](operators/common/README.md)
and the group overview in [`operators/fmcw/README.md`](operators/fmcw/README.md).
Precision is fixed per instance by the template parameter (`float` →
complex64, `double` → complex128).

| Operator | Implementation | Function |
| --- | --- | --- |
| `FMCWRangeProcessorOp<T>` | C++/CUDA (MatX) | Range compression: optional software dechirp, windowed fast-time FFT; `range_bins` axis output |
| `ClutterFilterOp<T>` | C++/CUDA (MatX) | Slow-time clutter suppression: mean subtraction or 1st/2nd-order MTI delay cancellers |
| `FMCWDopplerProcessorOp<T>` | C++/CUDA (MatX) | Doppler compression: windowed slow-time FFT, zero-Doppler centering; `velocity_bins` axis output |
| `FMCWRangeDopplerProcessorOp<T>` | C++/CUDA (MatX) | Fused range + optional clutter + Doppler in one node (bitwise parity with the split chain) |

Python bindings are deferred to a later phase.

### Namespace

- C++: `holoscan::radar`

- Python: `holoscan.radar` (placeholder until bindings land)

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

# Python (pytest via CTest)
ctest --test-dir build --output-on-failure -R pytest
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
