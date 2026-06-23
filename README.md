# radar

Accelerated radar signal processing operators for the Holoscan ecosystem

A **Holoscan Module** — a self-contained, redistributable library that extends
[Holoscan SDK](https://developer.nvidia.com/holoscan-sdk) with reusable operators under the
`holoscan.radar` namespace.

---

## Quick Start

```bash
# 1. Run the Python demo application
./holohub run radar_pipeline --language python

# 2. Run the C++ demo application
./holohub run radar_pipeline

```

---

## Operators

| Operator | Implementation | Ports | Parameters |
|---|---|---|---|
| `RadarOp` | C++ + pybind11 | TODO | TODO |

**Namespace**


- C++: `holoscan::radar`

- Python: `holoscan.radar`

---

## Usage

### Python

```python
from holoscan.core import Application
from holoscan.radar import RadarOp


class MyApp(Application):
    def compose(self):
        op = RadarOp(self, name="radar_op")
        # TODO: connect operators and add_flow calls


MyApp().run()
```



### C++

```cpp
#include <holoscan/holoscan.hpp>
#include <radar_op/radar_op.hpp>

class MyApp : public holoscan::Application {
 public:
  void compose() override {
    auto op = make_operator<holoscan::radar::RadarOp>("radar_op");
    // TODO: connect operators
  }
};

int main() { holoscan::make_application<MyApp>()->run(); }
```



---

## Building from Source (without HoloHub CLI)

| Requirement | Version |
|---|---|
| Holoscan SDK | ≥ 4.3.0 |
| CUDA Toolkit | 13.x (matches the Holoscan SDK CUDA pin; the dev `Dockerfile` uses `cuda13-dgpu`) |
| CMake | ≥ 3.24 |
| C++ compiler | C++17 (GCC 11+) |
| pybind11 | ≥ 2.11 |
| Python | 3.10–3.13 |

```bash
cmake -S . -B build -DBUILD_ALL=ON -DRADAR_BUILD_TESTING=ON
cmake --build build -j$(nproc)
```

---

## Testing

```bash
./holohub test
```

Or, without the HoloHub CLI:



```bash
# C++ (GTest via CTest)
ctest --test-dir build --output-on-failure -L unit
```



```bash
# Python (pytest)
PYTHONPATH=build/python${PYTHONPATH:+:$PYTHONPATH} RADAR_BUILD_DIR=build pytest tests/python/ -v
```

`PYTHONPATH` is **prepended** (with `${PYTHONPATH:+:$PYTHONPATH}`) so that an
ambient holoscan SDK install stays visible. A bare `PYTHONPATH=build/python`
would replace the variable and hide the SDK from pytest. A bare
`PYTHONPATH=build/python:$PYTHONPATH` looks safe but, when `$PYTHONPATH` is
unset (typical on a fresh shell or CI runner), it expands to a trailing colon
that Python reads as an empty path entry — equivalent to `.`, which silently
adds the test CWD to `sys.path` and lets a local file shadow installed
packages.

The pytest suite currently covers importability and build-smoke only; full
live-pipeline coverage is a TODO and may require real hardware. CTest is
configured with `SKIP_RETURN_CODE 5` on the pytest entry — if a CTest run
reports "Skipped" unexpectedly, the most common cause is that the holoscan
SDK is not importable in the test environment (pytest collects zero items
and exits 5).

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
