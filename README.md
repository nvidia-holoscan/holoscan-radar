# Holoscan Radar

Accelerated radar signal processing operators for the Holoscan ecosystem.

Holoscan Radar is a **Holoscan Module** — a self-contained, redistributable library that extends
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

```bash
./holohub run-container
cmake -S . -B build/tests -DHOLOSCAN_RADAR_BUILD_TESTING:BOOL=ON

# C++ (GTest via CTest)
ctest --test-dir build/tests --output-on-failure -L unit

# Python (pytest via CTest)
ctest --test-dir build/tests --output-on-failure -R pytest
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
