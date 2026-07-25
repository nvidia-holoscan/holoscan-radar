# Radar point-target simulator (`radar_common/sim`)

A synthetic FMCW radar **ADC-cube generator**: describe a scene (waveform +
moving point targets + array geometry), and the simulator synthesizes the raw
complex IF data cube the signal-processing operators consume. The runtime path
runs on the GPU (MatX); an independent CPU implementation under `tests/` is the
analytic ground truth it is validated against.

It is a closed-form signal model, **not** an EM solver: each target's
contribution (far-field two-way delay → beat frequency → slow-time carrier
phase) is summed over `(rx, slot, fast-time sample)`.

## Files

| File | Role |
|---|---|
| `scene.hpp` | Scene model — `SimScene`, `PointTarget`, `SimWaveform`, `derive_truth`, the `SceneMover` seam. STL-only, header-only. |
| `simulator.hpp` / `simulator.cu` | GPU solvers (`simulate_gpu`, `simulate_mimo_gpu`) + the noise additive (`add_thermal_noise_gpu`). Compiled into `holoscan::radar_common_simulator`. |
| `array_builders.hpp` | Array-geometry helpers (URA / virtual / linear element positions). |
| `../../../../tests/cpp/utils/sim_reference.hpp` | CPU single-TX reference (oracle). |
| `../../../../tests/cpp/utils/mimo_reference.hpp` | CPU MIMO reference (oracle). |
| `../../../../tests/cpp/utils/additives_reference.hpp` | CPU noise additive (oracle). |

## Pipeline

```
 SimScene ─────────────►  simulate_gpu / simulate_mimo_gpu  ──►  add_thermal_noise_gpu  ──►  device cube
 (targets, waveform,          (clean scatterer response,          (I/Q additive, in place)     (complex<float>
  array geometry, snr,          double → complex<float>)                                        holoscan::Tensor)
  seed, dechirped)
        ▲
        │  SceneMover (per-frame): advance targets / regenerate them
        └──────────────────────────────────────────────────────────
```

The three stages are independent: the **solver** is agnostic to how the scene
was built, **additives** compose linearly on the clean cube, and the **mover**
evolves the scene between frames — each is replaceable without touching the
others.

## Quick start

```cpp
#include "radar_common/sim/scene.hpp"
#include "radar_common/sim/simulator.hpp"
namespace sim = holoscan::radar::sim;

sim::SimScene scene;                       // defaults are a valid 77 GHz scene
scene.num_chirps = 64;
scene.samples_per_chirp = 256;
scene.targets.push_back({ /*pos m*/ {0, 40, 0}, /*vel m/s*/ {0, 5, 0}, /*amp*/ {1, 0} });
scene.snr_db = 20.0;                        // omit for noise-free

cudaStream_t stream; cudaStreamCreate(&stream);
auto cube = sim::simulate_gpu(scene, stream);              // device complex<float> tensor
sim::add_thermal_noise_gpu(*cube, scene.snr_db, scene.seed, stream);
// `cube` is ready to emit / feed the range processor. Caller owns the stream.
```

Link: `target_link_libraries(<you> PRIVATE holoscan::radar_common_simulator)`
(MatX is a private implementation detail; you do not link it).

## API

### Scene model — `scene.hpp`

**`SimScene`** — the full description you fill in:

| Field | Type | Meaning / units |
|---|---|---|
| `targets` | `vector<PointTarget>` | The point scatterers. |
| `waveform` | `SimWaveform` | FMCW waveform (below). |
| `num_chirps` | `int64_t` | Slow-time length (pulses per CPI). |
| `samples_per_chirp` | `int64_t` | Fast-time length (ADC samples/chirp). |
| `tx_positions` | `vector<Vec3>` | TX element positions, m, array frame. Single-TX path needs exactly 1. |
| `rx_positions` | `vector<Vec3>` | RX element positions, m, array frame. |
| `propagation_speed` | `double` | m/s (defaults to c). |
| `snr_db` | `optional<double>` | Noise SNR in dB vs. a unit-amplitude target; unset = noise-free. |
| `seed` | `uint64_t` | Noise RNG seed. |
| `dechirped` | `bool` | `true` = dechirped IF cube (normal); `false` = pre-mixer cube. |

- **`PointTarget`** — `position` (m, Cartesian array frame), `velocity` (m/s,
  relative — fold ego motion in yourself), `amplitude` (`complex<double>`).
- **`SimWaveform`** — `carrier_frequency` (Hz), `bandwidth` (Hz) **or** derive
  from `chirp_duration`, `sample_rate` (Hz), `chirp_repetition_interval` (s,
  PRI). Helpers: `slope()`, `wavelength()`. `Vec3 = std::array<double,3>`.

### Solvers — `simulator.hpp`

```cpp
std::shared_ptr<holoscan::Tensor>
simulate_gpu(const SimScene& scene, cudaStream_t stream);
```
Single-TX (SISO/SIMO). Requires exactly one `tx_positions` element. Returns a
device `complex<float>` cube (see Output contract).

```cpp
std::shared_ptr<holoscan::Tensor>
simulate_mimo_gpu(const SimScene& scene,
                  const std::vector<std::complex<double>>& weights,
                  int64_t slow_time, cudaStream_t stream);
```
Multi-TX. The multiplexing scheme is captured **entirely** by `weights` — a
row-major `(num_tx, slow_time)` matrix of the per-`(tx, slot)` excitation:

| Scheme | `slow_time` | `weights[tx, slot]` |
|---|---|---|
| TDM | `num_tx * num_chirps` | `1` on the slot's active TX, `0` otherwise |
| DDM (Doppler-coded) | `num_chirps` | `exp(i·2π·offset_tx·k)` |
| DDM (phase-coded) | `num_chirps` | the arbitrary unit-magnitude `(tx, k)` code |

(Single-TX is just `num_tx == 1` with unit weights — the two entry points share
one kernel internally.)

### Additive — `simulator.hpp`

```cpp
void add_thermal_noise_gpu(holoscan::Tensor& cube,
                           const std::optional<double>& snr_db,
                           uint64_t seed, cudaStream_t stream);
```
Adds seeded complex white Gaussian noise **in place** on the device (no-op when
`snr_db` is unset). Advance `seed` per frame for a fresh realization each tick.

### Geometry — `array_builders.hpp`

`make_ura_rx_positions`, `make_ura_geometry`, `make_virtual_geometry` (TX⊕RX),
`make_linear_positions`, `flatten_positions` — build element positions from
element counts + spacing (in wavelengths).

## Output contract

- dtype `complex<float>`, compact **row-major**, no padding.
- `simulate_gpu`: `(num_chirps, samples)` for 1 RX, else
  `(num_rx, num_chirps, samples)`.
- `simulate_mimo_gpu`: always `(num_rx, slow_time, samples)`.
- **Stream ownership:** the caller creates/owns `stream`; both solvers and the
  additive enqueue on it and synchronize before returning, so the tensor is
  ready to use/emit.

## Conventions (must match the operators)

- **Dechirped sample:** `s(k,t) = A·exp(+j·2π·f_b(k)·t − j·2π·f_c·τ_k)` with beat
  `f_b(k) = slope·τ_k`, so a forward FFT along fast-time peaks at the positive
  beat frequency (bin 0 = zero range).
- **Doppler:** the slow-time phase `−2π·f_c·τ_k` advances by `2π·f_d·PRI` per
  chirp with `f_d = −2·v_r/λ`; radial velocity `v_r` positive = receding.
- **Reference chirp:** `r(t) = exp(+j·π·slope·t²)`; pre-mixer = `s·r`, dechirp =
  multiply by `conj(r)`.
- Far-field plane wave, monostatic, stop-and-hop (target frozen within a chirp,
  advances between chirps), single-slope sweeps, ideal hardware.

## Extension points

- **Motion model / procedural targets — `SceneMover`:**
  `using SceneMover = std::function<void(SimScene&, double dt)>;`. Set it on the
  source operator; the default (`linear_mover()`) advances targets along their
  velocity. A custom mover can implement orbits/maneuvers or
  clear-and-regenerate `scene.targets` (a procedural scatterer source) — without
  touching the solver.
- **More additives:** compose additional `add_*(cube, …)` on the device cube
  after the solver (clutter, interference tones, IF/DC leakage) — they add
  linearly and are independent of the scene/solver.
- **Alternative solver backend:** honor the same `SimScene → device cube`
  contract (shapes, dtype, conventions above) and swap `simulate_gpu` at the
  call site.

## Validation

`simulate_gpu` / `simulate_mimo_gpu` are validated against the CPU references
(`tests/cpp/utils/*_reference.hpp`) to a tight tolerance — observed
`max|gpu − cpu| ≈ 0` (byte-identical float32 for most scenes), well below the
downstream float32 processing floor. The CPU reference is deliberately kept
STL-only and outside the operators tree so it stays an implementation-independent
oracle (see `tests/cpp/common/test_sim_gpu.cpp`).
