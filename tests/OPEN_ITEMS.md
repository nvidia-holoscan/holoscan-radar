# Test Suite — Open Items

Deferred, agreed-upon work for the test tree. Each item records the intent
and the reason it was deferred, so it can be picked up without re-litigating
the design. Remove items when they land (note the commit in the PR).

## 1. MatX / GPU path for the analytic point-target simulator?

**Open question (2026-07-19).** Should C++ demos synthesize ADC cubes on the
GPU via MatX, instead of (or in addition to) the host STL
`radar_common/sim/point_target_sim.hpp` path?

**Context.**

- `point_target_sim.hpp` is **host-only by design** (`STATUS.md` settled:
  simulator + window generation). STL-only, no MatX/CUDA — so tests assert
  against a signal model independent of the operator stack (`TESTING.md`
  tautology guard; fixed `mt19937_64` seed → bit-identical cubes).
- Python demos already synthesize on GPU (CuPy) and emit device tensors;
  C++ demos still call host `simulate()` then upload.
- Dense / MIMO cubes are large enough that host synthesis can matter for
  demo cadence; operators themselves are not the bottleneck in those cases.

**Constraints (do not silently violate).**

- Do **not** replace or MatX-ify the shared SoT header used by unit /
  integration tests. Ground truth must stay derived from the signal model,
  not from MatX code under test.
- A GPU path, if added, must mirror the same far-field dechirped formula and
  noise-power convention; bit-identical noise vs host `mt19937_64` is not
  required (CuPy already differs) unless someone explicitly needs cross-lang
  cube dumps.

**Candidate answers (not chosen).**

1. **Status quo** — keep host SoT + host C++ demo sources; Python stays on
   CuPy. Only revisit if profiling shows host sim blocking a target frame
   period.
2. **Demo-only MatX synthesizer** — e.g. `applications/common/...` or a
   non-SoT helper that emits device tensors; tests keep calling
   `point_target_sim.hpp`.
3. **Shared MatX helper + host SoT** — same as (2), but factored for reuse
   across C++ demos; still never the tautology source.

**Interim.** Leave host SoT alone. Measure dense-SIMO / MIMO C++ host-sim
cost before building a MatX path. Python CuPy already covers the “GPU-side
demo sim” experiment on that side.

**Status:** open — needs a deliberate pick (likely after a cheap timing
diagnostic); not a TEAM_BLOCKED contract stop.

## 2. Simulator should regenerate noise each frame

**Status:** open — low-impact change; performance hit needs to be studied before landing.

The host `point_target_sim.hpp` path draws noise once at construction with a
fixed `mt19937_64` seed, so every frame gets identical noise. For demo realism
the simulator should reseed or draw fresh noise on each `simulate()` call.

**Constraints before landing.**

- Unit / integration tests depend on bit-identical cubes across runs (fixed
  seed is a tautology guard; see `TESTING.md`). The fresh-noise path must be
  opt-in (e.g. a `per_frame_noise` flag or a separate entry point) so the
  deterministic SoT header is untouched.
- The Python / CuPy demo path already draws per-frame noise; the concern is
  the C++ host path only.
- Measure the host noise-generation cost at demo cube sizes (dense-SIMO / MIMO
  xl) before landing — noise draw is O(N) random samples and may be negligible
  relative to upload, or may matter at high frame rates.

<!--
Recently landed:
- Mirror the operator tree under `tests/cpp/` (common/, fmcw/, detection/,
  clustering/; integration label for cross-family tests).
- Split `test_common.cpp` along `radar_common` layers into common/test_window,
  test_descriptors, test_sdk, test_sim, test_tensor.
-->
