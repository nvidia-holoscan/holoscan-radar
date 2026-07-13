# Deferred Fixes

Review findings triaged as `future`: real but non-blocking (minor gaps,
optimizations, or cosmetics). Each records the intent and why it was deferred,
so it can be picked up without re-triaging. Remove items when they land (note
the commit in the PR).

Findings triaged as bugs are fixed immediately; OBE / already-fixed findings
are not recorded here.

## 1. Cache `velocity_bins` in `DopplerStageState`


**File:** `operators/fmcw/fmcw_doppler_processor_op/doppler_stage.hpp` (~155-209)

`fill_velocity_bins` rebuilds the axis, uploads it, and blocks on
`cudaStreamSynchronize` every tick even when `waveform`, `center_zero_doppler`,
and `num_doppler` are unchanged. Move it into the `ConfigCache` pattern (like
windows) so the buffer is computed/uploaded only on a cache miss and reused on
subsequent ticks without host-side blocking.

## 4. Make `find_package(CUDAToolkit)` REQUIRED

**File:** `CMakeLists.txt:40`

Discovery is currently `QUIET`, so a missing toolkit surfaces later as a
confusing target-resolution error where `CUDA::cudart` is consumed
(`tests/cpp/`, `applications/fmcw_siso/cpp/`). Make it `REQUIRED` so
configuration fails immediately with a clear message.

## 5. Rename `emplace` structured binding in `config_cache`

**File:** `operators/common/radar_common/sdk/config_cache.hpp:41`

`auto [inserted, ok] = cache_.emplace(...)` names the bindings backwards from
`unordered_map::emplace`'s `pair<iterator, bool>` order — the first element is
the iterator, not an "inserted" flag. Rename for clarity (cosmetic).
