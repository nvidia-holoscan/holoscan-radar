# Test Suite — Open Items

Deferred, agreed-upon work for the test tree. Each item records the intent
and the reason it was deferred, so it can be picked up without re-litigating
the design. Remove items when they land (note the commit in the PR).

## 1. Split `test_common.cpp` by concern

**Status:** agreed 2026-07-10. Unblocked — `tests/cpp/common/` now exists
(the family-tree reorg landed; `radar_add_integration_test` labels
`test_pipeline` as `integration`).

`common/test_common.cpp` covers window DSP, descriptors, the config cache, the
point-target simulator, and the YAML converters (~590 lines and growing).
Split it along the `radar_common` subdirectory layers, e.g.:

- `test_window.cpp` (generation + cache keys)
- `test_descriptors.cpp` (`FMCWWaveformDescriptor`, future descriptor types)
- `test_sdk.cpp` (config cache, YAML converters, arg setters)
- `test_sim.cpp` (simulator self-checks)
