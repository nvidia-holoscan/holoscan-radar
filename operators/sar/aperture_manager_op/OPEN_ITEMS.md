# ApertureManagerOp — open items

CPU-only; there is no GPU hot path, so no `perf_aperture_manager`
harness (checklist Performance section skipped).

## Implementation follow-ups

1. **D22 value encoding is still the interim string.** The manager
   reads `holoscan.radar.sar.stream_boundary` as
   `"end_of_stream"` / `"collection_boundary"`
   (`radar_common/sdk/stream_boundary.hpp`), matching
   `CPHDReaderOp` and item 11 there. The contract's `uint8`
   `StreamBoundary` enum is not switched in this landing.

2. **`weight` is unconnected in `sar_formation` modes 1 and 5.**
   All weights are `+1` on the spotlight row, and a connected
   `weight` port makes `BackprojectionOp` D2H-sync every tick to
   prove that. Mode 2 wires it (`max_aperture_pulses > 0`) so
   eviction can emit `−1`.

3. **TESTING.md items not covered as unit tests here:** a fragment
   with metadata disabled / default `kRaise`; the announcement
   surviving a coalescing (non-1:1) stage; bit-comparable re-form
   vs a direct formation *image* (needs `BackprojectionOp` in the
   same process). Pulse-identity coverage for reform, D22,
   reset-without-`−1`, and the growing-aperture continue row is in
   `tests/cpp/sar/test_aperture_manager.cpp`.

## Not ours to close

- **R10** — metadata key namespace. The manager reads the named
  constant; it does not choose the prefix.
