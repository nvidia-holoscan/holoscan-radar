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

2. **`weight` is unconnected in `sar_formation` spotlight and independent images.**
   All weights are `+1` on the spotlight row, and a connected
   `weight` port makes `BackprojectionOp` D2H-sync every tick to
   prove that. Video SAR wires it (`max_aperture_pulses > 0`) so
   eviction can emit `−1`.

3. **TESTING.md items not covered as unit tests here:** a fragment
   with metadata disabled / default `kRaise`; the announcement
   surviving a coalescing (non-1:1) stage; bit-comparable re-form
   vs a direct formation *image* (needs `BackprojectionOp` in the
   same process). Pulse-identity coverage for reform, D22,
   reset-without-`−1`, and the growing-aperture continue row is in
   `tests/cpp/sar/test_aperture_manager.cpp`.

4. **Four unguarded spots found in the 2026-10-30 `dpc/sar-autofocus`
   push review.** All predate that branch (blamed to `b591e829`). A fifth,
   the missing `TGeom` assertions, was fixed there. None of the four is
   reachable from a landed app config —
   they are listed so the next person in this file does not have to
   re-derive them. In rough order of consequence:

   - **`setup()` dereferences image extents the constructor never
     required** (`aperture_manager_op.cpp:160`). The ctor
     `require_extent`s only `max_pulses` and `max_range_bins`, but
     `setup()` reads `*extents_.max_image_v` / `*extents_.max_image_u`
     unconditionally to size the `sample_points_out` pool. An app that
     omits them gets an empty-optional dereference instead of the named
     error the other two produce. Either `require_extent` both in the
     ctor, or size that pool only when the port is wired.
   - **`read_scalar` checks size but not dtype**
     (`aperture_manager_op.cpp:48`). The guard is rank 1, one element,
     and `nbytes() == sizeof(Value)`. `evict` is read as `uint32_t`, so
     a float32 `evict` tensor passes every check and is reinterpreted:
     `1.0f` becomes 1065353216 pulses to drop. `control` is `uint8_t`,
     where the only 1-byte alternatives are also integral, so the
     exposure there is small. Add a `require_dtype`.
   - **`sample_points` is never dtype-checked**
     (`aperture_manager_op.cpp:268`). `positions` and
     `reference_position` both get `require_dtype(..., real_dtype<TGeom>())`
     a few lines later; the points tensor is moved straight into
     `pending_points_` and copied out to a pool declared
     `real_dtype<TGeom>()`. A float32 lattice into a `TGeom = double`
     instance is a reinterpretation, not a conversion.
   - **The cadence counter resets only when `control_out` is wired**
     (`aperture_manager_op.cpp:392`). `admitted_since_emit_ = 0` sits
     inside `if (emit_now && control_output_.active())`. With
     `emit_interval_pulses > 0` and `control_out` unwired, the counter
     never returns to zero and the `>=` test at `:312` latches true. The
     latch is currently unobservable rather than harmful: `end_aperture`
     zeroes the counter at `:434` in `reset` and `reform`, and in
     `continue` mode `emit_now` feeds nothing but the disabled
     `control_out`. Move the assignment out of the `active()` branch
     anyway — the reset does not belong behind a port's wiring.

## Not ours to close

- **R10** — metadata key namespace. The manager reads the named
  constant; it does not choose the prefix.
