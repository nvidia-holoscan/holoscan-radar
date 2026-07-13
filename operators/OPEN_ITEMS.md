# Operators — Open Items

Deferred, agreed-upon work that spans the operator libraries. Each item
records the intent and the reason it was deferred, so it can be picked up
without re-litigating the design. Remove items when they land (note the
commit in the PR).

## 1. Reduce per-tick output allocations (optional buffer reuse)

**Status:** agreed 2026-07-10, acceptable for now; revisit when latency
budgets tighten or benchmarking flags it.

Every operator allocates its emitted output tensors with
`make_device_tensor` (`radar_common/tensor/dlpack_buffer.hpp`) on every
`compute()` tick — a synchronous `cudaMalloc`/`cudaFree` pair per output per
tick (rd_maps, range_profiles, plus the axis outputs when enabled). Stage
*intermediates* already avoid this via `MATX_ASYNC_DEVICE_MEMORY`
(stream-ordered, pool-backed `cudaMallocAsync`), but emitted outputs cannot:
an emitted tensor escapes the operator and lives however long downstream
consumers hold it, so its buffer cannot be freed stream-ordered on the
producing operator's stream.

Future work: an optional buffer-reuse path to cut the allocation latency,
e.g.

- an optional `holoscan::Allocator` parameter (`BlockMemoryPool` /
  `RMMAllocator`) so the application chooses pooling, matching the pattern
  used by Holoscan SDK operators; or
- operator-owned recycling: cache output buffers keyed by (shape, dtype) and
  reuse when the previous tick's tensor has been released (shapes are stable
  in steady state but can change via nfft/clutter overrides, so keying —
  not assuming — is required).

Whichever form, it must stay optional with today's allocate-per-tick
behavior as the default, and buffer lifetime must remain correct for
downstream consumers that hold emitted tensors across ticks (the reason the
naive stream-ordered approach is off the table).

## 2. Cache the axis outputs (range/velocity bins) across ticks

**Status:** agreed 2026-07-13, acceptable for now; revisit when latency
budgets tighten or benchmarking flags it.

When axis outputs are enabled, the range/velocity bin builders recompute the
axis on the host, upload it, and block on `cudaStreamSynchronize` on *every*
`compute()` tick, even when the inputs that determine the axis are unchanged.
The window taps in the same stage states already avoid this via the
`ConfigCache` pattern (`radar_common/sdk/config_cache.hpp`); the axis builders
should follow it, computing/uploading only on a cache miss for a new
configuration and reusing the cached device buffer (or an async device-side
copy into the output) on subsequent ticks.

Affected paths (each called per tick from the operator's `compute()`):

- `fmcw_range_processor_op` — `fill_range_bins` keyed on the fields it reads
  (waveform, `nfft`/`samples_per_chirp`, `propagation_speed`,
  `num_bins`); see `range_stage.hpp`.
- `fmcw_doppler_processor_op` — `fill_velocity_bins` keyed on waveform,
  `center_zero_doppler`, and `num_doppler`; see `doppler_stage.hpp`.
- `fmcw_range_doppler_processor_op` — the fused operator calls both builders,
  so it inherits both once the two above are cached.

Keep the existing `fill_*` logic as the cache-fill helper so behavior on a
miss is unchanged; only the steady-state (cache-hit) path becomes non-blocking.
