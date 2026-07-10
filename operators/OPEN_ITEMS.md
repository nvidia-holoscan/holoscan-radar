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
