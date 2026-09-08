# Operators — Open Items

Deferred, agreed-upon work that spans the operator libraries. Each item
records the intent and the reason it was deferred, so it can be picked up
without re-litigating the design. Remove items when they land (note the
commit in the PR).

## 1. Reduce per-tick output allocations (optional buffer reuse)

**Status:** agreed 2026-07-10, acceptable for now; revisit when latency
budgets tighten or benchmarking flags it.

**Diagnostic (2026-07-16, `MIMODeinterleaveOp` xl, RTX PRO 6000 Blackwell):**
per-tick `cudaMalloc`/`cudaFree` via `make_device_tensor` dominates the
Holoscan-timed path at large output sizes. Standalone row-copy kernel ~
0.9 ms; harness with allocate-per-tick ~2.1 ms median; **ping-pong output
reuse (diagnostic only, not landed) ~0.75 ms median** — i.e. most of the
gap is allocation, not gather bandwidth or framework alone.

A drop-in `cudaMallocAsync`/`cudaFreeAsync` path in `make_device_tensor`
was also probed (diagnostic only). `cudaMallocAsync` works in isolation but
failed inside the Holoscan operator tick (`cudaErrorIllegalAddress`, likely
stale error from stream/pool interaction — needs Holoscan memory-pool /
stream setup before it is a safe module-wide change). **Defer** until we
pick an explicit pooling strategy below.

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
- operator-owned **ping-pong** recycling: two output buffers keyed by
  (shape, dtype); flip the slot each tick so emitted tensors are not
  overwritten while a downstream consumer may still hold them (the
  diagnostic above validated this pattern on `MIMODeinterleaveOp` xl); or
- module-wide `make_device_tensor` backed by `cudaMallocAsync` once Holoscan
  stream/memory-pool interaction is understood (not a one-line swap today).

Whichever form, it must stay optional with today's allocate-per-tick
behavior as the default, and buffer lifetime must remain correct for
downstream consumers that hold emitted tensors across ticks (the reason the
naive stream-ordered approach is off the table).

**Per-operator ping-pong (2026-07-16 analysis; not landed):**

Ping-pong does **not** require a team-wide change. Each operator can keep
two `shared_ptr<holoscan::Tensor>` slots, a `(shape, dtype)` key, and flip
the write slot each tick — no `make_device_tensor` / `radar_common` change,
no `OPERATOR_API.md` contract change. `OPERATOR_DEVELOPMENT.md` §3 already
describes this pattern for emitted outputs.

`MIMODeinterleaveOp` is a strong first candidate: static `tx_schedule`, no
`[live]` parameters, pure gather (no stale numerical state), and the
intended MIMO-TDM chain is linear (deinterleave → compensation → virtual
array). The xl diagnostic above is the evidence; implementation deferred for
now (allocate-per-tick remains in code).

`VirtualArrayAssemblyOp` (2026-07-16 review): same allocate-per-tick output
path via `make_device_tensor`. Realloc is **required** when output element
count changes tick-to-tick (`num_doppler`, `num_range_bins`, or
`M_tx`/`N_rx` → `num_virtual`); the sole `[live]` param (`wavelength`)
does **not** change tensor shapes. On the default perf matrix (xl median
~3.3 ms, setup ~3.5 ms) steady state does **not** show the malloc-dominated
profile seen on `MIMODeinterleaveOp` xl — defer ping-pong here as well;
revisit only if profiling flags allocation at target sizes.

`MIMOCoherentCompensationOp` (2026-07-16, TDM slice): same allocate-per-tick
output path. **Diagnostic (xl, RTX PRO 6000 Blackwell):** kernel ~6.7 ms
(~1.28 TB/s, nsys) vs harness tick median ~14.5 ms — roughly half the tick is
`cudaMalloc`/`cudaFree` on the ~4.3 GB output buffer, not compute. Kernel
launch was fixed (was one block per element with 256 redundant threads);
remaining gap is the module-wide alloc pattern in §1. Defer ping-pong here;
intended MIMO-TDM chain is linear (deinterleave → compensation → virtual
array). `[live]` `waveform` does not change output shape.

**Follow-up diagnostic (2026-07-19, DDM-NL `xl`, same GPU):** after the
tiled cuFFT correlate path, nsys shows GPU kernels ~8.0 ms busy / ~8.0 ms
span (pack + FFT + mul + IFFT + scatter; inter-kernel gaps ~0.4 ms) vs
harness median ~15.5 ms. Per tick: `cudaMalloc` of the ~4.3 GB correlate
output ~2.2–2.4 ms and `cudaFree` of the prior output ~8–10.5 ms — same §1
alloc pattern, now confirmed on the non-linear DDM path. NCU on the
correlate kernels (memory-bound; scatter heaviest) is a separate, smaller
lever once §1 is picked up. Still deferred; do not land ping-pong here
without revisiting this item.

Reasons it is reasonable to deploy there when picked up:

- Large perf win at xl (~2.1 ms → ~0.75 ms harness median) with no API
  surface change.
- Correct in the intended pipeline: one consumer, one frame in flight.

Caveats to respect when implementing (none are blockers for this op in
steady-state MIMO-TDM, but document them in the operator):

| Concern | Notes |
| --- | --- |
| Steady-state memory | Two output buffers resident for the session (~2× output bytes at xl). Acceptable on large GPUs; may want gating on memory-tight targets. |
| Lifetime assumption | Two slots are safe only if no downstream consumer holds tick *N*'s output past tick *N+2*. Linear Holoscan chains satisfy this; multi-frame buffering or slow fan-out branches do not. |
| Shape churn | Re-key on `(shape, dtype)` and reallocate both slots on change. Leading batch axes may vary per tick (rank-agnostic batching); ping-pong helps only when geometry is stable. |
| Tests | Add a two-tick correctness case (distinct payloads, no clobber) and a shape-change re-key case. Existing single-tick tests are not sufficient on their own. |

Module-wide pooling (`cudaMallocAsync` in `make_device_tensor`, Holoscan
`BlockMemoryPool` / `RMMAllocator`) remains a separate, deferred decision;
per-operator ping-pong does not commit the module to either path.

## 2. Cache the axis outputs (range/velocity bins) across ticks

**Status:** agreed 2026-07-13, acceptable for now; revisit when latency
budgets tighten or benchmarking flags it.

When axis outputs are enabled, the range-bin builder recomputes the axis on the
host, uploads it, and blocks on `cudaStreamSynchronize` on *every* `compute()`
tick, even when the inputs that determine the axis are unchanged. The Doppler
velocity-bin builder uses a MatX-native `fftfreq`/`fftshift1D` expression, is
stream-ordered, and does not perform a host staging upload or explicit
synchronization; it rebuilds the axis every tick. The window
taps in the stage states already avoid this via the `ConfigCache` pattern
(`radar_common/sdk/config_cache.hpp`); the remaining axis builders should
follow it, computing/uploading only on a cache miss for a new configuration and
reusing the cached device buffer (or an async device-side copy into the
output) on subsequent ticks.

Affected paths (each called per tick from the operator's `compute()`):

- `fmcw_range_processor_op` — `fill_range_bins` keyed on the fields it reads
  (waveform, `nfft`/`samples_per_chirp`, `propagation_speed`,
  `num_bins`); see `range_stage.hpp`.
- `fmcw_doppler_processor_op` — `fill_velocity_bins` is already MatX-native;
  cache its expression/output path keyed on waveform,
  `center_zero_doppler`, and `num_doppler`; see `doppler_stage.hpp`.
- `fmcw_range_doppler_processor_op` — the fused operator calls both builders,
  so it inherits both once the two above are cached.

Keep the existing `fill_*` logic as the cache-fill helper so behavior on a
miss is unchanged; only the steady-state (cache-hit) path becomes non-blocking.

## 3. MIMO marshaling — imposed assumptions (2026-07-16)

Cross-cutting decisions from landing the TDM + DDM marshaling operators. The
contract (`OPERATOR_API.md`) was updated where noted; anything still ambiguous
is pinned per operator:

| Operator | Open items |
| --- | --- |
| `MIMODeinterleaveOp` | none (uniformity / PRI / gather-axis pins live in the operator README; staggered TDM is STATUS.md L12 / Roadmap) |
| `MIMOCoherentCompensationOp` | none (TDM firing-slot map is `tx_schedule`; ÷M aliased velocity is the module pin below) |
| `VirtualArrayAssemblyOp` | none (flatten / overlap / geometry pins live in the operator README; `WireArrayGeometry` transport is HSDK5_PORT §10; Graph live unit skip is §1) |

**Module-level pins (not re-litigated per operator):**

- **Per-TX effective PRI.** `FMCWWaveformDescriptor.chirp_repetition_interval`
  is the slow-time sample period **per TX** under TDM (`M ×` physical
  inter-chirp spacing on the wire). Simulators and apps must not feed raw
  inter-chirp spacing to range/Doppler/compensation unless `M = 1`.
- **DDM clutter placement.** Slow-time clutter filtering belongs **after**
  DDM demux in the workflow (`WORKFLOWS.md`); not enforced by operators — a
  graph wiring concern.
- **Velocity ÷M ambiguity.** TDM and DDM reduce unambiguous velocity by `M`.
  Downstream sees the aliased per-TX Doppler axis. Disambiguation (staggered
  PRI, band tricks) is out of scope — not a `MIMOCoherentCompensationOp` mode.
- **Shared descriptors.** `DDMCodeDescriptor` and `TXSchedule` validation rules
  (mutually exclusive offset forms, uniform TDM schedule) live with the
  consuming operator README / descriptor code.

## 4. Add automatic C++ formatting checks

**Status:** open; the current pre-commit configuration runs `cpplint` but not
`clang-format`.

Add a checked-in `.clang-format` policy and a pinned `clang-format`
pre-commit hook, then run the same formatter check in CI. This would make
equivalent formatting choices deterministic across the operator sources (for
example, nested `Error{...}` initializer wrapping) instead of relying on
manual consistency. Editor format-on-save can use the same checked-in policy.

Before landing, choose and document the formatter version and verify the
initial repository-wide formatting diff is acceptable. Avoid mixing the
bootstrap formatting sweep with behavioral changes.
