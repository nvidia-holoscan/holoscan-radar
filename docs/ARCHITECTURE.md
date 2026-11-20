# Architecture

Public C++ headers, parameter schemas, and committed wire schemas are authoritative for declarations, types, and defaults. Operator READMEs define behavioral contracts and retain verified port and parameter summaries. Application READMEs own executable instructions and exact graphs. Review changes to any public surface together; do not silently resolve an unresolved product question in code.

## Shared invariants

1. **Plain-tensor data ports.** Operators exchange plain tensors on the data plane — never bespoke record/message types (e.g. no "detection" struct: CFAR emits `detection_idx`, `detection_score`, `noise_estimate` as separate tensors). Where producers and consumers need bridging, that is an explicit small add-on operator (`SnapshotGatherOp`, `FeatureAssemblyOp`, `DetectionReductionOp`), not a shared message type.
2. **Shared descriptors are bounded.** Exactly ten shared config-plane types exist: `ArrayGeometry`, `ShiftInvariantSubarrayPair`, `AngleConvention`, `WindowSpec`, `FMCWWaveformDescriptor`, `DDMCodeDescriptor`, `TXSchedule`, `CoordinateFrame`, `SampleGeometry`, `LFMWaveformDescriptor`. They are parameters, and ride input ports only as `[live]` overrides. The sanctioned descriptor *output* ports are `VirtualArrayAssemblyOp.array_geometry`, `TDMSparseCompensationOp.array_geometry`, `DDMSparseDemultiplexOp.array_geometry`, `SubarraySelectionOp.array_geometry`,
   `CPHDReaderOp.coordinate_frame`, `SARRangeCompressionOp.range_window_out`, `ApertureManagerOp.sample_geometry_out`, and `BackprojectionOp.sample_geometry_out`. Do not add new shared types or descriptor-carrying ports without a contract change to the public declarations and owner README. Separately, `MaxTensorExtents` is a **ctor-capture sizing type** for HSDK 5 Tensor pools (not a descriptor, never a Tensor port) — see [`operators/common/README.md`](../operators/common/README.md#maxtensorextents).
3. **Precision by compile-time template instantiation** (single + double). Never a runtime dtype flag, never inferred from data. Most operators take one parameter and every port on an instance matches it. The three SAR operators that carry geometry — `CPHDReaderOp`, `ApertureManagerOp`, `BackprojectionOp` — take `<T, TGeom>` instead, where `T` governs samples and images and `TGeom` the position and lattice ports, because ECEF coordinates cannot be carried in float32. `TGeom` defaults to `T` and may not be coarser than it. **Samples default to single in every application** (SAR takes `sample_precision: double` to opt back in): few radar systems deliver more. Both are fixed at instantiation; see the SAR operator READMEs for the precision of each port.
4. **Named trailing axes; no opaque leading batch.** Operate on the contract's named axes; infer those extents from the tensors each tick rather than taking them as parameters. Do **not** treat an open-ended leading `...` as an opaque module batch axis — multi-item throughput is an application concern (loop / fan-out). Named leading axes that are part of the contract (e.g. `N_rx`, `M_tx`, `num_az`/`num_el`) remain. Downstream never assumes fixed shapes for those named axes — that is what makes shape-changing `[live]` overrides legal. Fat Tensor outputs take `MaxTensorExtents` at `graph.op()` so pools can be sized before arm.
5. **`[live]` override convention.** Parameters are build-time config by default.
   A value qualifies as `[live]` only when it describes the tensor for this tick
   and an upstream producer can stamp the matching value onto that tick; making
   an encoding or marshaling parameter live at the consumer alone creates
   silent data/config desynchronization. A `[live]` parameter may additionally
   be overridden in-flight via a same-named input port. Every `[live]` port
   joins the data trigger with `Exact{}` and `on_missing SafeDefault{}`,
   declared unconditionally — wiring the port enables it. A match is the
   effective value for that tick only, a miss takes Params, and no latch is
   kept. Costly overrides (FFT plans, window coefficients, CFAR kernels) warn on expensive rebuilds and cache artifacts according to the owner README. Retained configuration caches use the bounded diagnostics described below. Structural parameters
   (e.g. `cfar_variant`, `angle_convention`, `multiplexing_scheme`) stay
   static — instantiate separate operators to vary them. Philosophy: don't
   over-constrain a caller who knows what they're doing; allow reasonable
   footguns and warn where relevant.
6. **Convention-free kernels, explicit angle boundary.** Kernels only dot direction vectors with element positions in the fixed canonical array-local frame. `AngleConvention` defaults to `radar_boresight` on every operator that consumes or emits angles; callers select another convention explicitly when needed. It is applied only as a boundary relabeling. `direction_cosine` (u, v) is the convention-free escape hatch.
7. **Industry-standard operator names** over internal schemes.
8. **A CUDA-capable NVIDIA GPU is a hard dependency** — for the operators, the apps, and the C++ test suite (decided 2026-07-10). Do not add CPU fallback compute paths, skip-if-no-GPU test guards, or GPU-optional build modes; host-only models, window generation, and control logic are deliberate, never fallback signal processing.

**Scope guardrail:** FMCW SISO/SIMO/MIMO-TDM/MIMO-DDM and the SAR image-formation port are in scope. SAR workflows are described in [WORKFLOWS.md](WORKFLOWS.md); do not add Pulse-Doppler or PVA operators; CDM/BPM is deferred to a future wave; FDM and PMCW are excluded outright.

## Temporal coherence and publication order

The frame's transmit time is the common event-time coordinate. TX publishes the applied configuration and acquisition trigger with one `capture_time`; RX copies that stamp to the returned ADC cube. Publish a matching config sample for every frame, even when its content is unchanged. Exact selection consumes a single-tick value; Params supply every miss. Arrival time, a generation number, and `max_age` cannot establish which waveform produced a cube.

A causal sideband must already be published when its `OnEach` driver arrives. Prepare both outputs, publish the sideband first, then the triggering Tensor. Equal timestamps alone are insufficient: publishing the Tensor first can resolve SafeDefault before the sideband arrives. This also applies to optional data sidebands such as `SnapshotGatherOp.is_valid`. `OnAligned` waits for all aligned members; independent diagnostics can follow the primary output. Preserve sample times through transforms and splitters.

Size non-dropping config queues for frames in flight: at least `1 + ceil(max publication-to-consumer latency / frame period)`. Retention is a capacity bound, not a time tolerance. Structural changes require recompiling the graph; mid-run parameter changes do not rewrite its contract. Recheck selector and retention semantics when changing SDK versions. Sustained skew retirement and drop-oldest control-plane behavior have not been qualified; do not rely on them as coherence mechanisms.

At a TX/RX driver boundary, applied TX time owns the stamp. Publish the waveform before marking a bounded frame-token record RX-ready. Early completions wait in bounded storage; duplicate claims, expiry, overflow, failed publication, and nonempty finalization fail closed. Tokens stay inside the coordinator and never become graph payloads.

## Accumulation and ownership

`ApertureManagerOp` owns aperture retention, signed evictions, and emit/reset control. `BackprojectionOp` owns the image accumulator and obeys that control. Exactly one component owns invalidation: independent resets would subtract contributions already discarded. The typed `stream_boundary` scalar is emitted on every reader batch (`0` none, `1` collection boundary, `2` end of stream) and is a required aligned manager input. A splitter puts the boundary on its final slice only. Flush the tail on the control path, never from `stop()`; stop has no output context. Missing control can leave pending accumulation un-emitted.

An emit-and-reset publishes the old image and its old geometry before accepting the next lattice and resetting. A failed required emit must preserve retry/reset ordering. Required formation outputs fail on full queues. CPHD unmetered ingest retries full required outputs; paced ingest retries only with static `wait_for_downstream: true`. Re-prepare Tensor loans after failed publication because the failed emit consumes the loan.

Never reuse a published buffer while a downstream consumer may retain it. Tensor pools are bounded by constructor-captured `MaxTensorExtents` before arm; actual named-axis sizes are read each tick. Ordinary data inputs must be contiguous and CUDA-accessible (device, managed, or pinned host). Control-vector exceptions are described below. Optional outputs use the shared latched-output helper; suppress work after an unwired port is discovered, and distinguish disconnection from backpressure.

## Retained configuration caches

Static `cache_budget_bytes` is an instance-wide byte ceiling across retained caches, reserves nothing upfront, and accepts zero to disable retention. Defaults and participating operators are listed in their headers and README parameter tables: most use 64 MiB, dense DDM uses 128 MiB, and SAR range compression uses 2 GiB. LRU eviction rebuilds a configuration when revisited; oversized artifacts still execute uncached with a bounded warning. Active GPU borrowers survive through completion. Eviction and uncached cleanup may synchronize; ordinary hits introduce no synchronization.

Accounting includes payloads, cached object sizes, retained key copies, and a conservative node/control-block allowance. SAR shape scratch and dense DDM tiles/direct cuFFT-reported workspace participate. This is not a peak-memory cap: builders, active borrows, output pools, allocator rounding, opaque metadata, single-current execution state, and MatX caches are excluded. Do not drain MatX caches. Diagnostic history is bounded per instance. FFT plans, CFAR calibration, and large manifold rebuilds can be expensive; budget both cold configurations and revisits after eviction.

## Application layering

App entry points express the graph and call the shared driver. Keep app-private config/custom operators beside the entry point; reusable Holoscan-aware helpers belong in `applications/common/radar_app`, pure simulation/model helpers in `operators/common/radar_common`. Each executable README includes a Mermaid graph matching its connections.

Operator YAML flows through native `from_config()` and parameter schemas. Apps parse app-owned sections only. An operator-field read solely to size visualization or geometry is allowed as a commented read-only mirror. Header-only shared helpers are inline. GUI configuration, presentation, lifecycle, and shutdown belong to the [shared application guide](../applications/common/README.md) and [visualization guide](../applications/common/radar_app/viz/README.md).

## Schema payloads and evolution

Use one of three physical representations:

1. generated FlatBuffers descriptor-Tensor schemas for shared owning
   descriptors;
2. ChannelPod payloads for bounded, fixed-layout scalar tuples; and
3. rank-1 pageable-host `Input<Tensor>` payloads for variable-length scalar
   vectors, with a per-port bound captured at operator construction.

This mixed policy is settled. It does not add a shared descriptor type or
permit bespoke record types on the data plane.

## Source and generated artifacts

Commit each `.fbs` file. It is the wire-contract source and must be reviewed
like a public API. Do not commit generated outputs: `*_generated.h`,
`*_bfbs_generated.h`, descriptor-traits headers, and other `flatc` products
belong in the build tree and are regenerated by
`holoscan_add_flatbuffer_schema`.

Each shared logical value has one schema owner and one canonical identity.
Do not copy a schema into individual operator directories: two copies can
mint incompatible identities for the same logical payload.

## Choosing the physical payload

Use a generated schema payload when all of these conditions hold:

1. the value is `[live]` and must cross an HSDK 5 typed port; and
2. its logical type is one of the shared owning descriptors; and
3. preserving the descriptor as one coherent value is preferable to exposing
   its members as unrelated ports.

The generated owning object carries descriptor metadata, while variable bulk
members use descriptor-Tensor fields. Metadata and bulk arrays then share one
sample identity and `capture_time`, without a second temporal join. Examples
include `WindowSpec`, `FMCWWaveformDescriptor`, and `ArrayGeometry`.

Use ChannelPod only for a bounded scalar tuple whose physical field sequence
and arity are fixed. It must satisfy the SDK's padding-free exact-tiling,
fixed-width-scalar, standard-layout, trivially-copyable, and no-external-
ownership rules. `SnapshotWindowOverride {doppler, range}` is the reference
shape. A logical Params field may remain a vector where YAML compatibility
requires it; the bridge validates and normalizes that logical value before
forming the fixed physical tuple.

Use a rank-1 pageable-host Tensor for a variable-length vector of one admitted
scalar dtype. The port is physically `Input<Tensor>`; its logical Params and
YAML type remains `std::vector<T>` (or the existing equivalent). Capture a
finite maximum element count in the operator constructor, reject rank, dtype,
memory-kind, and extent mismatches at the receive boundary, and copy or
convert into operator-owned state before returning from `compute()`. The
pageable-host requirement is a control-plane exception to the normal
CUDA-accessible data-Tensor rule; these small vectors are not bulk data-plane
inputs. Empty rank-1 Tensors represent empty vectors.

Do not use a schema merely because a standalone scalar vector owns memory.
Do not use ChannelPod for variable arity, pointers, strings, nested owning
objects, or Tensor members. Keep ordinary numeric bulk data on Tensor ports.

## Adding a schema

- [ ] Put the committed `.fbs` under the single shared
      `operators/common/radar_schemas/` owner; generated files stay in the
      build output.
- [ ] Record every generator argument: `SCHEMA`, `MAX_SERIALIZED_SIZE`
      (metadata bound; also enables traits / descriptor-Tensor inference),
      `BFBS_EMBED`, include roots, output directory, and the four-byte
      `file_identifier` on the `.fbs` root. The SDK infers root type, native
      type, package, canonical name, and descriptor-Tensor mode from the BFBS;
      do not pass the retired `DESCRIPTOR_TENSOR` / `TRAITS_*` CMake flags.
- [ ] Use the `holoscan.radar.wire` FlatBuffers namespace and verify the
      generated canonical name is the intended stable payload identity.
- [ ] Give every field an explicit id. Never infer compatibility from C++
      spelling or table layout.
- [ ] Bound every Tensor field from the applicable `MaxTensorExtents`; keep
      metadata serialization bounds separate from Tensor allocation bounds.
- [ ] Add logical-descriptor ↔ wire-type conversion at an explicit validated
      bridge. Reject malformed enums, dimensions, and descriptor invariants
      there.
- [ ] Add field-by-field round-trip tests, publication/receipt Graph tests,
      Tensor shape/dtype/memory-kind bound tests, oversize rejection tests,
      and drift guards across the logical descriptor, Params schema, `.fbs`,
      and generated object API.
- [ ] Install generated public headers with one include spelling that works
      in both build and install trees, and link every consumer to the one
      schema target.
- [ ] Update the owner README and the synchronization surface below.

## Evolution rules

Schema compatibility is exact identity: canonical name, schema version, and
schema hash must all match. Treat any `.fbs` edit as a versioned wire-contract
change and deploy producers and consumers together.

- Field ids are append-only. Never reuse an id.
- Deprecate fields; do not delete them.
- Additive fields still require the repository's compatibility review and
  coordinated producer/consumer deployment.
- A semantic change to an existing field requires a schema-version bump,
  even when its FlatBuffers storage type is unchanged.
- Keep old and new versions distinct during a transition; do not make two
  meanings share one identity.

## Tensor members and optional arrays

Empirical HSDK 5 result (2026-08-23): an ordinary descriptor-Tensor publish
rejects a null Tensor member with `kInvalidArgument`. A rank-1, zero-length
Tensor allocated and published with `emit_tensor` succeeds and arrives as a
present Tensor with `nbytes() == 0`.

Therefore an optional array is physically always a present Tensor. Rank 1
with extent zero is the absent representation; a nonzero extent is present.
Null is invalid. Marshaling must normalize absence to the zero-length Tensor
and must not assign semantic meaning to a missing pointer. Tensor dtype and
memory kind remain fixed by the schema field in both cases.

Encode each optional scalar losslessly as a value plus an explicit `has_*`
boolean. This keeps presence explicit across generated-language bindings and
avoids reserving an otherwise valid numeric value as an unset sentinel.

## Synchronization surface

Changing a schema-backed value requires one coordinated review of the full
surface:

- the logical descriptor or parameter type and its validation;
- `HOLOSCAN_PARAM_SCHEMA`, YAML/config conversion, and defaults;
- the committed `.fbs`, canonical name, version, explicit field ids, and
  generator arguments;
- logical ↔ generated-wire marshaling and optional-array normalization;
- Tensor field bounds derived from `MaxTensorExtents`;
- operator public typed ports and temporal contract;
- the shared schema target, generated-header include path, linkage, install,
  export, and package surface;
- C++ round-trip, drift-guard, allocation-bound, and Graph publication tests;
- operator READMEs and Python limitations;
- every producing and consuming application or control-plane component; and
- Python configuration and bindings documentation.

The generated wire object types are C++-only. Current Python operator support
cannot construct or publish them, so Python callers configure the equivalent
logical values through Params. That limitation does not change the logical
`[live]` contract and must be re-evaluated when the Python payload surface
gains schema-object support.

`VirtualArrayAssemblyOp.array_geometry` and
`SubarraySelectionOp.array_geometry` are runtime producers of
`WireArrayGeometry`. Their pageable-host descriptor-Tensor pools are bounded
from `MaxTensorExtents`; the internal `(N + 1, 3)` float64 packing remains an
implementation detail of the shared wire schema. C++ graphs can connect these
outputs directly to angle operators' same-schema Exact live inputs. Python
applications remain Params-only until generated payload ports are exposed.

## Implementation lifetime rules

Prepare every required sibling output and complete fallible fills before publishing any sibling, so a late fill failure cannot expose a partial output set. Keep loans and write guards alive through publication. Classify drivers, causal sidebands, aligned members and independent diagnostics before ordering outputs. Optional outputs with no dedicated computation can publish each tick (`always`); costly optional products use `latched` disposition to suppress dedicated work after disconnection is known. A declined optional output must not fail the primary tick.

Pointer-keyed library plans depend on the lifetime of their scratch addresses: retire the plans before releasing that scratch. Exercise repeated execution both alone and within a whole-suite process when stable-address scratch or persistent plans change. Host calibration remains outside fused GPU kernels; cache keys must include every value affecting the artifact. Never use asynchronous stale configuration serving unless an explicit reviewed contract permits it.
