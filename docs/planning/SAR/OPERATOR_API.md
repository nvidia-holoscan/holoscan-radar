# Holoscan Radar Module — SAR Operator API Lockdown

## Purpose

A **SAR-focused** operator API lockdown, the analogue of the FMCW [`OPERATOR_API.md`](../OPERATOR_API.md) one level up. Scope is derived from [`WORKFLOWS.md`](WORKFLOWS.md): the operators here are exactly those the stripmap and spotlight chains require, plus the standard-format I/O operators those chains tap. Each operator is specified by its **Inputs / Outputs / Parameters / Algorithm / Notes**.

Design conventions carry over unchanged: plain-tensor data ports (bridge with small explicit add-on operators, never shared record types); rank-agnostic batching, with leading axes treated as opaque batch dims and shapes inferred from the tensors each tick; industry-standard operator names; MatX implementation accepting tensors in any CUDA-accessible memory space (device, managed, pinned host — pageable-host input is rejected) and producing device memory.

Two conventions are **extended** and are specified here rather than assumed: precision (below) and cross-tick accumulation (§ *Cross-tick accumulation*).

**Precision (all operators).** Every operator supports both **single** (`complex64` / `float32`) and **double** (`complex128` / `float64`) precision floating-point I/O. Precision is fixed at compile time by the operator's template instantiation — neither a runtime parameter nor inferred from the data — and an instance's inputs and outputs match that precision. The per-operator dtype labels list both forms. SAR adds a second, **orthogonal** axis on `BackprojectionOp` alone: the internal arithmetic strategy for the range computation, which is not a binary and is not the same question as the I/O dtype. See `BackprojectionOp.precision_mode`.

**Proposed amendments.** Text marked **[PROPOSED]** is a working proposal, not accepted lockdown. Every open question in this document is enumerated in § *Open items* with its class, and the team-blocked ones are indexed in [`TEAM_BLOCKED.md`](../TEAM_BLOCKED.md) as R1–R5.

---

## Coordinate conventions

FMCW's cross-domain hazard is the azimuth/elevation parameterization. SAR's is the **Cartesian frame itself** — a different problem, which is why `AngleConvention` does not address it and a second descriptor is proposed.

The backprojection kernel computes, for every (pulse, sample point) pair, the norm of a position difference. That is invariant under rigid transforms, so the kernel is inherently frame-free: it returns an equally plausible answer whether positions and sample points are in ECEF, a local ENU frame, or scene-local metres — provided both are in the **same** frame. Nothing in the data distinguishes the frames; every value is metres and order-of-magnitude reasonable.

A frame mismatch therefore neither throws nor produces noise. It produces a defocused or geometrically displaced image, which is indistinguishable at first inspection from a poor collection, an autofocus failure, or a navigation error.

### Module stance

- **One declared Cartesian frame, right-handed, metres.** Phase-center positions, sample points, and any DEM-derived geometry are all expressed in it.
- **The frame is explicit at the API boundary** via a shared `CoordinateFrame` descriptor (see *Supporting data structures*), a **required parameter with no default** on every operator that consumes or emits positions or sample geometry. Never silently inherited — the same rule and rationale as `AngleConvention`.
- **It is a label, not a transform.** Operators do not convert between frames. The descriptor lets the boundary reject a mismatch and lets the writers record what the geometry means; frame-to-frame conversion is the caller's job, done upstream with standard geodesy.
- **The formation kernels ignore it.** Only boundary validation, `GeocodeOp`, and the writers read it.

### Frame landscape

| Frame | Origin | Axes | Typical use | Caveat |
| :--- | :--- | :--- | :--- | :--- |
| **ECEF** | Earth centre | Earth-fixed | spaceborne; CPHD/SICD native | rotates — positions must be at their own pulse times |
| **ENU** | geodetic anchor | East / North / Up | airborne; scene-local processing | flat-Earth error grows with scene extent |
| **NED** | geodetic anchor | North / East / Down | aerospace navigation | `+z` down; the sign trap FMCW's aerospace preset also carries |
| **scene-local** | scene reference point | caller-defined | simulation, unit tests | no geodetic meaning; writers cannot geolocate |

### High-risk silent mismatches the descriptor must prevent

- **ENU/NED z-sign flip:** `+z` up versus down places the platform below the scene, or inverts terrain height. Identical in kind to the FMCW aerospace-preset hazard.
- **ECEF versus local:** positions read from a CPHD file are ECEF; sample points built by a test fixture or a scene-planning tool are usually local. Both are metres, so the mismatch surfaces only as image quality.
- **Geodetic versus geocentric latitude** in `origin_llh`, which displaces a local-frame origin by up to ~20 km at mid-latitudes.
- **Phase-center versus platform-reference point:** not a frame error, but the same failure signature, which is why the `positions` port pins it explicitly.

---

## Workflows

Abbreviated from [`WORKFLOWS.md`](WORKFLOWS.md); see that doc for the full rationale and the port-level diagrams. `→` is a port connection, `[...]` is optional.

Modes in scope: **stripmap** (continuously advancing aperture — the streaming case that drives the design) and **spotlight** (bounded aperture). ScanSAR/TOPS and circular are deferred. Monostatic, single-pol, time-domain backprojection only; PFA, range-Doppler, chirp scaling and omega-k are named and deferred.

```text
CPHD file
  → CPHDReaderOp           → phase_history   (num_pulses, num_samples)
                           + positions       (num_pulses, 3)
                           + reference_range (num_pulses)
  → SARRangeCompressionOp  → range_profiles (num_pulses, num_range_bins)
  → BackprojectionOp       → image          (num_v, num_u)      [accumulating]
  → [AutofocusOp]          → image          + phase_correction
  → [MultilookOp]          → multilook
  → [GeocodeOp]            → product
```

Three ingress forms converge at **phase history**: recorded or live deramped data (via `RVPRemovalOp` + `BandEqualizationOp`), CPHD (directly), and direct-sampled pulses (via `RangeMatchedFilterOp`). Everything downstream of that node is shared.

Two chain inputs are **application responsibilities** with no library operator: per-pulse phase-center positions (the reader supplies them on the CPHD path; the application derives them from navigation on a live platform) and the image sample geometry.

---

## Operator inventory

The operators the SAR workflows require, grouped by function — a one-line map; full contracts are under *Operator APIs*. Build order and per-operator risk are in [`SIGPROC_PLAN.md`](SIGPROC_PLAN.md), deliberately not here: this document states the contract, not the plan.

### Ingress

- **`CPHDReaderOp`** — reads a CPHD collection into phase history plus the per-pulse geometry that rides with it. Core, not an optional tap: it is the ingress the simulator and real collections both use.
- **`CRSDReaderOp`** — the raw-side counterpart, for recorded uncompensated captures.
- A **live radar** source is app-supplied and sensor-specific — no library operator, following the FMCW `custom_ops.hpp` pattern. Its front end emits *either* deramped signal data or direct-sampled pulses; that choice selects the branch and is a property of the sensor, not a separate source.

### Signal-processing core

- **`SARRangeCompressionOp`** — range compression: windowed, optionally zero-padded FFT along fast-time; phase history → range profiles. The stage every branch reaches.
- **`RVPRemovalOp`** — deramped branch: removes the residual video phase term deramp-on-receive leaves behind, and undoes deramp reference bookkeeping.
- **`BandEqualizationOp`** — deramped branch: corrects amplitude and phase taper across the band, producing phase history proper.
- **`RangeMatchedFilterOp`** — direct-sampled branch: FFT along fast-time and multiply by the conjugate chirp-replica spectrum. Emits **phase history**, so the shared range compression supplies the inverse transform; a branch-entry stage symmetric with `RVPRemovalOp`, not a peer of range compression.

### Image formation

- **`BackprojectionOp`** — time-domain image formation. Accumulating across ticks; consumes range profiles, per-pulse phase-center positions, and sample geometry; emits the complex image.
- **`AutofocusOp`** — estimates and applies residual phase correction. A separate, omittable stage, not a mode inside image formation.

### Products

- **`MultilookOp`** — incoherent averaging of adjacent looks, trading resolution for speckle reduction.
- **`GeocodeOp`** — resamples to a map grid. Optional in a backprojection chain, since formation can target DEM-derived map points directly.

### Standard-format I/O

Library-provided sources and sinks at chain boundaries, **never inline stages**. Housed in their own `operators/io/` group so the XML and NITF dependencies sit behind their own `OP_<name>` gates.

- **`CPHDWriterOp`** — records phase history at the shared node. Paired with the reader, it gives a round-trip test.
- **`CRSDWriterOp`** — records uncompensated signal data near ingress.
- **`SICDWriterOp`** — writes the SLC image as a sensor-independent complex product.
- **`SIDDWriterOp`** — writes the multilooked or geocoded detected product. Not gated behind target detection: SIDD's "detected" means *amplitude* detection.
- **`TensorDumpOp`** — a generic tensor dump at any port boundary.

Fifteen operators in total: two ingress, four signal-processing, two image-formation, two products, five I/O.

### Two deliberate non-operators

Both of `BackprojectionOp`'s non-pulse inputs are the application's to produce — a pattern rather than two coincidences, since each is specific to the deployment in a way a library operator could only guess at.

- **Phase-center positions.** No `PhaseCenterTrajectoryOp`, and specifically no `MotionCompensationOp` — see [`WORKFLOWS.md`](WORKFLOWS.md) → *What "motion compensation" means here* for why only one of the three senses of that term survives, and why it is not a library stage. The obligation this creates falls on the `positions` port specification.
- **Sample geometry.** Grid mode is ordinary parameters; explicit points are a plain tensor on an optional port. Neither warrants a source operator.

### Explicitly out of scope for this SAR pass

Frequency-domain image formation (PFA, range-Doppler, chirp scaling, omega-k); interferometry and coherent change detection; polarimetry; ScanSAR/TOPS and circular collection; multi-channel GMTI; bistatic geometry; SAR detection. Each is deferred on **merit** — a distinct capability with its own contract surface. See [`WORKFLOWS.md`](WORKFLOWS.md) → *Out of scope* for the per-item rationale, and *Why FMCW detection does not carry over* for the clutter-model argument against reusing `CFARDetectorOp`.

---

## Cross-tick accumulation

A **new module-wide convention**, alongside the `[live]` override convention. The module has no stateful operators today; SAR forces the question, and it is written generally rather than as a backprojection special case. Two operators here are accumulating: `BackprojectionOp` (an image) and `CPHDWriterOp` (a collection).

An **accumulating operator** integrates across ticks and emits less often than it consumes. The convention:

- **Accumulation is declared in the contract.** An operator that accumulates says so, names its accumulated quantity, and states its reset semantics. Statefulness is never something a caller discovers from behaviour.
- **Emit and reset are independent axes.** *Emit-and-continue* keeps integrating after emitting (a sliding window); *emit-and-reset* clears the accumulator (a block aperture). Both must be expressible.
- **A geometry change forces a reset.** Partial sums are bound to the output geometry they were accumulated into, so any `[live]` override that changes that geometry invalidates them. The operator resets and warns rather than mixing incompatible partial sums.
- **The emitted buffer is not the accumulator.** An accumulator that keeps mutating cannot be handed downstream, so the emit path copies or double-buffers. This interacts directly with [`operators/OPEN_ITEMS.md`](../../../operators/OPEN_ITEMS.md) § 1 on output-buffer reuse.
- **Reset is observable and callable.** A caller must be able to reset deliberately, not only as a side effect of emitting.
- **Termination flushes.** On `stop()` an accumulator holding a partial result emits it (or discards it, per policy) rather than dropping it silently. A finite input — a file — otherwise loses its tail.

### [PROPOSED] How the application supplies the emit decision

Real applications want their own rule for when enough aperture has been traversed, and **the mechanism is team-blocked (R2)**. Three are viable, verified against the Holoscan 4.3 source:

- A **closed `emit_policy` enum** — pulse count, integration angle subtended, along-track aperture length, target cross-range resolution, external trigger. All but the last are derivable from the trajectory the operator already holds, and cross-range resolution is the term users specify requirements in. The only one of the three that is YAML-expressible, so the policy can live in config alongside every other knob.
- A **non-blocking `emit_now` control port** driven by an app-owned controller operator. This reuses the existing `[live]` override machinery (`add_override_port` in `radar_common/sdk/override_param.hpp`), so it introduces no new concepts and behaves identically from Python and C++. Its limit is that a controller can only decide on externally observable state — pulse count and geometry yes, partial-image sharpness no.
- A **`std::function` parameter** invoked inside `compute()`. Mechanically confirmed: `spec.param<T>()` constrains nothing, the setter's YAML branch is `if constexpr`-guarded, and `FeatureAssemblyOp` already hand-registers a setter for a native-only parameter type. Costs: invisible to `from_config()`, will not survive a distributed deployment, and a bare lambda fails to bind because `Arg` stores the exact static type.

Ruled out, recorded so they are not re-derived: `set_dynamic_flows` controls *routing*, not whether `compute()` does the work, so the expensive part still runs; `BooleanCondition` and `async_condition` gate whether the operator ticks at all, which inverts the intent of accumulating every tick and emitting occasionally.

Whether `BackprojectionOp` publishes its aperture state on an output port is the dependent sub-question (R3): that is what would let an external controller decide on operator-internal state.

---

## Operator APIs

Per-operator **Inputs**, **Outputs**, and **Parameters**. Every port carries a dtype + shape; every parameter carries a `Default:` (required shown as `Default: none (required)`) and is tagged **[live]** or **Static**. Each port lists both precision forms; an instance uses one, fixed by its template instantiation. Leading `...` axes are opaque, preserved batch dims.

**Parameter override convention.** Unchanged from the FMCW doc: parameters are build-time configuration; a **[live]** parameter may additionally be overridden in-flight via a same-named non-blocking input port; overrides are **sticky**; output shapes are re-inferred downstream, so a shape-changing override is legal; a costly override warns once per new configuration and caches prior ones; genuinely structural parameters are **Static** and vary only by instantiating separate operators. SAR adds one rule: **a [live] override that changes an accumulating operator's output geometry forces a reset** (see *Cross-tick accumulation*).

### Ingress Operators

#### CPHDReaderOp

Reads a CPHD collection and emits phase history together with the per-pulse geometry stored alongside it. A source operator: no data inputs.

- **Out:** `phase_history` — complex64 / complex128, `(num_pulses, num_samples)`. `num_pulses = pulses_per_tick`, except on the final tick where it is the remainder.
- **Out:** `positions` — float32 / float64, `(num_pulses, 3)`. Per-pulse Aperture Reference Point positions in `coordinate_frame`. CPHD stores these natively, which is why this path needs no trajectory computation.
- **Out:** `reference_range` — float32 / float64, `(num_pulses,)`. Per-pulse range from the phase centre to the point that phase-history sample zero is referenced against (the CPHD SRP range).
- **Out:** `range_bins` (optional) — float32 / float64, `(num_samples,)`. The range axis implied by the file's sample spacing, before range compression re-grids it. Emitted for the writers and for diagnostics.
- **Out:** `pulse_times` (optional) — float64, `(num_pulses,)`. Seconds from the collection epoch, always double regardless of the instantiated precision, since single precision cannot hold a collection-relative timestamp to useful accuracy.
- **Out:** `velocities` (optional) — float32 / float64, `(num_pulses, 3)`. Per-pulse phase-center velocity, where the file carries it.

**Parameters:**

- `path` — string. CPHD file path. Static. Default: none (required).
- `coordinate_frame` — CoordinateFrame. The frame the emitted positions are expressed in. Where the file declares a frame, a disagreement is an **error**, not a silent reinterpretation. Static. Default: none (required).
- `pulses_per_tick` — int. Batch size along the pulse axis; sets the granularity at which the accumulating formation stage sees the aperture. Static. Default: none (required).
- `channel` — int. Which CPHD channel to read, for multi-channel collections. Static. Default: 0.
- `pulse_range` (optional) — int, `(2,)`. Half-open `[first, last)` pulse range to read, for processing a subset of a long collection. Static. Default: none (whole collection).

**Algorithm:**

1. At `start()`, open the file, parse the XML metadata, and validate the declared frame against `coordinate_frame`.
2. Per tick, read the next `pulses_per_tick` pulses of the selected channel, apply the file's per-pulse amplitude scale factors, and upload to device memory.
3. Emit the pulse block with its geometry.

**Notes:**

- *Termination.* The operator emits until the selected pulse range is exhausted, then signals end-of-stream and stops the graph. The **final batch may be short** (`num_pulses < pulses_per_tick`), so every downstream operator must infer the pulse count from the tensor rather than the parameter. A short final batch is also what makes the accumulation convention's *termination flushes* rule load-bearing: without it, the tail of the aperture is silently dropped.
- *Ingest uploads.* As the first operator in the chain it reads host memory and produces device memory, which is the sanctioned exception to the module's CUDA-accessible-input rule (the same role the FMCW apps' ADC sources play).
- *Round-trip.* Pairing with `CPHDWriterOp` gives a read-write-compare test that validates field interpretation without an independent implementation to diff against.

#### CRSDReaderOp

Reads recorded uncompensated signal data, feeding the deramped branch. Same shape of contract as `CPHDReaderOp` — `path`, `coordinate_frame`, `pulses_per_tick`, `channel`, the same termination semantics — differing in that its `signal_data` output is **not** phase history: it still carries residual video phase and deramp reference bookkeeping, so it must pass through `RVPRemovalOp` and `BandEqualizationOp` first.

Full port list is deferred with the branch (see *Open items*, I2).

### Signal-Processing Operators

#### SARRangeCompressionOp

- **In:** `phase_history` — complex64 / complex128, `(..., num_pulses, num_samples)`. Compression along the last (fast-time) axis.
- **Out:** `range_profiles` — complex64 / complex128, `(..., num_pulses, num_range_bins)`. `num_range_bins = num_samples`, growing to `nfft` when set. No range-axis `fftshift`.
- **Out:** `range_bins` (optional) — float32 / float64, `(num_range_bins,)`. Per-bin range (m) relative to `reference_range`, uniform. Wire it into `BackprojectionOp.range_bins` to keep the two stages' range axes consistent by construction; emitted only when `waveform` is available.

**Parameters:**

- `range_window` **[live]** — WindowSpec, coherent-gain normalized. Window applied along fast-time before the transform; the dominant control on impulse-response sidelobes, and so on what a bright scatterer does to its neighbours. An unseen window rebuilds its coefficients and warns once; prior windows are cached. Default: none (rectangular — best resolution, ~−13 dB sidelobes).
- `waveform` **[live]** — LFM waveform descriptor. Source for the `range_bins` axis. Default: none (required when `range_bins` is used; unused otherwise).
- `fft_norm` **[live]** — enum {backward, ortho, forward}. FFT normalization; `backward` is the radar-standard, keeping coherent-integration gain visible. Default: backward.
- `nfft` **[live]** — int, ≥ `num_samples`. Fast-time transform length; larger zero-pads for a finer interpolated range grid over the same span, and is the escape hatch for a large-prime `num_samples`. The output bin count follows it (downstream infers). An unseen length builds a new plan and warns; seen lengths reuse the MatX plan cache. Default: none (= `num_samples`, exact N-point).
- `propagation_speed` **[live]** — float. Wave speed for the frequency → range conversion (`range_bins` only). Default: speed of light ≈ 2.998e8 m/s.

**Algorithm:**

1. Apply `range_window` along fast-time.
2. Zero-pad to `nfft` when set, and transform along fast-time under `fft_norm`.
3. Derive `range_bins` from `waveform` bandwidth, `nfft`, and `propagation_speed`.

**Notes:**

- *This is the digital stage, and hardware does not make it redundant.* Analog deramp-on-receive is bandwidth reduction, not compression; the transform stays digital and downstream, which is why CPHD phase history is stored pre-transform.
- *Window choice is a product decision, not a tuning knob.* It sets the impulse response the SICD product records and the resolution/sidelobe trade a downstream analyst sees, so it is one of the few parameters whose value belongs in the collection's metadata.

#### RangeMatchedFilterOp (direct-sampled branch)

- **In:** `pulses` — complex64 / complex128, `(..., num_pulses, num_samples)`. Direct-sampled returns at full bandwidth.
- **Out:** `phase_history` — complex64 / complex128, `(..., num_pulses, num_samples_out)`. Feeds the shared phase-history node. `num_samples_out = num_samples` under the default `mode = same`; under `mode = valid` it is `num_samples − replica_length + 1`, the fully-overlapped support.

**Parameters:**

- `waveform` **[live]** — LFM waveform descriptor. Supplies the transmitted chirp from which the replica spectrum is synthesized. Default: none (required unless `replica` is given).
- `replica` **[live]** — complex vector, `(replica_length,)`. An explicit measured replica, overriding the synthesized one; the measured form matters when the transmitter is non-ideal. Default: none (synthesize from `waveform`).
- `mode` — enum {same, valid}. Whether the output keeps the input sample count or is cropped to the fully-overlapped support. Static (fixes the output-length rule). Default: same.
- `nfft` **[live]** — int. Transform length for the multiply, ≥ `num_samples + replica_length − 1` to avoid circular wrap. Default: none (next fast length above that bound).

**Algorithm:** transform along fast-time, multiply by the conjugate replica spectrum, and emit without inverting — the shared range compression supplies the inverse transform.

**Notes:**

- *The apparent round trip is deliberate.* Direct-sampled pulses are time-domain, get matched-filtered into frequency-domain phase history, and are then transformed back to the range domain downstream. The alternative — a fused direct-sample-to-range-profile operator — saves one buffer and forfeits the standard-format interchange point at the phase-history node. The buffer is the cheaper of the two.
- *The name is provisional.* See *Open items*, I3: the literature calls the whole FFT-multiply-IFFT sequence "range compression", which leaves this truncated form without a settled industry name.

### Deramped-branch entry stages

`RVPRemovalOp` removes the residual video phase term that deramp-on-receive leaves behind and undoes any deramp reference bookkeeping — the per-pulse range-bin-to-absolute-range mapping a tracking reference point introduces. `BandEqualizationOp` corrects amplitude and phase taper across the band. Together they turn deramped signal data into phase history.

Full contracts are deferred with the branch (see *Open items*, I2). Both are **evidence-gated**: real raw data is what pins the sign and reference conventions, and getting those wrong is undetectable from synthetic data generated under the same assumptions.

### Image-Formation Operators

#### BackprojectionOp

Time-domain image formation. **Accumulating across ticks** — see *Cross-tick accumulation*.

- **In:** `range_profiles` — complex64 / complex128, `(..., num_pulses, num_range_bins)`.
- **In:** `positions` — float32 / float64, `(..., num_pulses, 3)`. Per-pulse antenna phase-center positions. **The specification of this port is load-bearing**: no library operator computes these (see *Two deliberate non-operators*), so the contract is the only thing standing between an application and a plausibly-degraded image. It pins:
  - **Frame** — the one declared by `coordinate_frame`; the same frame as `sample_points`.
  - **Units** — metres.
  - **Ordering** — row `i` corresponds to pulse `i` of `range_profiles`, with matching leading batch axes.
  - **Instant** — the phase centre at the pulse's **transmit** instant, under the stop-and-hop assumption. Receive instants or midpoints introduce a systematic along-track shift.
  - **Point** — the *antenna phase centre*, not the IMU and not the platform centre of mass. Lever-arm correction is the caller's.
- **In:** `reference_range` — float32 / float64, `(..., num_pulses)`. Per-pulse range from the phase centre to the point that range bin zero is referenced against.
- **In:** `range_bins` (optional) — float32 / float64, `(num_range_bins,)`. The range axis from `SARRangeCompressionOp`. When connected it **supersedes** `range_bin_spacing` and `range_bin_origin`, which are read from its first element and uniform step; this is the wiring that keeps the two stages' range axes consistent without the caller mirroring `nfft`. Non-uniform input is an error.
- **In:** `sample_points` (optional) — float32 / float64, `(num_points, 3)`. Explicit sample locations in `coordinate_frame`. Present only when `sample_geometry_mode = points`. **No leading batch axes**: the geometry is shared across the batch. Treated as configuration that happens to need a tensor, so a change forces an accumulator reset.
- **In:** `initial_image` (optional) — complex64 / complex128, matching the output shape. Seeds the accumulator, so a caller can resume or composite.
- **Out:** `image` — complex64 / complex128, `(..., num_v, num_u)` in grid mode or `(..., num_points)` in points mode. Leading batch axes are **preserved**, one independent accumulator per batch element (see Notes). Emitted per the aperture policy, not per tick.

**Parameters:**

- `coordinate_frame` — CoordinateFrame. Static — it fixes the meaning of every position in the graph. Default: none (required).
- `sample_geometry_mode` — enum {grid, points}. Where sample locations come from. Static: the modes differ in output **rank**, not merely shape. Default: grid.
- `grid_origin` **[live]** — float, `(3,)`. Position of sample `(0, 0)`. Grid mode. Changing it resets the accumulator. Default: none (required in grid mode).
- `grid_axis_u`, `grid_axis_v` **[live]** — float, `(3,)` each. Unit vectors along the two grid axes; they need not be axis-aligned or horizontal, which is what lets a caller form onto a slant plane or a tilted facet. Non-orthogonal axes are permitted (a sheared grid) but not normalized for the caller. Changing either resets the accumulator. Default: none (required in grid mode).
- `grid_spacing` **[live]** — float, `(2,)`. Metres per sample along `u` and `v`. Resets the accumulator. Default: none (required in grid mode).
- `grid_counts` **[live]** — int, `(2,)`. `(num_u, num_v)`; sets the output shape. Resets the accumulator and reallocates. Default: none (required in grid mode).
- `range_bin_spacing` **[live]** — float. Metres per range bin in `range_profiles`. Ignored when the `range_bins` port is connected. Default: none (required when that port is unconnected).
- `range_bin_origin` **[live]** — float. Range (m) of bin zero relative to `reference_range`. Ignored when `range_bins` is connected. Default: 0.
- `interpolation` **[live]** — enum {nearest, linear, sinc}. Range-profile interpolation at the computed range: `nearest` is a speed floor that costs impulse-response quality, `linear` the usual choice, `sinc` the accuracy ceiling. Default: linear.
- `sinc_taps` **[live]** — int, odd. Kernel length when `interpolation = sinc`. Default: 8.
- `precision_mode` — enum {fp32, mixed, float_float, taylor}. The **internal** arithmetic strategy for the range computation, orthogonal to the I/O dtype. `fp32` is fully single-precision; `mixed` computes ranges in FP64 and accumulates in FP32; `float_float` uses compensated double-single arithmetic; `taylor` expands about a per-block reference. Static (selects the kernel). Default: mixed.
- `emit_policy`, `reset_policy` — **[PROPOSED]**, team-blocked (R2). See *Cross-tick accumulation*; deliberately not specified here as though settled.

**Algorithm:**

1. Expand the sample geometry to a point list once (grid mode materializes it via a `radar_common` host helper), so both modes reach one kernel path.
2. For each sample point and each pulse in the batch, compute the two-way range from the phase-center position to the point under `precision_mode`.
3. Convert to a fractional range-bin index from `reference_range` and the active range axis, and interpolate the complex range profile there per `interpolation`.
4. Multiply by the matched phase term for that range and accumulate into the image.
5. Emit per the aperture policy, copying out of the accumulator; reset per the reset policy.

**Notes:**

- *Batch semantics.* Leading axes are preserved as independent images, each with its own accumulator — the rank-agnostic reading of invariant 4, and what a caller wants for independent sub-collections. To integrate **across** a leading axis instead, flatten it into the pulse axis upstream; the operator will not guess which is meant. Accumulator memory scales with the batch, so a wide batch of large images is a real footprint.
- *Precision is not a binary, and that is the substantive departure from invariant 3.* The right mode is platform-dependent, not a speed/accuracy preference. The third-order Taylor term is an airborne requirement rather than an optimization. Plain FP32 is decorrelated from truth at spaceborne ranges — the range magnitude is large enough that single precision cannot resolve a wavelength-scale phase difference — while often being adequate at a few kilometres.
- *Sliding windows are exact.* Backprojection is linear in pulses, so removing an outgoing block means re-forming and subtracting exactly that block. There is no accumulated drift to bound and no periodic re-form needed for correctness.
- *Geometry is not inferred from an input tensor*, the departure from invariant 4's second clause: the output shape comes from the caller's sample geometry. Legal here because that geometry is declared — parameters, or a port whose change forces a reset — so it is still never a hidden assumption.
- *Arbitrary points are the native interface*; a grid is what the caller materializes. That ordering matters because it is what makes forming directly onto DEM-derived map points a first-class case rather than a workaround.
- *Backend seam.* The kernel is expected to sit on `matx::experimental::sar_bp`, whose interface already covers arbitrary sample points, initial-image accumulation, and a comparable precision matrix. It is in the experimental namespace with an API subject to change, so the operator wraps it behind a seam rather than exposing it.

#### AutofocusOp

Estimates and applies a residual phase correction along the azimuth-conjugate axis. A separate, omittable stage: airborne collections generally need it, spaceborne collections with good ephemeris generally do not.

- **In:** `image` — complex64 / complex128, `(..., num_v, num_u)`. The formed SLC image.
- **Out:** `image` — complex64 / complex128, same shape. The corrected image.
- **Out:** `phase_correction` (optional) — float32 / float64, `(..., num_azimuth)`. The estimated phase along the azimuth-conjugate axis, where `num_azimuth` is the length of the image axis named by `azimuth_axis`. Emitted separately so it is composable, testable in isolation, and available to `SICDWriterOp` as metadata.
- **Out:** `sharpness` (optional) — float32 / float64, `(..., num_iterations + 1)`. The focus metric before each iteration and after the last, so a caller can see whether the estimate converged rather than inferring it from the image.

**Parameters:**

- `algorithm` — enum {pga}. Phase gradient autofocus. Static (selects the kernel). See [`WORKFLOWS.md`](WORKFLOWS.md) → *Autofocus algorithms* for the alternatives and why PGA is the default. Default: pga.
- `azimuth_axis` — enum {u, v}. Which image axis is the azimuth (slow-time-conjugate) axis. **Static** and **required with no default**: PGA is meaningless on the wrong axis, and unlike a frame mismatch this one is not even dimensionally checkable, so it is never inferred. Default: none (required).
- `num_iterations` **[live]** — int. Refinement iterations. Default: 4.
- `window_fraction` **[live]** — float in `(0, 1]`. Fraction of the azimuth extent retained in the first window; shrinks geometrically across iterations. Default: 0.5.
- `min_scatterers` **[live]** — int. Minimum strong scatterers required. Below it the estimate is not trustworthy: the operator passes the image through unchanged, emits a zero `phase_correction`, and warns. Default: 16.
- `max_correction` **[live]** — float (radians). Ceiling on the peak-to-peak estimated correction; exceeding it indicates a failed estimate rather than a badly defocused image, so the operator passes through and warns. Default: none (uncapped).

**Algorithm (PGA):**

1. Select the brightest scatterer per range bin, requiring at least `min_scatterers` above the local background.
2. Circularly shift each selected scatterer to the azimuth centre, removing its linear phase.
3. Apply a window of the current `window_fraction` about the centre.
4. Transform along `azimuth_axis` to the conjugate domain and estimate the phase gradient across azimuth samples (the linear-unbiased-estimator form).
5. Integrate the gradient to a phase, apply its conjugate, and shrink the window for the next iteration.
6. Accumulate the per-iteration corrections into `phase_correction`.

**Notes:**

- **[PROPOSED] — the grid-conjugacy requirement is team-blocked (R4).** PGA assumes the image's azimuth axis is the Fourier conjugate of slow-time. That holds for a polar or range-Doppler grid; it does **not** hold in general for an image backprojected onto an arbitrary ground grid or point set, so this operator cannot be appended after an arbitrary `BackprojectionOp` configuration. Three resolutions, for the team to pick rather than discover:
  - (a) restrict to grid mode with a declared azimuth-conjugate axis and reject other geometries — the narrowest, and what `azimuth_axis` already assumes;
  - (b) emit only `phase_correction` and re-form through `BackprojectionOp`, making autofocus a feedback loop rather than a downstream stage — the most faithful to the physics and the most expensive;
  - (c) implement a backprojection-domain autofocus operating on the actual sample geometry.
- *Why `phase_correction` is indexed by azimuth, not by pulse.* The image-domain estimate lives on the azimuth-conjugate axis, whose length is an image dimension. It equals `num_pulses` only when the image was formed on a conjugate grid at one azimuth sample per pulse. A general azimuth-to-pulse mapping is exactly what resolution (b) would have to define, which is why this port is not specified as `(num_pulses,)` today.
- *Points mode is rejected.* With `sample_geometry_mode = points` there is no azimuth axis to name, so `azimuth_axis` cannot be satisfied and the operator errors at build rather than producing a meaningless correction.

### Product Operators

#### MultilookOp

Incoherent averaging of adjacent looks. Trades resolution for radiometric quality: a single-look image is dominated by speckle, which is coherent interference rather than noise and does not average down within one look.

- **In:** `image` — complex64 / complex128, `(..., num_v, num_u)`.
- **Out:** `multilook` — float32 / float64, `(..., num_v_out, num_u_out)`. Real-valued intensity; phase is gone by construction, which is why this is a product stage and not an image-formation one. `num_v_out = floor(num_v / looks_v)`, likewise `u`.

**Parameters:**

- `looks` **[live]** — int, `(2,)`, each ≥ 1. Averaging factor along `(v, u)`. Speckle standard deviation falls roughly as `1/sqrt(looks_v · looks_u)`. Default: none (required).
- `remainder` — enum {truncate, error}. What to do when a dimension is not divisible by its look count: `truncate` drops the ragged final block, `error` rejects. Static. Default: truncate.
- `output_scale` **[live]** — enum {linear, amplitude, db}. Mean intensity, its square root, or decibels. Default: linear.
- `db_floor` **[live]** — float. Clamp applied before the log when `output_scale = db`, so that zero-valued samples do not produce `-inf`. Default: −100.

**Algorithm:** form `|image|²`, average over non-overlapping `looks_v × looks_u` blocks, handle the ragged edge per `remainder`, and convert per `output_scale`.

**Notes:**

- *Mean-preserving.* Averaging intensity (not amplitude) keeps the estimator unbiased in radiometry, which is what makes the output comparable across look counts.

#### GeocodeOp

Resamples to a map grid. **Optional in a backprojection chain** — see the note below.

- **In:** `multilook` — float32 / float64, `(..., num_v, num_u)`.
- **In:** `dem` (optional) — float32 / float64, `(num_dem_lat, num_dem_lon)`. Terrain heights on the output grid, or on a grid described by `dem_geotransform`.
- **Out:** `product` — float32 / float64, `(..., num_northing, num_easting)`. Resampled to the output grid; samples with no input coverage are filled with `no_data_value`.

**Parameters:**

- `coordinate_frame` — CoordinateFrame. The input geometry's frame; `origin_llh` is required on it here, since a `scene_local` frame without a geodetic anchor cannot be geolocated. Static. Default: none (required).
- `input_grid_origin`, `input_grid_axis_u`, `input_grid_axis_v`, `input_grid_spacing` — the formation geometry the input was produced on, mirroring `BackprojectionOp`'s grid parameters. Static. Default: none (required). *A read-only mirror of the formation stage's configuration — see the note on why this is not a shared descriptor.*
- `projection` — string. Output projection as an EPSG code or PROJ string. Static. Default: none (required).
- `output_origin` — float, `(2,)`. Northing/easting of the output sample `(0, 0)` in `projection`. Static. Default: none (required).
- `output_spacing` — float, `(2,)`. Metres per output sample. Static. Default: none (required).
- `output_counts` — int, `(2,)`. `(num_easting, num_northing)`. Static. Default: none (required).
- `interpolation` **[live]** — enum {nearest, bilinear}. Default: bilinear.
- `no_data_value` **[live]** — float. Fill for output samples outside the input footprint. Default: NaN.
- `dem_geotransform` (optional) — float, `(6,)`. Affine mapping from `dem` indices to projected coordinates, when the DEM is not already on the output grid. Static. Default: none (DEM assumed co-registered with the output grid).

**Notes:**

- *Optional, unlike in a frequency-domain chain.* Frequency-domain formation produces an image in radar geometry and resamples afterwards, which makes geocoding mandatory. Backprojection has no such constraint: it forms onto whatever sample points it is given, so DEM-derived points on a projected grid yield a geocoded image **directly**, with no resampling stage and no second interpolation loss. This operator serves the caller who wants a radar-geometry SLC *and* a map product from one formation pass.
- *Without a DEM*, geocoding assumes a reference surface and misplaces terrain by an amount that grows with relief and with shallower incidence.
- *Why the output grid is parameters, not a descriptor.* It would be one more shared type with exactly one consumer. The input-grid mirror is the uncomfortable part — it duplicates `BackprojectionOp`'s configuration — and if a second consumer ever appears, that duplication is the trigger to promote both to a shared grid descriptor. Recorded as an accepted wart, not an oversight.

### Standard-Format I/O Operators

Sinks and sources at chain boundaries, never inline. Each is optional; the chain is complete without any of them. All accept both precision instantiations and write the format's native sample representation, which is selected per writer rather than following the instantiated precision — a `complex128` graph may still write 16-bit integer CPHD samples.

#### CPHDWriterOp

**Accumulating** (see *Cross-tick accumulation*): a CPHD collection spans many pulse batches, so the writer buffers pulses and writes one file per collection rather than one per tick.

- **In:** `phase_history` — complex64 / complex128, `(..., num_pulses, num_samples)`.
- **In:** `positions` — float32 / float64, `(..., num_pulses, 3)`.
- **In:** `reference_range` — float32 / float64, `(..., num_pulses)`.
- **In:** `pulse_times` (optional) — float64, `(num_pulses,)`.
- **In:** `velocities` (optional) — float32 / float64, `(..., num_pulses, 3)`.
- No outputs — a sink.

**Parameters:**

- `path` — string. Output file path, or a pattern containing `{index}` when `max_pulses_per_file` is set. Static. Default: none (required).
- `coordinate_frame` — CoordinateFrame, with `origin_llh` required. Declares what the recorded geometry means. Static. Default: none (required).
- `waveform` — LFM waveform descriptor. Supplies the collection-level fields CPHD requires and the tensors do not carry. Static. Default: none (required).
- `sample_type` — enum {cf32, ci16, ci8}. CPHD sample representation. `ci16` with per-vector scale factors is the common recorded form and roughly halves file size against `cf32`. Static. Default: cf32.
- `max_pulses_per_file` (optional) — int. Roll to a new file after this many pulses, for collections too long for one file. Static. Default: none (single file).

**Notes:**

- *Buffering cost is real.* Phase history for a long collection does not fit in device memory, so the writer stages to host memory and appends to the file incrementally, keeping only the XML header pending until close.
- *Flush on stop.* Per the accumulation convention, `stop()` finalizes the header and closes the file. A crash mid-collection leaves an unfinalized file, which is the accepted failure mode.

#### SICDWriterOp

One file per emitted image — the image is the product, so there is nothing to accumulate.

- **In:** `image` — complex64 / complex128, `(..., num_v, num_u)`.
- **In:** `phase_correction` (optional) — float32 / float64, `(..., num_azimuth)`. Recorded where autofocus ran.
- No outputs — a sink.

**Parameters:**

- `path` — string. Output path pattern; `{index}` is substituted with the emitted-image counter. Static. Default: none (required).
- `coordinate_frame` — CoordinateFrame, with `origin_llh` required. Static. Default: none (required).
- `grid_origin`, `grid_axis_u`, `grid_axis_v`, `grid_spacing` — the formation geometry, needed for SICD's grid and geolocation fields. A read-only mirror of `BackprojectionOp`'s configuration. Static. Default: none (required).
- `waveform` — LFM waveform descriptor. Source for the collection and radiometric fields. Static. Default: none (required).
- `range_window`, `azimuth_window` — WindowSpec. The weightings applied upstream, which SICD records as the impulse-response description. Mirrors of the range-compression and formation configuration. Static. Default: none (rectangular).
- `sample_type` — enum {cf32, ci16, amp8phs8}. SICD pixel representation. Static. Default: cf32.

**Notes:**

- *Mirrored configuration is the metadata-plane symptom.* The four grid parameters and the two window parameters exist only because the operator must describe processing it did not perform, and each is a place where a caller can silently disagree with the upstream stage. That is the concrete cost of having no metadata plane, and the argument for R5.
- *NITF container.* SICD and SIDD are NITF-wrapped, so this operator brings that dependency into the `io` group.

#### TensorDumpOp

- **In:** `tensor` — any dtype, any shape.
- No outputs — a sink.

**Parameters:**

- `path` — string. Output path pattern; `{index}` is substituted with the tick counter. Static. Default: none (required).
- `format` — enum {npy, raw}. `npy` carries shape and dtype and is directly readable from the Python tests; `raw` is bytes only. Static. Default: npy.
- `every_n` **[live]** — int. Write only every nth message, for long runs. Default: 1.
- `max_files` (optional) — int. Stop writing after this many files and warn once. Static. Default: none (unlimited).

#### CRSDWriterOp and SIDDWriterOp

`CRSDWriterOp` records uncompensated signal data near ingress; `SIDDWriterOp` records the detected (amplitude) product after multilook or geocode. Their contracts follow `CPHDWriterOp` and `SICDWriterOp` respectively — accumulating pulse sink and per-product sink — and are deferred with the seams they attach to, neither of which exists in wave one. See *Open items*, I2.

---

## Supporting data structures

[`AGENTS.md`](../../../AGENTS.md) invariant 2 locks **exactly seven** shared config-plane descriptor types: `ArrayGeometry`, `ShiftInvariantSubarrayPair`, `AngleConvention`, `WindowSpec`, `FMCWWaveformDescriptor`, `DDMCodeDescriptor`, `TXSchedule`. SAR reuses `WindowSpec` unchanged, needs one addition, and proposes one rename. Both changes need review as contract changes (R1).

Six of those seven exist today in `operators/common/radar_common/descriptors/`. The seventh, `ShiftInvariantSubarrayPair`, arrives with `ESPRITEstimatorOp` — specified in the FMCW contract but not yet landed — which is why the FMCW [`OPERATOR_API.md`](../OPERATOR_API.md) → *Supporting data structures* documents six and carries that type only as ESPRIT's `subarray_spec` parameter. Invariant 2 states a **bound**, not an inventory, so seven is the number SAR's addition applies to.

### CoordinateFrame (new)

Declares the Cartesian frame that positions and sample geometry are expressed in. A **standalone, required parameter with no default** on every operator that consumes or emits positions or sample geometry. Applied only as a boundary declaration — the formation kernels never read it. Immutable value type, validated once at construction, following the `FMCWWaveformDescriptor` pattern.

- `frame` — enum {ecef, enu, ned, scene_local}. The frame family. Default: none (required).
- `origin_llh` (optional) — float, `(3,)`. Geodetic anchor as (latitude °, longitude °, height m above the ellipsoid) for the local frames. **Geodetic, not geocentric, latitude.** Required by `GeocodeOp` and every writer; ignored by the formation kernels. A `scene_local` frame without one has no geodetic meaning — legal for simulation and unit tests, rejected by the writers.
- `epoch` (optional) — float64, seconds since a stated reference. The time reference the frame is realized at; meaningful for `ecef` over long apertures, where the frame rotates relative to inertial space.
- `ellipsoid` — enum {wgs84}. The reference ellipsoid `origin_llh` is expressed against. Default: wgs84.

Two frames are **compatible** when `frame`, `origin_llh` and `ellipsoid` all match; an operator holding positions and sample geometry from incompatible frames rejects them at build time. That check is the descriptor's whole purpose.

Rationale for a new type rather than a field on an existing one: it is required by operators sharing no other descriptor (`BackprojectionOp` and `GeocodeOp` have almost nothing else in common), and it is exactly the `AngleConvention` pattern — a required, no-default, boundary-only declaration guarding a silent-failure mode.

### LFM waveform descriptor (proposed rename)

SAR needs the transmitted chirp for the range axis, the matched-filter replica, and the writers' collection fields. `FMCWWaveformDescriptor` already carries precisely the right fields — carrier frequency, bandwidth or slope, duration, sample rate, pulse repetition interval, and an optional measured reference — because FMCW and SAR both transmit **linear-FM chirps**. The type is FMCW only in its name.

**Proposal: rename it `LFMWaveformDescriptor`** and use it unchanged from both modalities, rather than adding an eighth type duplicating every field. That keeps the count at **eight** (the seven plus `CoordinateFrame`) and forecloses the drift two near-identical waveform types would invite. The cost is a mechanical rename across the FMCW operators. Alternatives: keep the FMCW name and use it from SAR (accurate fields, misleading name at every SAR call site), or add a parallel type (drift).

One field needs a SAR reading: `chirp_repetition_interval` is the pulse repetition interval, and for SAR it is the inter-pulse time along the aperture — the same quantity, but the FMCW doc describes it in terms of per-TX effective spacing for MIMO. The rename should generalize that wording.

### What stays out

- **Platform trajectory is data-plane**, not a descriptor: per-pulse and scaling with the batch, which is the definition of data.
- **Sample geometry is parameters** (grid) or a plain tensor (points).
- **The output map grid is `GeocodeOp` parameters.** It is the one place a shared grid descriptor looks tempting, and the input-grid mirroring across `GeocodeOp` and `SICDWriterOp` is the duplication that would justify it. Kept operator-local until a third consumer appears; see that operator's Notes.

---

## Contract changes against the locked invariants

Each is an explicit change to the invariants in [`AGENTS.md`](../../../AGENTS.md) and needs review as one. Collectively they are R1.

- **Invariant 2, bounded descriptors.** Seven becomes eight, plus a rename (`FMCWWaveformDescriptor` → `LFMWaveformDescriptor`). See *Supporting data structures*.
- **Invariant 3, precision by template instantiation.** Unchanged for I/O dtype. Extended by `BackprojectionOp.precision_mode`, an orthogonal internal-arithmetic selection with no FMCW analogue, because the right choice is platform-dependent rather than a preference.
- **Invariant 4, rank-agnostic batching.** Leading axes stay opaque and preserved, so the first clause holds. The second — infer shapes from the tensors each tick — does not: `BackprojectionOp` takes its output shape from the caller's declared sample geometry. Legal because that geometry is declared rather than assumed, and shape-changing overrides still force downstream re-inference.
- **Invariant 5, `[live]` overrides.** Adds one rule: a live override that changes an accumulating operator's output geometry forces a reset. Static-structural here: `coordinate_frame`, `sample_geometry_mode`, `precision_mode`, `algorithm`, `azimuth_axis`, `mode`, `remainder`, and every writer's format and geometry-mirror parameter.
- **A new convention.** Cross-tick accumulation — the module had none, and SAR cannot be specified without one.

---

## Open items

Every unresolved question in this document, in the two classes [`TEAM_BLOCKED.md`](../TEAM_BLOCKED.md) uses. Nothing here may be resolved in code.

**Team-blocked** — need a decision before the contract is accepted:

- **R1 — contract acceptance**, including the invariant amendments above.
- **R2 — the aperture emit-decision mechanism** for `BackprojectionOp`. Three verified-viable options; two ruled out with reasons.
- **R3 — whether `BackprojectionOp` publishes aperture state** on an output port. Dependent on R2.
- **R4 — `AutofocusOp` grid conjugacy.** Three resolutions; determines whether autofocus is a downstream stage, a feedback loop, or geometry-restricted.
- **R5 — whether a metadata plane is in scope.** Forced by the writers, whose mirrored geometry and window parameters are the symptom. Cross-cutting, so it belongs in the contract or is explicitly deferred, never discovered during implementation.

**Implementation-time** — settled by writing the code, not by a team decision:

- **I1 — `matx::experimental::sar_bp` release status.** Experimental namespace, API subject to change, and absent from the MatX the Holoscan SDK bundles here. Confirm before depending on it; do not bump MatX in this wave.
- **I2 — the deferred branches' full port lists:** `CRSDReaderOp`, `RVPRemovalOp`, `BandEqualizationOp`, `CRSDWriterOp`, `SIDDWriterOp`. Evidence-gated on real raw data and on the seams existing.
- **I3 — `RangeMatchedFilterOp`'s name.** The literature calls the whole FFT-multiply-IFFT sequence "range compression", leaving the truncated form unnamed. Settle against invariant 7 (industry-standard names) before the operator lands.
