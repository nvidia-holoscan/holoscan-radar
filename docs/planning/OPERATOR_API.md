# Holoscan Radar Module — FMCW Operator API Lockdown

## Purpose

An **FMCW-focused** operator API lockdown. Scope is derived from the FMCW workflows doc: the operators here are exactly those the SISO / SIMO / MIMO-TDM / MIMO-DDM chains require, plus the MIMO data-marshaling operators and the I/O bridges those chains lean on. Each operator is specified by its **Inputs / Outputs / Parameters / Algorithm / Notes**.

Design conventions (from the workflows doc): plain-tensor ports (bridge with
small explicit add-on operators, never shared record/struct message types);
precision fixed by compile-time template instantiation (single + double), not a
runtime flag or dtype inference; MatX implementation accepting data tensors in
any CUDA-accessible memory space (device, managed, pinned host — pageable-host
data input is rejected; amended 2026-07-10), with a narrow pageable-host
exception for rank-1 scalar-vector `[live]` payloads, and producing device
memory; **named contract axes with no opaque leading `...` batch** (multi-item
throughput is an application concern; see
[`MAX_TENSOR_EXTENTS.md`](MAX_TENSOR_EXTENTS.md)); industry-standard operator
names; and angle operators that make the az/el convention explicit at the API
boundary (standalone, defaulting to `radar_boresight`).

Fat Tensor outputs take `MaxTensorExtents` at `graph.op()` for HSDK 5 pool sizing.

**Precision (all operators).** Every operator supports both **single** (`complex64` / `float32`) and **double** (`complex128` / `float64`) precision floating-point I/O. Precision is fixed at compile time by the operator's template instantiation — it is neither a runtime parameter nor inferred from the data — and an operator's input and output tensors must match that templated precision. The per-operator dtype labels below list both forms (e.g. `complex64 / complex128`, `float32 / float64`); a given operator instance uses one, fixed by its instantiation, and its inputs and outputs share it. Any operator that must fix precision for a qualitative reason is called out in its own Notes.

**Proposed amendments.** Text marked **PROPOSED — needs team review (not locked)** (or tagged **[PROPOSED]**) is a working proposal only. It is not accepted lockdown until the team explicitly adopts it. No proposed-amendment blocks are currently open. (The `CFARDetectorOp` calibration / `alpha_lut` block was adopted as locked text on 2026-07-21 — see STATUS.md → Settled decisions, L5.)

---

## Coordinate conventions

The angle-estimation operators carry azimuth/elevation, which is the module's single biggest cross-domain interoperability hazard — radar, robotics, automated-driving, aerospace, and remote-sensing communities each define these angles differently. The key to a domain-agnostic API is recognizing that "coordinate frame" is really **two independent problems** that are routinely conflated:

1. **Cartesian frame transforms** — ECEF ↔ ECI ↔ ENU ↔ NED ↔ body/sensor. These are all Cartesian frames related by rigid transforms (a rotation, plus a translation for global origins). They are fully standardized by geodesy/navigation and **unambiguous**. **These are out of scope for the operators:** placing and orienting the array in a world frame is the caller's responsibility, done upstream with standard transforms. The operators never know about the Earth, datums, or platform attitude.

2. **The azimuth/elevation (AER) parameterization within a frame** — how a unit direction vector is named by two angles. This is **not** standardized across domains and is the real ambiguity: even after two parties agree on a Cartesian frame, navigation (azimuth clockwise from North, elevation up from the horizon) and physics (azimuth counter-clockwise from +x, polar angle from zenith) name the *same* direction with *different* angles. Standard libraries each hard-code one AER convention for their domain; a multi-domain library cannot.

### Module stance

- **Array-local Cartesian frame is the source of truth.** Operators are defined purely in the right-handed, meters frame set by `ArrayGeometry.element_positions`. The beamforming math only ever dots element positions with a look-direction unit vector, so the kernels are inherently convention-free.
- **The internal canonical parameterization is fixed** (boresight = +x, +y to the left, +z up; azimuth CCW from +x about +z; elevation up from the x-y plane). Direction cosines: `u = cos(el)·sin(az)`, `v = sin(el)`, `w = cos(el)·cos(az)`. All internal kernels operate in this one convention.
- **The AER parameterization is made explicit at the API boundary** via a shared `AngleConvention` descriptor (see Supporting data structures), reused by every angle operator and by anything that emits `(az, el)`.
  - It ships with named per-domain presets plus a fully-parameterized `custom` form, and is applied as a thin relabeling on the input scan grids and the output `doa` — never inside the kernels.
  - It is a **standalone parameter defaulting to `radar_boresight`** on operators that consume or emit angles; callers select another convention explicitly when needed.
  - Cartesian `element_positions` carry no convention, but the convention's axis references are interpreted in that same array-local frame.
  - Frame-to-frame transforms remain the caller's job; `AngleConvention` resolves only the angular naming. Angles are radians throughout.
- **A convention-free direction-cosine (u, v / sine-space) scan mode** is offered as an escape hatch: unambiguous, singularity-free (no degeneracy at the pole), uniform in beam spacing, and the natural form for the spatial-FFT fast path on uniform arrays.

### Cross-domain landscape

The basis for the `AngleConvention` presets. (Radar array conventions vary between texts; the row below is the common boresight-relative choice.)

| Domain | Standard | Cartesian frame (RH) | az = 0 | az sense | Elevation reference | Units |
| :----- | :------- | :------------------- | :----- | :------- | :------------------ | :---- |
| Robotics | ROS REP-103 | FLU body / ENU world | +x (forward / East) | CCW | from horizon, +z up | rad |
| Automated driving | ISO 8855 / ISO 23150 | FLU (legacy SAE J670 = FRD) | +x (forward) | CCW | from horizon, +z up | deg/rad |
| Aerospace / aviation | ISO 1151 / AIAA G-003 | FRD body / NED | +x (nose / North) | **CW** | from horizon, **up = −z** | deg |
| Satellite / SAR | WGS-84 / CEOS | ECEF / ECI + orbit frame | *n/a — look/squint* | cross-track / squint | from nadir (look angle) | deg |
| Radar / phased array | IEEE 686 / 149 | boresight +x, +y left, +z up | +x (boresight) | CCW | from horizon, +z up | u,v / deg |
| Geodesy / nav libs | pymap3d, MATLAB | ENU / NED | True North (+y ENU, +x NED) | **CW** | from horizon, up | deg |

### High-risk silent mismatches the presets must prevent

- **Geodetic azimuth (x↔y swap):** geodesy measures azimuth clockwise from North (+y in ENU), so `az = 0` means +y, not +x — a 90° rotation versus the math convention. Feeding math-convention angles into a geodetic mapper silently swaps the axes.
- **NED elevation sign flip:** in NED / FRD frames +z points *down*, so "positive elevation = up" maps to −z. Reusing the standard `z = R·sin(el)` puts the track below the platform.
- **Left/right flip:** FLU (ISO 8855, +y = left) versus legacy SAE J670 FRD (+y = right) silently mirrors azimuth — a detected obstacle ends up on the wrong side.
- **Look angle ≠ elevation:** SAR look / squint / grazing angles are measured from nadir / the zero-Doppler plane, not the horizon; they are not drop-in elevation values.

The takeaway for the API: pin the internal math to one fixed convention, and force every angle to cross the boundary through an explicit, domain-labeled `AngleConvention` (or direction cosines) rather than a bare `az` / `el` that silently assumes a frame.

---

## Workflows

The standard operator flow graphs the module must support, abbreviated from `holoscan-radar-fmcw-workflows.md` (see that doc for the full rationale). These are the target chains the locked operators have to compose into — the reason the inventory below exists. `→` is a port connection; `[...]` is optional; shapes follow each stage.

### Scope

| Configuration | TX × RX | TX multiplexing | Angle capability |
| :------------ | :------ | :-------------- | :--------------- |
| **SISO** | 1 × 1 | — | none (range-Doppler only) |
| **SIMO** | 1 × N | — | receive beamforming (physical RX array) |
| **MIMO — TDM** | M × N | time-division | virtual array (up to `M·N`) |
| **MIMO — DDM** | M × N | Doppler-division | virtual array (up to `M·N`) |

Modality acronyms (SISO / SIMO / MIMO) follow the wireless channel convention: **input** = transmit side, **output** = receive side — so SIMO is 1 TX × N RX. See `WORKFLOWS.md` *Modality naming* for the full table and rationale.

Out of scope here. **CDM / BPM** (code-division / binary-phase multiplexing) are **under consideration for a future wave** — architecturally a small extension of the DDM path (all TX on, per-chirp phase code, separated after the Doppler stage), reusing `TDMDenseCompensationOp / DDMDenseDemultiplexOp` in a code-correlation mode and the rest of the MIMO chain unchanged — but not implemented here. **FDM** (frequency-division) and **PMCW** (fast-time code division — a different waveform, not FMCW) are excluded.

### Cross-cutting structure

- **Two angle paths.** *Dense* — `PhaseShiftBeamformerOp` forms an angle for every RD cell → RDA cube → `CFARDetectorOp` on the cube. `PhaseShiftBeamformerOp` is the only dense estimator (forming a full MVDR/MUSIC cube over every RD cell is prohibitive), so the dense slot is hardcoded. *Sparse* — `CFARDetectorOp` detects first (non-coherently integrating the channel axis), then a per-detection channel snapshot is gathered and a sparse angle estimator resolves one DOA per detection.
- **The sparse angle estimator is a pluggable slot** (`⟨sparse angle estimator⟩` in the chains below), orthogonal to the antenna/multiplexing modality — whatever produced the snapshots (physical RX for SIMO, assembled virtual array for MIMO) feeds the same slot. Any estimator fills it: `BeamscanEstimatorOp` (conventional/Bartlett, the default), `MVDREstimatorOp`, `MUSICEstimatorOp`, `RootMUSICEstimatorOp`, or `ESPRITEstimatorOp`. The choice affects only the estimator's own I/O (single- vs. multi-snapshot input, source count, geometry constraint), not the surrounding graph — see the Angle-estimation enumeration.
- **Clustering placement.** `ClusterDBSCANOp` is optional and orthogonal to modality. When selected, it attaches at the detection/angle stage in one of two patterns: *angle-then-cluster* (default; cluster in range/velocity/az/el — the only clustering placement on the dense path) or, sparse path only, *cluster-then-angle* (cluster in range-Doppler, reduce to one representative per cluster, then estimate angle once per representative). A dense workflow may instead report N-D CFAR peaks directly.
- **The three bridges** (plain-tensor add-on operators, shown inline below). `SnapshotGatherOp`: RD maps + `detection_idx` → per-detection channel snapshot, feeding the sparse estimators. `FeatureAssemblyOp`: `detection_idx` (+ bin axes, `detection_score`, angle) → the `points` feature matrix that clustering consumes. `DetectionReductionOp`: labels + `detection_idx` + `points` → one representative detection per cluster, enabling cluster-then-angle.

### SISO (1 × 1)

Range-Doppler only; no spatial information, no angle stage.

```text
ADC (num_chirps, samples)
  → FMCWRangeProcessorOp     → (num_chirps, num_range)
  → [ClutterFilterOp]        → (num_chirps, num_range)
  → FMCWDopplerProcessorOp   → (num_doppler, num_range)     [RD map]
  → CFARDetectorOp           → mask / score / noise (+ detection_idx)
  → [FeatureAssemblyOp]      → points                       [range, velocity, score]
  → [ClusterDBSCANOp]        → labels (+ sizes)             [optional; range-Doppler clustering]
```

The fused `FMCWRangeDopplerProcessorOp` replaces the three signal-processing operators one-for-one. No marshaling operators; only trailing range-Doppler clustering applies (no cluster-then-angle).

### SIMO (1 × N)

The physical RX array *is* the channel axis; angle from receive beamforming. No TX multiplexing → no marshaling operators.

```text
ADC (N_rx, num_chirps, samples)
  → FMCWRangeProcessorOp     → (N_rx, num_chirps, num_range)
  → [ClutterFilterOp]        → (N_rx, num_chirps, num_range)
  → FMCWDopplerProcessorOp   → (N_rx, num_doppler, num_range)   [per-RX RD maps]
  → angle path (dense or sparse)
```

Dense angle path:

```text
  … (N_rx, num_doppler, num_range)
  → PhaseShiftBeamformerOp   → (num_az, num_el, num_doppler, num_range)   [RDA cube]
  → CFARDetectorOp (on cube) → detection_idx over the cube (+ score)
  → [FeatureAssemblyOp]      → points                          [range, velocity, az, el, score]
  → [ClusterDBSCANOp]        → labels (+ sizes)                [angle-then-cluster]
```

Sparse angle path — *angle-then-cluster*:

```text
  … (N_rx, num_doppler, num_range)
  → CFARDetectorOp (nci_axes = {rx})   → detection_idx (+ score)
  → SnapshotGatherOp                   → snapshot (num_points, N_rx, num_snapshots)
  → ⟨sparse angle estimator⟩           → aoa_estimates (num_points, 2)
  → [FeatureAssemblyOp → ClusterDBSCANOp]  → labels (+ sizes)  [range/vel/az/el]
```

Sparse angle path — *cluster-then-angle* (fewer beamform calls when detections are dense):

```text
  … (N_rx, num_doppler, num_range)
  → CFARDetectorOp (nci_axes = {rx})   → detection_idx (+ score)
  → FeatureAssemblyOp                  → points                [range, velocity, score]
  → ClusterDBSCANOp                    → labels                [range-Doppler clustering]
  → DetectionReductionOp               → rep_idx (1 cell per cluster)
  → SnapshotGatherOp                   → snapshot (num_clusters, N_rx, num_snapshots)
  → ⟨sparse angle estimator⟩           → aoa_estimates (num_clusters, 2)
```

### MIMO — TDM (M × N, time-division)

M TX fire in separate time slots, so their chirps interleave along slow-time. All three marshaling operators are needed.

```text
ADC (N_rx, M·L, samples)     [L = chirps per TX; slow-time interleaves the M TX]
  → MIMODeinterleaveOp         → (M_tx, N_rx, L, samples)              [TX becomes an explicit axis]
  → FMCWRangeProcessorOp       → (M_tx, N_rx, L, num_range)
  → [ClutterFilterOp]          → (M_tx, N_rx, L, num_range)
  → FMCWDopplerProcessorOp     → (M_tx, N_rx, num_doppler, num_range)
  → TDMDenseCompensationOp / DDMDenseDemultiplexOp → (M_tx, N_rx, num_doppler, num_range)  [per-Doppler-bin TX phase correction]
  → VirtualArrayAssemblyOp     → (num_virtual, num_doppler, num_range) + virtual element_positions
  → angle path (dense or sparse) + CFARDetectorOp
```

The fused `FMCWRangeDopplerProcessorOp` may replace the range/Doppler pair *after* de-interleave; compensation and assembly still follow it. Clustering attaches on the assembled virtual RD maps exactly as in SIMO.

### MIMO — DDM (M × N, Doppler-division)

M TX transmit simultaneously with per-chirp phase codes placing each TX in a distinct Doppler sub-band. No slow-time interleave → no de-interleave; TX separation is a post-Doppler band demux.

```text
ADC (N_rx, num_chirps, samples)   [all M TX present every chirp, per-chirp phase codes]
  → FMCWRangeProcessorOp       → (N_rx, num_chirps, num_range)
  → FMCWDopplerProcessorOp     → (N_rx, num_doppler, num_range)               [Doppler holds M TX bands]
  → DDMDenseDemultiplexOp    → (M_tx, N_rx, num_doppler_out, num_range)   [demux → TX axis]
       num_doppler_out = num_doppler_per_tx  (linear gather / doppler_offsets)
                       = num_doppler         (non-linear phase_codes correlate)
  → [DopplerNotchOp]         → (M_tx, N_rx, num_doppler_out, num_range)
  → VirtualArrayAssemblyOp     → (num_virtual, num_doppler_out, num_range) + virtual element_positions
  → angle path (dense or sparse) + CFARDetectorOp
```

No `MIMODeinterleaveOp` (nothing is interleaved in slow-time). DDM does not use `ClutterFilterOp`: coded pre-demux chirps do not admit its ordinary slow-time mean, while demultiplexed outputs are already Doppler-domain maps. `DopplerNotchOp` supplies optional physical-velocity suppression after dense demultiplexing. Folded sparse DDM remains unsupported because one folded bin represents multiple physical velocities. Clustering remains the same as SIMO/TDM.

### Workflow comparison

| Modality   | TX × RX | `MIMODeinterleaveOp` | Dense TX separation           | `VirtualArrayAssemblyOp` | Angle path      | Channel axis      |
| :--------- | :------ | :------------------- | :---------------------------- | :----------------------- | :-------------- | :---------------- |
| SISO       | 1 × 1   | —                    | —                             | —                        | none            | — (`1`)           |
| SIMO       | 1 × N   | —                    | —                             | —                        | dense or sparse | physical RX (`N`) |
| MIMO — TDM | M × N   | ✔ (front)            | ✔ (per-Doppler-bin phase)     | ✔ (`p_tx ⊕ p_rx`)        | dense or sparse | virtual (`≤ M·N`) |
| MIMO — DDM | M × N   | —                    | ✔ (band demux + code removal) | ✔ (`p_tx ⊕ p_rx`)        | dense or sparse | virtual (`≤ M·N`) |

The signal-processing core, `CFARDetectorOp`, and the angle operators are common to every angle-bearing workflow; virtual-array assembly is identical across TDM and DDM — only the TX-separation step differs. Clustering is not a column: it is orthogonal to the modality.

---

## Operator inventory

The operators the FMCW workflows require, grouped by function — a one-line map; full contracts are under *Operator APIs*. Where an operator's role differs across modalities, the workflows doc chains are the reference.

### Signal-processing core

Common to every workflow (SISO through MIMO). The split trio plus the fused drop-in.

- **FMCWRangeProcessorOp** — range compression: windowed FFT along fast-time, ADC cube → range profiles.
- **ClutterFilterOp** — static-clutter / zero-Doppler suppression along slow-time, between range and Doppler.
- **FMCWDopplerProcessorOp** — Doppler compression: windowed FFT along slow-time, range profiles → range-Doppler maps.
- **DopplerNotchOp** — optional physical-velocity stopband on an already formed RD tensor; hard null when its transition width is zero, otherwise a raised-cosine transition.
- **FMCWRangeDopplerProcessorOp** — fused range + optional clutter + Doppler in one graph node (drop-in for the split trio where the intermediate range profiles aren't needed; for MIMO, usable up to the point a marshaling step must interpose).

### Detection & clustering

- **CFARDetectorOp** — 2D CFAR on the (Doppler, range) plane; batches/integrates leading axes (`nci_axes`), so it serves both the dense (per RDA-cube slice) and sparse (non-coherent channel integration) paths.
- **ClusterDBSCANOp** — general N-D density clustering over a caller-assembled feature matrix; orthogonal to modality, placeable before or after angle estimation.

### Angle estimation

Orthogonal to the antenna/multiplexing modality: whatever produces channel snapshots (physical RX for SIMO, assembled virtual array for MIMO) plugs into the same slot. Dense path is Bartlett-only; the rest are sparse-path estimators.

- **PhaseShiftBeamformerOp** — conventional/Bartlett beamforming, **dense** path: forms an angle for every RD cell → RDA cube.
- **BeamscanEstimatorOp** — conventional/Bartlett, **sparse** path: one DOA per detection snapshot; the default sparse estimator.
- **MVDREstimatorOp** — adaptive/Capon beamforming; grid-scan; needs multi-snapshot covariance.
- **MUSICEstimatorOp** — subspace (spectral); grid-scan; needs covariance + source count.
- **RootMUSICEstimatorOp** — subspace, gridless (polynomial rooting); ULA (azimuth-only) **or** filled URA (native az+el, reduced-dimension per-axis rooting + pairing); no scan grid, no spectrum.
- **ESPRITEstimatorOp** — subspace, gridless (rotational invariance); shift-invariant geometry — ULA (azimuth-only) **or** filled URA (native az+el, dual-invariance 2-D ESPRIT with automatic pairing); closed-form DOAs.

#### Covariance estimator open items

The covariance-estimator open items are now **settled** (see STATUS.md): the source-count source (S2 — `num_sources` default `auto` = per-item EFT on the forward-backward-smoothed covariance eigenvalues, clamped to `[1, max_sources]`; fixed `int` fixes it; upstream-seed **rejected**, RMT/AIC deferred); the ULA/subarray-selection step for non-uniform virtual arrays (S3 — provided by `SubarraySelectionOp`, emitting a reduced `(snapshot, array_geometry)` pair); and the multi-snapshot / spatial-smoothing input form (S1 — forward / forward-backward sub-array averaging on a filled ULA/URA, per-aperture-axis sub-array counts).

### MIMO marshaling

Three operators, one per responsibility, spanning TX separation through virtual-array assembly.

- **MIMODeinterleaveOp** — TDM only: split the interleaved slow-time axis into an explicit TX axis (front of chain). Pure data marshaling.
- **TDMDenseCompensationOp / DDMDenseDemultiplexOp** — TX separation + phase-aware coherent compensation (TDM: per-Doppler-bin phase correction; DDM: band demux + code removal + alignment). The one genuine MIMO compute step.
- **VirtualArrayAssemblyOp** — flatten the separated `(tx, rx)` block into a virtual channel axis and emit the matching virtual `ArrayGeometry` (`p_virtual = p_tx + p_rx`, with overlap dedup).

### I/O bridges

The small add-on operators that connect detection, clustering, and the sparse angle estimators via plain-tensor ports. Each is filed under its functional group above; their contracts are in *Operator APIs*.

- **`SnapshotGatherOp`** — reads RD maps + `detection_idx` → per-detection channel **snapshot** matrix feeding the sparse angle estimators (sparse angle path; SIMO and both MIMO modalities).
- **`DetectionReductionOp`** — reads cluster labels + `detection_idx` (+ features) → one representative detection per cluster, for the cluster-then-angle (pre-angle) sparse path.
- **`FeatureAssemblyOp`** — maps `detection_idx` (+ bin axes, `detection_score`, angle) → the `points` feature matrix for `ClusterDBSCANOp` / `DetectionReductionOp`.

### Explicitly out of scope for this FMCW pass

**Not** required by the FMCW workflows, so excluded from this lockdown: Pulse-Doppler operators, SAR image-formation operators, and PVA SDK operators. Multiplexing scope (see Workflows → Scope): **CDM / BPM** are under consideration for a future wave (would reuse `TDMDenseCompensationOp / DDMDenseDemultiplexOp` in a code-correlation mode, rest of the MIMO chain unchanged); **FDM** and **PMCW** are excluded.

---

## Operator APIs

Per-operator **Inputs**, **Outputs**, and **Parameters**. Every port carries a dtype + shape; every parameter carries a `Default:` (required parameters shown as `Default: none (required)`). Algorithm and Notes are added when each operator is fully locked. Each port lists both precision forms (`complex64 / complex128`, `float32 / float64`); an operator instance uses one, fixed by its template instantiation, with inputs and outputs matching. Port shapes use **named** axes only (no opaque leading batch; see [`MAX_TENSOR_EXTENTS.md`](MAX_TENSOR_EXTENTS.md)).

**Parameter override convention.** Parameters are configuration set at graph build.

- A parameter tagged **[live]** may additionally be overridden in-flight via a same-named input port. The port is listed under Parameters (with the tag), never duplicated under Inputs; only true data tensors appear under Inputs.
- **Wiring is the application's only obligation.** Every `[live]` port's `Exact{}` + `on_missing SafeDefault{}` entry is in `contract()` unconditionally; an app enables the override by `add_flow`-ing the port and disables it by leaving it unwired. An unwired entry compiles (warning `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT`) and reports `kNotConnected` on every activation, which the operator reads as "no override this tick". There is no per-port static: the parameter is the whole configuration surface.
- **The rule holds under every trigger shape**, `OnEach` and `OnAligned` alike; `SnapshotGatherOp` and `DetectionReductionOp` are no longer special-cased. See [`HSDK5_PORT.md`](HSDK5_PORT.md) §11.
- Join is **Exact** against the operator's data trigger (`OnEach` on the cube / map). Capture times must match. The override for frame *k* must already be published when the trigger sample arrives (application causal order — typically the sequencer stamps transmit time on both). A same-tick sibling that has not yet published yields `kNeverSeen`.
- **The parameter is the default; an override lasts one tick.** Unwired → the Params/YAML value on every tick. Wired with an Exact match → that sample, **for that activation only**. Wired with a miss → the Params value again. Nothing is carried forward between ticks, and the operator keeps no last-value state. Never treat a SafeDefault default-constructed payload as an override.
- **To hold an override, publish it every tick.** This is what `Exact{}` already asks for — the sequencer stamps the override and the trigger sample from one transmit time — so per-tick publication is the normal shape, not an extra burden. Unmatched ticks still run (SafeDefault); apps needing fail-closed coherence handle the miss at the application.
- **Authoritative source exception.** A source operator may output one of the existing shared descriptors when the value is the configuration its device backend actually applied for that frame. It publishes the descriptor before releasing the matching data trigger and stamps both with the TX-applied capture time. This narrow control-plane boundary does not permit processing-operator descriptor echoes and does not add a shared descriptor type.
- The philosophy is to not over-constrain a caller who knows what they are doing: allow a reasonable set of footguns, and warn where relevant.
- Output shapes are inferred from the input tensors each tick (never assumed fixed downstream), so a **[live]** change that alters an output shape is allowed — consumers re-infer.

**Cost of a `[live]` override.** Most overrides are scalars an operator simply reads. Some select a *built artifact* — an FFT plan, window coefficients, a CFAR α LUT or window kernel, a steering manifold. Those go through `ConfigCache`, which warns once per unseen configuration and keeps prior ones, so re-publishing a value the operator has already seen is a hash lookup, not a rebuild.

Two usage patterns do degrade, and both are the application's to avoid:

- **A genuinely new value every tick** rebuilds the artifact every tick *and* grows the cache without bound, since it is keyed by configuration and never evicts. Sweeping a grid or a window length frame-by-frame is the case to watch; alternating among a few values is fine.
- **Publishing intermittently on a costly port** is worse than it looks now that overrides last one tick: the effective value alternates between the sample and the Params value, so the operator flips configuration on every gap. That is a correctness surprise first — the operator is not running what the last publish asked for — and only then a cost. Publish every tick, or leave the port unwired and set the parameter.

The costly ports, by operator: `nfft` and `range_window` / `doppler_window` on the FMCW range, Doppler, and fused range-Doppler operators (FFT plan + window coefficients); `waveform` on the same three (range/velocity axis rebuild); `pfa`, `threshold_scale`, `guard_cells` and `training_cells` on `CFARDetectorOp` (α LUT and window kernels); and `array_geometry`, `az_coordinates`, `el_coordinates` and `array_taper` on every scan-grid estimator — `BeamscanEstimatorOp`, `MVDREstimatorOp`, `MUSICEstimatorOp`, `PhaseShiftBeamformerOp` — where they rebuild the steering manifold, the largest of these artifacts.

- A few parameters are genuinely fixed for an instance's lifetime (e.g. the CFAR variant, which selects the kernel) and are **static**; instantiate separate operators to vary those.

**Optional output convention.** An output marked **(optional)** is one an application may leave unwired. Some are also *conditional* — published only on the ticks a producing condition holds (a waveform descriptor is configured, an `is_valid` input arrived); that is a separate property, noted per port.

- **Wiring is the application's only statement**, exactly as for `[live]` inputs. Nothing exposes an output's wiring to the operator — `Output` has no connection query and `contract()` has no output side — so the operator cannot ask, and publishing to an output with no consumer fails. The operator absorbs that failure: a declined optional output never fails the activation and never disturbs the tick's other outputs. Required outputs keep `HOLOSCAN_TRY`, where a failed emit *is* an error.
- Every optional output declares one of exactly **two dispositions**. The disposition is a property of the operator, not a parameter; an application does not select it, and both are equally safe to leave unwired. It decides only whether the operator keeps doing the work.
  - **`always`** — attempt the emit every tick the port is produced, discarding the result. Silent. Correct when the value is a byproduct of work done for another output, so there is nothing to save by stopping.
  - **`latched`** — on the first failed emit, warn once, then skip everything done **solely** to produce that output — its allocation, its fill kernels, its commit, its emit — for the remainder of the run. Correct when the port has dedicated work worth skipping.
- **Only `ErrorCode::kFailure` latches.** That is what an unconsumed output reports. Any other code is a genuine fault that says nothing about wiring, and is ignored for that tick without disabling the port — otherwise one transient fault would cost the whole run instead of one frame.
- Latching is sound because **wiring is fixed for a run**: the plan is compiled before the first activation and cannot be rewired while it runs.
- Where dedicated work is shared by two optional ports (`doa_multi` and `source_count` come from one peak-extraction pass), it stops only once **both** have latched — until then the pass runs and needs somewhere to write, so both buffers are still allocated. Each port always gates its own emit.

**HSDK 5 physical `[live]` payload policy.** The parameter bullets below name
the logical Params/YAML types. Those types do not change to satisfy
`PublishedPayload`; only the same-named port's physical representation changes.

- Shared owning descriptors use generated `holoscan::radar::wire`
  FlatBuffers descriptor-Tensor schemas. The logical operator values remain
  the corresponding shared descriptors.
- Bounded fixed-layout scalar tuples use padding-free ChannelPod wrappers.
  The logical Params type remains the type documented by the operator.
- Variable-length scalar vectors use rank-1 pageable-host `Input<Tensor>`
  payloads. Their logical Params remain `std::vector<T>` or the documented
  equivalent. Each such port has a finite maximum element count captured at
  operator construction; rank, dtype, pageable-host memory kind, and extent
  are validated before conversion to the logical value.

These control-plane Tensors are the sole pageable-host exception to the data
Tensor memory-space rule. The physical payload does not alter Exact /
SafeDefault joining, wiring, validation, defaults, or
shape-change semantics. Schema identity/evolution, generated-artifact, EA
generator-workaround, and C++-only limitations are locked in
[`SCHEMA_PAYLOADS.md`](SCHEMA_PAYLOADS.md).

**Bridge operators.** The workflows also use small add-on operators to connect detection, clustering, and the sparse angle estimators. Per the plain-tensor-ports convention, bridging is done by explicit add-on operators (never shared structs). Each carries a **delivery-model** decision — a mainstream library operator, a per-application implementation, or a shared helper called inside another operator's `compute()` — recorded per bridge. Three exist — `SnapshotGatherOp`, `DetectionReductionOp`, and `FeatureAssemblyOp` — grouped into the functional sections below alongside the operators they connect (each flagged in its own Notes).

---

### Signal Processing Operators

#### FMCWRangeProcessorOp

- **In:** `adc_cube` — complex64 / complex128 (IQ IF) **or** float32 / float64 (real IF), `(num_chirps, samples_per_chirp)`. Range FFT along the last (fast-time) axis.
- **Out:** `range_profiles` — complex64 / complex128, `(num_chirps, num_range_bins)`. `num_range_bins = samples_per_chirp` (complex IF) or `samples_per_chirp//2 + 1` (real IF); grows to `nfft` when set. No range-axis `fftshift` (bin 0 = zero range).
- **Out:** `range_bins` (optional, **latched**) — float32 / float64, `(num_range_bins,)`. Per-bin range (m) relative to zero beat; emitted only when a waveform descriptor is available.

The logical `[live]` types remain `WindowSpec` and
`FMCWWaveformDescriptor`. Their HSDK 5 physical port payloads are generated
`holoscan::radar::wire` schema types with metadata plus a descriptor Tensor
bulk field. The Tensor is always present: rank-1 zero-length means the
optional array is absent; null is invalid. See
[`SCHEMA_PAYLOADS.md`](SCHEMA_PAYLOADS.md).

**Parameters:**

- `range_window` **[live]** — WindowSpec {none, Hann, Hamming, Blackman, Blackman-Harris, Kaiser, Chebyshev, Taylor, flat_top, Tukey, custom}, coherent-gain normalized. Window applied along fast-time before the range FFT. Overriding to an unseen window rebuilds its coefficients and warns once; prior windows are cached. Default: none (rectangular — best resolution, ~−13 dB sidelobes).
- `apply_dechirp` **[live]** — bool. Conjugate-multiply the input against the reference chirp along fast-time before windowing/FFT (for data not dechirped in hardware). Requires a `waveform` and complex-IF input when `true` — software dechirp produces complex samples, which is inconsistent with the real-IF `samples_per_chirp//2 + 1` output size, so real-IF input with `apply_dechirp = true` is an error. Default: `false`.
- `waveform` **[live]** — FMCWWaveformDescriptor. Shared chirp descriptor; source for the dechirp reference and the `range_bins` axis. Overriding supports a waveform that changes between frames (the fused reference is cached, rebuilt only when the descriptor changes). Default: none (required when `apply_dechirp = true` and/or `range_bins` is used; unused otherwise).
- `fft_norm` **[live]** — enum {backward, ortho, forward}. FFT normalization; `backward` (unnormalized forward) is the radar-standard, keeping coherent-integration gain visible. Default: backward.
- `propagation_speed` — float. Wave speed for the beat-frequency → range conversion (`range_bins` only). Default: speed of light ≈ 2.998e8 m/s.
- `nfft` **[live]** — int, ≥ `samples_per_chirp`. Fast-time FFT length; larger zero-pads for a finer interpolated range grid over the same span (also the fast-length escape hatch for a large-prime `samples_per_chirp`). The output range-bin count follows `nfft` (downstream infers it). Overriding to an unseen length builds a new FFT plan and warns; seen lengths are reused from the MatX plan cache. Default: none (= `samples_per_chirp`, exact N-point).

#### ClutterFilterOp

- **In:** `range_profiles` — complex64 / complex128, `(num_chirps, num_range_bins)` or with named leading `N_rx` / `(M_tx, N_rx)` (no opaque leading batch). Slow-time (chirp) axis second-to-last.
- **In:** `doppler_window` (optional, **[live]**) — logical `WindowSpec`; the HSDK 5 physical payload is the generated `holoscan::radar::wire::WindowSpecT`. Same-named one-tick override joined to `range_profiles` with `Exact{}` + `SafeDefault{}`; a match applies only to that tick, while a miss uses Params without latching.
- **Out:** `range_profiles_out` — complex64 / complex128, same leading + `(num_chirps_out, num_range_bins)`. `num_chirps_out = num_chirps` (`none` / `mean_subtraction`), `num_chirps − 1` (`single_delay`), `num_chirps − 2` (`double_delay`). `none` materializes an identity copy into the operator-owned output pool; it does not alias the input. HSDK 5 requires unique port names within an operator (inputs and outputs share one namespace), so the physical output name is `range_profiles_out` rather than repeating `range_profiles` ([EA HSDK5 ONLY] — same pattern will apply to other operators that previously reused an input name on the output).
- **Ctor:** `MaxTensorExtents` required (`max_chirps`, `max_range_bins`; optional `max_rx` / `max_tx` / `max_channels` admit named leading axes).

**Parameters:**

- `method` **[live]** — enum {none, mean_subtraction, single_delay, double_delay}. `none` is a materialized pass-through; `mean_subtraction` subtracts the slow-time mean weighted by `doppler_window` and has no chirp loss; the delay cancellers are 1st/2nd-order MTI high-pass (wider near-DC notch, and drop 1–2 chirps). Switching to/from a delay canceller changes `num_chirps_out`; the downstream Doppler stage infers the chirp count, so this is allowed. No setup cost. Default: mean_subtraction.
- `doppler_window` **[live]** — WindowSpec. For `mean_subtraction`, taps `w[k]` define `mu_w = sum(w[k] * x[k]) / sum(w[k])`, and the output is `x[k] - mu_w`; `none` uses the existing uniform mean exactly. The chirp count is unchanged. `none`, `single_delay`, and `double_delay` ignore this parameter. Unseen configured windows rebuild and cache their taps with a warn-once notice. Default: none.

#### FMCWDopplerProcessorOp

- **In:** `range_profiles` — complex64 / complex128, `(num_chirps, num_range_bins)`. Doppler FFT along the `num_chirps` (slow-time) axis.
- **Out:** `rd_maps` — complex64 / complex128, `(num_doppler, num_range_bins)`. `num_doppler = num_chirps` (or `nfft`); zero-Doppler centered by default.
- **Out:** `velocity_bins` (optional, **latched**) — float32 / float64, `(num_doppler,)`. Per-bin radial velocity; emitted only when a descriptor is available.

The logical `[live]` types remain `WindowSpec` and
`FMCWWaveformDescriptor`. Their HSDK 5 physical port payloads are generated
`holoscan::radar::wire` schema types with metadata plus a descriptor Tensor
bulk field. The Tensor is always present: rank-1 zero-length means the
optional array is absent; null is invalid. See
[`SCHEMA_PAYLOADS.md`](SCHEMA_PAYLOADS.md).

**Parameters:**

- `doppler_window` **[live]** — WindowSpec. Window along the chirp (slow-time) axis; sidelobe control trades against zero-Doppler clutter leaking into moving-target bins. Overriding to an unseen window rebuilds its coefficients and warns once; prior windows cached. Default: none.
- `center_zero_doppler` **[live]** — bool. `fftshift` the Doppler axis so zero-Doppler is centered (conventional RD layout); else DC at bin 0. Default: `true`.
- `waveform` **[live]** — FMCWWaveformDescriptor. Source for the `velocity_bins` axis (carrier → λ, chirp repetition interval → PRF). Override for a changing waveform. Default: none (required only when `velocity_bins` is used).
- `fft_norm` **[live]** — enum {backward, ortho, forward}. Same convention and rationale as FMCWRangeProcessorOp. Default: backward.
- `nfft` **[live]** — int, ≥ `num_chirps`. Doppler FFT length; larger interpolates the velocity grid over the same span (also the large-prime escape hatch). The output Doppler-bin count follows `nfft` (downstream infers it). Overriding to an unseen length builds a new FFT plan and warns; seen lengths reused from the MatX plan cache. Default: none (= `num_chirps`, exact N-point).

#### DopplerNotchOp

- **In:** `rd_maps` — complex64 / complex128, one of the named shapes `(num_doppler, num_range_bins)`, `(N_rx, num_doppler, num_range_bins)`, or `(M_tx, N_rx, num_doppler, num_range_bins)`. Doppler is the second-to-last axis; no opaque leading batch is accepted.
- **In:** `velocity_bins` — float32 / float64 matching the templated precision, `(num_doppler,)`. Required and aligned exactly with `rd_maps`; values are finite, strictly increasing physical radial velocities in m/s.
- **Out:** `rd_maps_out` — same dtype and shape as `rd_maps`. Every named channel receives the same Doppler gain, preserving relative channel phase.
- **Ctor:** `MaxTensorExtents` required (`max_doppler_bins`, `max_range_bins`; optional `max_rx` / `max_tx` admit the named leading axes).

The operator acts only on an unfolded physical velocity axis. It must not consume `DDMPowerFoldOp` output: one folded bin represents multiple physical velocities. Omit the operator from a graph for pass-through; there is no `none` mode. No mask output exists in v1; whether CFAR needs a generic exclusion mask is deferred in the operator-local open items.

**Parameters:**

- `center_velocity` **[live]** — float, center of the notch in m/s. Cyclic distance is measured over the uniformly spaced aliased velocity interval. Default: `0.0`.
- `stop_half_width` **[live]** — non-negative float in m/s. Bins at or inside this distance receive zero gain; the nearest bin to `center_velocity` is always included so zero width remains a one-bin hard null. Default: `0.0`.
- `transition_width` **[live]** — non-negative float in m/s. `0.0` gives a hard notch. Positive values apply a raised-cosine gain from zero at the stopband edge to one at `stop_half_width + transition_width`. Default: `0.0`.
- `cyclic_velocity_axis` — bool, static. When `true`, distance wraps over `num_doppler × bin_spacing`; when `false`, distance is ordinary absolute velocity difference. Default: `true`.

#### FMCWRangeDopplerProcessorOp (fused)

- **In:** `adc_cube` — complex64 / complex128 (IQ IF) **or** float32 / float64 (real IF), `(num_chirps, samples_per_chirp)`.
- **Out:** `rd_maps` — complex64 / complex128, `(num_doppler, num_range_bins)`. `num_range_bins` per FMCWRangeProcessorOp, `num_doppler = num_chirps` (reduced by the delay-canceller chirp loss when `clutter_filter` is a delay canceller). No intermediate range-profile output; the inline clutter filter is fused into the Doppler pre-FFT expression (no materialized intermediate), bit-identical to `ClutterFilterOp` → `FMCWDopplerProcessorOp`.
- **Out:** `range_bins` (optional, **latched**) — float32 / float64, `(num_range_bins,)`.
- **Out:** `velocity_bins` (optional, **latched**) — float32 / float64, `(num_doppler,)`.

The logical `[live]` types remain `WindowSpec` and
`FMCWWaveformDescriptor`. Their HSDK 5 physical port payloads are generated
`holoscan::radar::wire` schema types with metadata plus a descriptor Tensor
bulk field. The Tensor is always present: rank-1 zero-length means the
optional array is absent; null is invalid. See
[`SCHEMA_PAYLOADS.md`](SCHEMA_PAYLOADS.md).

**Parameters:** (fused; union of the split operators, same semantics)

- `range_window` **[live]** — WindowSpec. Window along fast-time (range FFT). Overriding to an unseen window rebuilds its coefficients and warns once; cached. Default: none.
- `doppler_window` **[live]** — WindowSpec. Window along the chirp axis (Doppler FFT). When `clutter_filter = mean_subtraction`, this same resolved window also weights the clutter projection before it is applied as the Doppler analysis window. This intentionally corrects the previous fused behavior, which used a uniform projection even with a tapered analysis window. A split chain must configure matching `ClutterFilterOp.doppler_window` and `FMCWDopplerProcessorOp.doppler_window` values for equivalent weighted projection and analysis. Overriding to an unseen window rebuilds its coefficients and warns once; cached. Default: none.
- `apply_dechirp` **[live]** — bool. Software dechirp before the range FFT. Requires a `waveform` and complex-IF input when `true` (same constraint as FMCWRangeProcessorOp). Default: `false`.
- `clutter_filter` **[live]** — enum {none, mean_subtraction, single_delay, double_delay}. Optional inline clutter stage between the range and Doppler stages (same semantics as ClutterFilterOp `method`); `mean_subtraction` uses the resolved `doppler_window`, while the other methods ignore it for clutter projection. The delay cancellers reduce `num_doppler` (downstream infers it). No setup cost. Default: none.
- `waveform` **[live]** — FMCWWaveformDescriptor. Dechirp reference and the `range_bins` / `velocity_bins` axes. Override for a changing waveform. Default: none (required for software dechirp and/or those outputs).
- `fft_norm` **[live]** — enum {backward, ortho, forward}. One normalization convention for both transforms. Default: backward.
- `center_zero_doppler` **[live]** — bool. Center zero-Doppler on the Doppler axis. Default: `true`.
- `propagation_speed` — float. For the `range_bins` output. Default: speed of light ≈ 2.998e8 m/s.
- `nfft_range`, `nfft_doppler` **[live]** — int, each ≥ its natural length. Per-axis FFT lengths (interpolation / large-prime escape hatch); the output bin counts follow them (downstream infers). Overriding to an unseen length builds a new FFT plan and warns; seen lengths reused from the plan cache. Default: none (= exact `samples_per_chirp` / `num_chirps`).

---

### Detection & Clustering Operators

#### CFARDetectorOp

- **In:** `rd_maps` — complex64 / complex128, **or** a real float32 / float64 power map, `(num_doppler, num_range_bins)`. CFAR plane = last two axes; leading axes batched/integrated per `nci_axes`. The path is auto-selected by input dtype (no parameter, mirroring the range processor): **complex** → form the square-law power `|·|²`; **real** → treat as an already-formed power map and threshold it directly.
- **Out:** `detection_mask` (optional, **latched**) — uint8 (`0`/`1`), `(num_doppler, num_range_bins)` (axes in `nci_axes` reduced; other leading axes preserved). `1` where detected — equivalently `detection_score > 1`. When `peak_mode` selects NMS, the mask instead marks only retained local maxima (a strict subset of `detection_score > 1`), so it stays consistent with `detection_idx`; `detection_score` / `noise_estimate` remain dense. Emitted as `uint8` rather than `bool` because the HSDK 4.3 and current HSDK 5 EA NumPy bridges have no typestr for `kDLBool`; `arr.view(bool)` recovers a boolean view.
- **Out:** `detection_score` (optional, **latched**) — float32 / float64, same shape as `detection_mask`. The CFAR test statistic normalized by its per-cell adaptive threshold: `detection_score = P / (threshold_scale · α · noise_estimate)`, where `P` is the cell's power statistic (`|·|²`, or the NCI sum). `> 1` ⇔ detected; the value is the detection margin, and since the threshold tracks local noise it is proportional to the cell SNR — so it doubles as the per-detection **strength / ranking score** used downstream (gathered by feature-assembly into `points` for `ClusterDBSCANOp` / `DetectionReductionOp`). Replaces the former `detection_threshold` output (the absolute threshold is recoverable from `noise_estimate` and the params).
- **Out:** `noise_estimate` (optional, **latched**) — float32 / float64, same shape as `detection_mask`. Per-cell noise from the training cells (`N`); kept for absolute calibration / diagnostics.
- **Out:** `cfar_power` (optional, **latched**) — float32 / float64, same shape as `detection_mask`. The exact surface the threshold is applied to: the post-NCI integrated power `Σ_k |x_k|²` over the `nci_axes` fold, or the plain square-law power map `|x|²` when `nci_axes` is empty. **Power, not amplitude, and not noise-normalized** — that is `detection_score`. Exists so a consumer can display or analyze what the detector actually saw instead of re-deriving it from `rd_maps` (a re-derivation silently diverges the moment `nci_axes` / `nci_fold` change). HSDK 5 copies it from the operator's ping-ponged workspace into a pool loan when the output is active.
- **Out:** `detection_idx` (optional, **latched**) — int, `(num_detections, 2 [+ n_batch])` — `(doppler_bin, range_bin)` per row (widening to include any preserved leading batch axes, or emitted per batch slice). Bare indices only. When active, it is sized to the actual detection count and reallocated per tick — the row count *is* `num_detections`, so there is no companion scalar. If `max_detections` is set and the scene exceeds it, the surplus is dropped and a warning is raised.

**Parameters:**

- `cfar_variant` — enum {CA, OS, GO, SO}. Noise-estimation rule: CA (cell-averaging), GO (greatest-of), SO (smallest-of), OS (order-statistic). **Static** — selects the kernel; to run different variants, instantiate separate detectors. Default: CA.
- `os_rank` **[live]** — int. Order-statistic rank into the sorted training cells; used only when `cfar_variant = OS`. **0-based, ascending:** training-cell powers are sorted low→high and the noise estimate is the value at index `os_rank` (rank 0 = smallest power). Must be `≥ 0` and `<` the training-cell count; border cells with fewer training cells clamp to `train_cnt − 1`. Default: none, represented in Params and bindings by the `-1` unset sentinel; OS rejects that sentinel with a required-parameter error. An explicit rank is required when OS; ≈ `floor(0.75·N)` for `N` training cells is typical.
- `guard_cells` **[live]** — int per axis `(doppler, range)`. Guard extent each side of the cell under test (excluded from the noise estimate). The detection math is a single fused stencil that reads the extents as kernel arguments, so an unseen window footprint needs no kernel rebuild (it may still trigger an α-LUT recalibration — see `pfa` / `alpha_lut`). Default: none (required).
- `training_cells` **[live]** — int per axis `(doppler, range)`. Training / reference extent each side, for the noise estimate. Also updates the training-cell count `N` (and thus α). Like `guard_cells`, changing the footprint needs no kernel rebuild but may recalibrate the α LUT. Default: none (required).
- `pfa` **[live]** — float. Probability of false alarm; sets the per-variant threshold multiplier α **when `alpha_lut` is unset**. For CA-CFAR the exact relation for the Gamma(`K`) integrated statistic over `N` training cells is `pfa = I_{1/(1 + α/N)}(N·K, K)` (regularized incomplete beta), inverted numerically for α; at `K = 1` this reduces in closed form to `α = N·(pfa^(−1/N) − 1)`. **CA, and OS with `K = 1`,** calibrate α by closed-form inversion (cheap). **GO, SO, and OS with `K > 1`** have no closed form and calibrate α by Monte Carlo on a LUT-key cache miss (seconds on large footprints) — supply a precomputed `alpha_lut`, or rely on `blocking_alpha_lut_rebuild = false` to serve a stale LUT while the new table builds. Default: 1e-6.
- `alpha_lut` **[live]** — optional float vector. Precomputed threshold multiplier LUT. When non-empty, used directly and `pfa` is ignored for α calibration. Indexing matches the runtime stencil:
  - **CA / OS:** `alpha_lut[n]` where `n` is the in-bounds training-cell count at the cell under test.
  - **GO / SO:** `alpha_lut[i]`, `i = go_lut_index(range_neg, range_pos, doppler_neg, doppler_pos)` (the GOGO/SOSO stencil's side-count key, `cfar_alpha.hpp`): per axis, `min(neg, pos)` if both half-sides alive else the alive side's count; then `min(range_side, doppler_side)` if both axes alive else the alive axis. Border geometries can alias to one `i`; `pfa`-derived builds store the max α across them so every aliased cell meets the target Pfa. Offline tables must use this `go_lut_index`, not raw per-side length or CA-style `N`.

  Slot 0 is unused. Length must exceed the maximum LUT index for the configured variant, footprint, and map size (`len > max_n`). CA and OS with `K = 1` require every index in `1..max_n` to be finite and strictly positive. GO/SO and OS with `K > 1` may leave zero only at indices unreachable for the current map and window geometry; every reachable `go_lut_index` (GO/SO) or training-cell count `N` (OS) must be finite and strictly positive. Negative or non-finite entries are always rejected, including unreachable slots; slot 0 and entries past `max_n` are ignored. Cheap: validation and host→device upload only. Default: `[]` (unset → `pfa`-derived calibration). Precedence: `alpha_lut` if provided, else `pfa`-derived calibration.
- `blocking_alpha_lut_rebuild` — bool. When `true`, a `pfa`-derived LUT cache miss blocks the tick until calibration finishes. When `false` (default), an unseen key starts a background build and the previous LUT is served for the tick (warn once); if no previous LUT exists yet, the tick still blocks on the first build. Has no effect while an external `alpha_lut` is supplied. Static. Default: `false`.
- `preintegrated_looks` — int. Number of statistically independent power looks already summed in a real-valued input sample before CFAR. The effective Gamma shape is `K = preintegrated_looks · ∏ sizes of nci_axes`; complex input with no leading-axis NCI therefore keeps `K = 1` at the default. This is an independent-look model: correlated RX noise, shared-LO noise, or clutter residue reduce the effective degrees of freedom and require empirical calibration (or an external `alpha_lut`). Static. Default: 1.
- `nci_fold` — int. Optional leading-axis integration-fold hint `∏ sizes of nci_axes`, used to pre-calibrate the α LUT at `start()` when `nci_axes` is set (the fold is otherwise unknown until the first tick). The pre-warm uses `K = preintegrated_looks · nci_fold`. `<= 0` → unset (defer calibration to the first tick). Static. Default: 0.
- **Start-time pre-calibration (CA / OS).** At `start()` the operator pre-builds the `pfa`-derived α LUT so the first tick does not stall on a Monte Carlo build. **CA and OS only:** their LUTs are indexed by training-cell count `N`, map-size-independent while the RD map is ≥ the CFAR window, so the warm build keys on a canonical window-sized geometry (no RD-map shape needed). **GO / SO** are indexed by the aliased `go_lut_index` and are map-dependent (a window-sized map is non-convergent at low `pfa`), so they are **not** pre-warmed — they calibrate on the first tick (supply `alpha_lut`). A sub-window CA/OS map, an NCI fold not covered by `nci_fold`, or a `[live]` override falls back to a runtime build. **Any runtime (in-`compute`) calibration warns once per configuration.**
- `threshold_scale` **[live]** — float. Empirical multiplier on top of the active α source (`alpha_lut` or `pfa`-derived): `threshold = threshold_scale · α · noise_estimate`; `1.0` = pure CFAR. Cheap; hand-tuning without changing the CFAR model. Default: 1.0.
- `nci_axes` **[live]** — set/vector of leading-axis indices to non-coherently integrate before thresholding; other leading axes stay batched. Changes the output shape (integrated axes reduced — downstream infers) and contributes `∏ sizes of nci_axes` to the effective Gamma shape `K = preintegrated_looks · ∏ sizes of nci_axes`; α takes `K` and is recomputed from `pfa` (closed form for CA; Monte Carlo for GO/SO/OS when `K > 1` and `alpha_lut` is unset). Default: `[]` (no additional integration; `K = preintegrated_looks`). Validated by Monte Carlo (`test_cfar_detector_statistical` — `CaMonteCarloPfaK1`, `CaMonteCarloPfaNciK8`, and pre-integrated-power cases).
- `max_detections` (optional) **[live]** — int. Upper cap on the number of detections emitted in `detection_idx`; if the scene produces more, the surplus is dropped and a warning raised. `detection_idx` is always sized to the actual count (≤ this cap, reallocated per tick). Unset → no cap (emit all detections). Default: none (uncapped).
- `peak_mode` (optional) — enum {none, range_doppler, all}. **Static.** Peak selection over detected neighbors using raw `cfar_power`: `none` emits every above-threshold cell; `range_doppler` performs NMS independently in each leading plane; `all` performs N-D NMS over every active output axis. Exact ties retain the lowest full row-major linear index. Only `detection_idx` / `detection_mask` are reduced; dense score, noise, and power outputs are unchanged. Default: none.
- `peak_radius` (optional) — int64 vector. **Static.** Non-negative half-width in bins for the full hyperrectangular NMS neighborhood, clipped at tensor boundaries. `range_doppler` accepts one value (broadcast) or `(doppler, range)`. `all` accepts one value (broadcast) or exactly one value per **post-NCI output axis**, in output order; this is checked against the effective shape each tick, so a live `nci_axes` rank change may invalidate a rank-specific radius. Zero excludes an axis, and an NMS mode requires at least one active axis. Default: `[1]`.

**Deferred calibration fast paths (out of scope for v1).** `alpha_lut` is the v1 explicit fast path (same per-index threshold semantics as `pfa`-derived calibration). The following remain deferred and are **not** part of the current contract:

- an interior-cell scalar `alpha` for homogeneous maps where border exactness is not required;
- scalar `alpha` plus simplified edge-handling modes, such as cropping the CFAR-valid region, treating border cells as non-detections, or accepting approximate border thresholds;
- hybrid scalar/table modes, such as using scalar `alpha` for fully in-bounds interior cells and a small border LUT for reduced-`N` edge cells;
- hybrid scalar/policy modes, such as scalar `alpha` for the interior plus an explicit edge policy that crops, suppresses, approximates, or delegates border handling to a downstream stage;
- helper utilities that generate `alpha_lut` tables offline from `(variant, pfa, K, os_rank, guard_cells, training_cells)`;
- domain-specific table packaging or deployment conventions outside the operator API.

GO/SO start-time pre-calibration (needs an RD-map shape hint, since their LUTs are map-dependent) is a tracked future enhancement, not part of the v1 contract.

#### ClusterDBSCANOp

- **In:** `points` — float32 / float64, `(num_points, num_features)`. Caller-assembled feature matrix (e.g. `[range, velocity, az, el]`); radar-agnostic.
- **Out:** `labels` — int, `(num_points,)`. Cluster id per point, noise = `-1`; row-aligned with `points`. Clusters are numbered by **descending point count** (cluster `0` is the largest), ties broken by ascending component root (lowest member row index). A **border point** reachable from cores in more than one cluster joins the core neighbor with the **numerically smallest component root** (lowest member row index) — the deterministic, order-independent tie-break for DBSCAN's inherent inter-cluster-border ambiguity. (The partition — core structure and noise set — is identical under any faithful DBSCAN; only which cluster a straddling border point joins is pinned here.) Deterministic: `labels` is a pure function of `(points, epsilon, min_points)`.
- **Out:** `sizes` (optional, **latched**) — int, `(num_clusters,)`. Point count per cluster, in the same descending order as the ids (so `sizes` is non-increasing); sized to the actual cluster count and reallocated per tick. If `max_clusters` is set and exceeded, the smallest surplus clusters are dropped (their points → noise) and a warning raised.

**Parameters:**

- `epsilon` **[live]** — float vector, length `1` (isotropic) or `num_features` (per-axis). A per-axis vector defines `Σ_i ((p_i − q_i) / epsilon_i)² ≤ 1` over entries with `epsilon_i > 0`. A per-axis `0` ignores that feature while preserving the full `points` row for downstream consumers. Values must be finite and non-negative with at least one positive entry; scalar `0` is invalid. Write the isotropic case as a one-element list (`epsilon: [0.5]`); a positive length-1 vector broadcasts to all axes. Overridable. Default: none (required).
- `min_points` **[live]** — int. Minimum neighbors (including the point itself) for a core point (DBSCAN `minPts`). Overridable. Default: `2 · active_features` (common heuristic, excluding zero-epsilon axes).
- `max_clusters` (optional) **[live]** — int. Upper cap on the number of clusters retained in `labels` and, when connected, reported in `sizes`; if exceeded, the **smallest** clusters are dropped (their points relabeled to noise `-1`, keeping `labels`/`sizes` mutually consistent) and a warning raised. Because ids are numbered by descending size, the kept clusters are the `max_clusters` largest. `sizes` is sized to the actual cluster count (≤ this cap). Unset → no cap. Default: none (uncapped).

#### FeatureAssemblyOp

Assembles the `points` feature matrix for clustering (and `DetectionReductionOp`) by gathering each detection's physical attributes from its cell — mapping CFAR's bare bin indices to physical range / velocity, gathering the detection strength, and appending angle when available. Pure gather / lookup — the feature-engineering step between detection and clustering.

- **In:** `detection_idx` — int, `(num_points, 2 [+ n_batch])`. The `(doppler_bin, range_bin)` cells (CFAR). Row order sets the output row order (so `ClusterDBSCANOp.labels` join back positionally).
- **In:** `range_bins` (optional port; required by `range`) — float32 / float64, `(num_range,)`. Physical range (m) per range bin (range-processor `range_bins`; in-tree: `FMCWRangeProcessorOp`); indexed by the `range_bin` column. Select `range_bin` instead when the raw index is wanted.
- **In:** `velocity_bins` (optional port; required by `velocity`) — float32 / float64, `(num_doppler,)`. Physical velocity (m/s) per Doppler bin (Doppler-processor `velocity_bins`; in-tree: `FMCWDopplerProcessorOp`); indexed by `doppler_bin`. Select `doppler_bin` instead when the raw index is wanted.
- **In:** `detection_score` (optional) — float32 / float64, `(num_doppler, num_range)`. CFAR's per-cell score map; gathered at each detection cell → the strength feature.
- **In:** `aoa_estimates` (optional) — float32 / float64, `(num_points, 2)`. Per-detection `(az, el)` **values** (sparse post-angle path), row-aligned with `detection_idx`; supplies `az_estimate` / `el_estimate`.
- **In:** `az_grid` (optional) — float32 / float64, `(num_az,)`. First scan-coordinate grid; indexed by the preserved `az_bin` column for `az` (dense path). Values are radians for angular conventions and dimensionless `u` for `direction_cosine`. Required when `az` is selected.
- **In:** `el_grid` (optional) — float32 / float64, `(num_el,)`. Second scan-coordinate grid; indexed by the preserved `el_bin` column for `el`. Values are radians for angular conventions and dimensionless `v` for `direction_cosine`. Required when `el` is selected.
- **Out:** `points` — float32 / float64, `(num_points, num_features)`. Assembled feature matrix, columns in the configured order, sized to the detection count. Feeds `ClusterDBSCANOp.points` and `DetectionReductionOp.points`.

**Constructor configuration:**

- `features` — ordered list over {range, velocity, score, az, el, az_estimate, el_estimate, doppler_bin, range_bin, az_bin, el_bin}. Selects columns and their order, fixing `num_features`, the layout, and the compile-time temporal join. `range`/`velocity` use physical RD axes; dense `az`/`el` use angle-grid lookups; sparse `az_estimate`/`el_estimate` use row-aligned `aoa_estimates`; the four `*_bin` tokens emit indices directly.
  Selected source ports are required zero-window `OnAligned` members; raw tokens require only `detection_idx`. Unselected gathers are optional `Exact` + `SafeDefault` side inputs. A selected source input that is absent is an error.
  Constructor-only because it determines input requiredness before `setup()`; it is not an ordinary Params field. Changing it requires a new operator instance and graph recompilation. Default: none (required).

**Algorithm:**

1. For each `detection_idx` row, read `(doppler_bin, range_bin)` (last two columns); on the dense path the two preserved columns just before them are `(az_bin, el_bin)`.
2. Build each configured feature: `range = range_bins[range_bin]`, `velocity = velocity_bins[doppler_bin]`, `az = az_grid[az_bin]`, `el = el_grid[el_bin]`, `az_estimate` / `el_estimate = aoa_estimates[row]`, the four `*_bin` features = raw indices, and `score =` gather `detection_score` at the cell.
3. Stack the columns in `features` order → `points`. Pure gather / 1-D
   lookup (custom CUDA; MatX is a poor fit for sparse irregular gathers);
   no scaling.

**Notes:**

- *Delivery model — mainstream thin operator:* the gather + bin→physical lookup + column stack is mechanical and reusable, so it ships as a **mainstream operator with a configurable `features` list**. It is the most application-specific of the three bridges (which columns to cluster on is a modeling choice), so a per-application implementation remains a reasonable alternative; the configurable op is the default.
- *No scaling here:* `points` carries raw physical columns; per-axis scaling for clustering / reduction lives in `ClusterDBSCANOp.epsilon` and `DetectionReductionOp.feature_weights`, so mixed units are handled once, downstream — not baked into the matrix.
- *Placement:* pre-angle (cluster-then-angle) uses `features = [range, velocity, score]` (no coordinate inputs); sparse post-angle (angle-then-cluster) adds `[az_estimate, el_estimate]` from `aoa_estimates`; the **dense** path (RDA-cube CFAR) adds convention-labelled `[az, el]` from the `az_grid`/`el_grid` inputs or raw `[az_bin, el_bin]`. The `features` list + optional coordinate inputs cover all three without a variant (resolves S4 / PhaseShift §6).
- *Separate gather kernel:* `FeatureAssemblyOp` gathers heterogeneous scalar sources into configurable columns with one thread per detection. `SnapshotGatherOp` gathers complex channel vectors over a configurable 2-D neighborhood with edge policies and a different execution shape. They intentionally keep separate optimized kernels; only their plain `detection_idx` contract is shared.
- *Row alignment:* `points` rows are 1:1 with `detection_idx` rows, so `labels` positionally join back and `DetectionReductionOp.rep_idx` resolves to real cells.
- *Carried over:* plain-tensor ports; precision by template instantiation;
  device tensors on the Holoscan CUDA stream (custom CUDA gather kernel).

#### DetectionReductionOp

Reduces each cluster to one **representative detection** and, optionally, one reduced feature row. Member-selection policies support the cluster-then-angle path by returning a real detected cell; `weighted_mean` additionally exposes a synthetic aggregate without weakening that real-cell index contract. This is a thin wrapper over the shared group-by-label reduction primitive.

- **In:** `labels` — int, `(num_points,)`. Cluster id per detection (noise = `-1`), row-aligned with `detection_idx`. From `ClusterDBSCANOp.labels`.
- **In:** `detection_idx` — int, `(num_points, 2 [+ n_batch])`. The detected cells, row-aligned with `labels` (CFAR's `detection_idx`).
- **In:** `points` — float32 / float64, `(num_points, num_features)`. A per-detection feature matrix row-aligned with `labels` and `detection_idx`; it may be the matrix clustered on or a row-aligned superset augmented with selection/weight columns (power, SNR, …). `peak`, `medoid`, and `weighted_mean` derive their score, metric, or aggregate from these columns; `first` ignores the values. Always required and part of the zero-window aligned trigger so any live `representative` value is legal without changing the compiled contract.
- **Out:** `rep_idx` — int, `(num_clusters, 2 [+ n_batch])`. One real representative cell per cluster — always a subset of `detection_idx` rows, sized to the actual cluster count (reallocated per tick). `weighted_mean` selects the member nearest the same weighted centroid computed over all `detection_idx` columns rather than rounding to an invented cell. Feeds `snapshot-gather.detection_idx`. Noise points (label `-1`) form no cluster and yield no representative.
- **Out:** `reduced_points` — float32 / float64, `(num_clusters, num_features)` (optional, latched). One row per occupied cluster in ascending label-id order. `first` / `peak` / `medoid` gather the selected real `points` row; `weighted_mean` emits the synthetic weighted feature mean. An unwired port warns once, then skips its dedicated output loan and compaction; weighted-mean accumulation still runs because `rep_idx` depends on the corresponding index centroid.

**Parameters:**

- `representative` **[live]** — enum {peak, medoid, first, weighted_mean}. Which reduction to apply: `peak` = the real member maximizing a feature-derived score; `medoid` = the real member nearest the cluster's feature centroid; `first` = lowest-index member; `weighted_mean` = a synthetic weighted feature mean plus the real `detection_idx` member nearest the weighted index centroid. Default: peak.
- `score_method` **[live]** — enum {weighted_sum, weighted_l2, weighted_max, custom}. For `peak` and non-uniform `weighted_mean`, how the weighted feature columns collapse to a scalar row score: `weighted_sum` = Σ wᵢ·featureᵢ; `weighted_l2` = ‖w ⊙ feature‖₂; `weighted_max` = maxᵢ wᵢ·featureᵢ; `custom` = the caller's `score_fn` functor. `peak` maximizes it; `weighted_mean` uses it as each row's aggregate weight. A one-hot `feature_weights` with `weighted_sum` selects a single strength column such as detection score. Default: weighted_sum.
- `feature_weights` **[live]** — float `(num_features,)`. Per-feature weights `w`. For `peak` they weight the columns before `score_method` combines them; for `medoid` they scale the per-axis centroid distance (mixed-unit weighting — the same idea as `ClusterDBSCANOp.epsilon`, best matched to it); for `weighted_mean` they produce each row's scalar aggregate weight through `score_method`. Default: none. Required for `peak` / `medoid`; for `weighted_mean`, empty means the ordinary unweighted mean (`αᵢ = 1`).
- `score_fn` — custom scoring functor, used only when `score_method = custom`. A **compile-time** device functor (a template parameter, like precision), mapping a length-`num_features` feature row to a scalar — not a runtime value. Default: none.
- Each `[live]` port above has no companion static; the contract entry is unconditional under this operator's `OnAligned` trigger just as it is under `OnEach` (see the Parameter override convention).

**Algorithm:**

1. Group the detection rows by `labels`, excluding noise (`-1`).
2. Per cluster, reduce by `representative`, via the shared group-by-label primitive: `peak` = argmax of the per-row score (`score_method` applied to `feature_weights ⊙ points`, or the `score_fn` functor); `first` = lowest index; `medoid` = mean the `feature_weights`-scaled points and take the argmin scaled-distance member; `weighted_mean` = compute `αᵢ = 1` when `feature_weights` is empty, otherwise derive `αᵢ` through `score_method`, then form `Σ αᵢ·pointsᵢ / Σ αᵢ` and `Σ αᵢ·detection_idxᵢ / Σ αᵢ`. Every `αᵢ` must be finite and non-negative and every occupied cluster must have a finite positive total.
3. Emit `rep_idx` from real input rows. For `weighted_mean`, select the member minimizing squared distance across all integer index columns to the weighted index centroid; ties choose the lowest input row. Optionally emit `reduced_points`, gathering selected rows for the member policies and compacting synthetic means for `weighted_mean`.

**Notes:**

- *Delivery model — thin operator over a shared library primitive:* the generic group-by-label reduction (segmented argmax/argmin/medoid, and aggregate stats) is a **public library function** any operator can call — the reusable core.
  - Cluster-reduction ships as a **thin standalone operator** wrapping it, **not** folded into `ClusterDBSCANOp`: it combines `labels` + radar `detection_idx` + a `points`-derived score and emits representative *cell* indices, which would pollute the deliberately radar-agnostic clusterer (a `points`-only contract), and the representative *policy* is consumer-specific ("no universal representative").
  - Keeping it a node also makes the cluster-then-angle path wireable out of the box. The primitive stays public, so a future clustering op could call it directly if a strong case appears.
- *Representative is a real cell:* every policy emits an actual cluster member through `rep_idx`, so it remains valid for `SnapshotGatherOp`. Synthetic values exist only on `reduced_points`; the operator never rounds a centroid into a possibly undetected index.
- *Score source:* the `peak` score and non-uniform `weighted_mean` row weights are derived from `points` columns (via `feature_weights` / `score_method`), so any selection attribute — power, SNR — must be present in a row-aligned feature matrix. Supplying it is the feature-assembly bridge's job. The natural strength column is CFAR's positive `detection_score`, gathered per detection by feature assembly; a caller may provide a row-aligned superset to reduction without forcing that strength column into DBSCAN's clustering metric.
- *Custom scoring:* `score_method = custom` uses a compile-time device functor (`score_fn`) — the module's template-based customization, GPU-callable — not a runtime callback. `weighted_mean` applies the same finite, non-negative row-weight requirement to custom scores.
- *Feature centroid (medoid only):* the centroid is computed **inside the operator** — the per-cluster mean of the `feature_weights`-scaled `points` — and the medoid is the member nearest it (min scaled distance). It is never emitted (not a real cell). Matching `feature_weights` to the upstream DBSCAN `epsilon` keeps "nearest to centroid" consistent with the metric that formed the cluster. Both `peak` and `medoid` read `points` now; only `first` is feature-free.
- *Weighted index centroid:* all `detection_idx` columns participate in the nearest-real-member calculation, including named preserved coordinates. No index normalization is inferred; callers that need a different real-member metric should use `medoid` over appropriately scaled physical features.
- *Preserved-index bound:* `detection_idx` / `rep_idx` admit at most eight preserved-index columns in addition to the two cell coordinates, matching the module's CFAR/feature-assembly bound. `MaxTensorExtents.max_features` bounds the optional `reduced_points` pool and is required at construction.
- *Carried over:* plain-tensor ports; precision by template instantiation; device tensors on the Holoscan CUDA stream (custom CUDA reduction/compact kernels).

---

### Angle Estimation Operators

#### PhaseShiftBeamformerOp (dense)

- **In:** `rd_maps` — complex64 / complex128, `(num_channels, num_doppler, num_range_bins)`. The channel axis sits immediately before the `(doppler, range)` plane and is reduced into the beam axes; `num_channels` must match `array_geometry`.
- **Out:** `rda_cube` — complex64 / complex128, `(num_az, num_el, num_doppler, num_range_bins)`. Channel axis replaced by the `az × el` beam grid (or `u × v` under the `direction_cosine` convention); feeds CFARDetectorOp's dense path directly.
- **Out:** `az_grid` (optional, **latched**) — float32 / float64, `(num_az,)`. The effective first scan-coordinate grid used to form `rda_cube` on this tick, including a matched `[live] az_coordinates` override. Values are radians for angular conventions and dimensionless `u` for `direction_cosine`. Feeds `FeatureAssemblyOp.az_grid`.
- **Out:** `el_grid` (optional, **latched**) — float32 / float64, `(num_el,)`. The effective second scan-coordinate grid used to form `rda_cube` on this tick, including a matched `[live] el_coordinates` override. Values are radians for angular conventions and dimensionless `v` for `direction_cosine`. Feeds `FeatureAssemblyOp.el_grid`.

**Parameters:**

- `array_geometry` **[live]** — ArrayGeometry (virtual-element positions + wavelength). `num_elements` must match the `num_channels` axis (mismatch errors). Overriding (reconfigurable aperture / changed wavelength) rebuilds the beam weights and warns once; cached, keyed on the descriptor. Default: none (required).
- `angle_convention` — AngleConvention. How `az_coordinates` / `el_coordinates` and the output beam axes map to unit vectors in the array frame, or `direction_cosine` to scan in sine-space `(u, v)`. **Static** — it fixes the *meaning* of the output angles for the whole graph; vary it by instantiating a separate operator. Default: `radar_boresight`.
- `az_coordinates` **[live]** — float vector. First scan-coordinate grid: azimuth radians for angular conventions, or dimensionless `u` under `direction_cosine`. Overriding steers the scan and rebuilds weights (cached). Default: none (required).
- `el_coordinates` **[live]** — float vector. Second scan-coordinate grid: elevation radians for angular conventions, or dimensionless `v` under `direction_cosine`; a single value (e.g. `[0]`) gives a 1-D azimuth scan. Default: none (required).
- `array_taper` **[live]** — WindowSpec. Amplitude taper for spatial sidelobe control (aperture shading); Taylor / Chebyshev are canonical. A non-custom window is applied element-wise on a ULA or unrecognized geometry and as the separable product `w_y[iy] * w_z[iz]` on each recognized filled URA. A custom window remains an explicit per-element vector in geometry/channel order. Overriding rebuilds weights (cached). Default: none.

*(The beam weights W depend on `array_geometry`, `az_coordinates`, `el_coordinates`, and `array_taper`; overriding any of them rebuilds W — cached keyed on the inputs, so an unchanged override is free and a continuously-steered grid rebuilds as its expected per-tick cost.)*

Application YAML may spell either scan axis as an explicit array or as `{min: ..., max: ..., step: ...}`. The shorthand includes both endpoints and requires finite values, positive `step`, `min <= max`, and an integral number of steps within `64 * epsilon * max(1, abs((max - min) / step))`. It is expanded before Params decoding; the operator Params remain `std::vector<double>` and live overrides remain rank-1 float64 Tensors.

#### SnapshotGatherOp

Gathers, for each detection, the per-channel complex snapshot(s) from the RD maps — the channel vector at the detected cell plus a configurable range-Doppler neighborhood — into the `snapshot` matrix the sparse angle estimators consume. Pure indexed gather; no arithmetic.

- **In:** `rd_maps` — complex64 / complex128, `(num_channels, num_doppler, num_range)`. The **per-channel** RD maps (SIMO: physical RX; MIMO: the assembled virtual array), channel axis immediately before the `(doppler, range)` plane, leading axes batched. Must be the pre-integration maps — *not* the NCI-integrated maps CFAR thresholded on (the channel axis has to be intact to gather from).
- **In:** `detection_idx` — int64, `(num_detections, 2 [+ n_batch])`. Detected `(doppler_bin, range_bin)` cells — CFAR's `detection_idx` contract (sized to the actual count; its row count *is* `num_detections`). Every preserved-batch, Doppler, and range coordinate must be in bounds for `rd_maps`.
- **Out:** `snapshot` — complex64 / complex128, `(num_detections, num_channels, num_snapshots)`. Row *i* is detection *i*'s `(num_channels, num_snapshots)` block; sized to the actual detection count and reallocated per tick (no fixed capacity / padding). `num_snapshots = (2·doppler_half + 1)·(2·range_half + 1)` from `snapshot_window` (1 at the default) — the full window product for every detection regardless of edge mode; under a `clamp` edge, out-of-bounds slots are left unfilled and flagged in `is_valid`. Feeds `⟨sparse angle estimator⟩.snapshot`.
- **Out:** `is_valid` (optional, **always**) — uint8 (`0`/`1`), `(num_detections, num_snapshots)`. Per-detection mask of which snapshot slots hold a real gathered cell — all `1` except where a `clamp` edge truncated the window (ragged). It is emitted on every tick because the gather kernel produces it as a byproduct; consumers may leave it unwired. Sparse estimators use it to restrict covariance / beamforming to valid snapshots. `uint8` rather than `bool` for HSDK 4.3 NumPy visibility (see `CFARDetectorOp.detection_mask`); the sparse estimators accept a `bool` or `uint8` mask.

**Parameters:**

- `snapshot_window` **[live]** — int per axis `(doppler, range)`, the neighborhood half-extent gathered around each detection. `(0, 0)` → one snapshot (the detection cell itself). `num_snapshots = (2·doppler_half + 1)·(2·range_half + 1)`. Overriding changes `num_snapshots` (an output-shape change — downstream infers); cheap (a gather, no setup). Default: `(0, 0)` (single-snapshot — matches Beamscan).
- Each `[live]` port above has no companion static; the contract entry is unconditional under this operator's `OnAligned` trigger just as it is under `OnEach` (see the Parameter override convention).
- `doppler_edge` — enum {wrap, clamp, shift}. Neighborhood behavior at the Doppler-axis edges. `wrap`: periodic — bin 0 is adjacent to bin `num_doppler − 1` in the aliased spectrum (physically correct for the Doppler FFT axis). `clamp`: truncate the window at the axis bound, yielding fewer real snapshots near the edge (ragged) — the missing slots are flagged in `is_valid`. `shift`: slide the window inward to keep `num_snapshots` constant using real distinct cells (asymmetric about the detection near an edge); the requested window length must not exceed the axis extent. Static. Default: wrap.
- `range_edge` — enum {wrap, clamp, shift}, the same three behaviors as `doppler_edge`. (`wrap` is offered for symmetry but is rarely physical for range — range bins are not periodic.) Static. Default: shift.

**Algorithm:**

1. Take the `detection_idx` rows (its row count is `num_detections`).
2. For each detection `(doppler_bin, range_bin)`, enumerate the neighborhood cells per `snapshot_window`, resolving each axis's out-of-bounds indices by its edge mode (`wrap` / `clamp` / `shift`).
3. Gather the length-`num_channels` complex vector at each in-bounds neighborhood cell from `rd_maps` and stack into a `(num_channels, num_snapshots)` block.
4. Write it to `snapshot[i]`, marking any `clamp`-truncated slots `false` in `is_valid` (all-true otherwise). The output is sized to `num_detections` (allocated per tick). A focused CUDA gather handles the channel-neighborhood copy; leading batch axes are selected from `detection_idx`'s index width.

**Notes:**

- *Delivery model — mainstream operator:* the gather is generic, plain-tensor in / plain-tensor out, and reusable across any CFAR + any sparse estimator, so it ships as a **mainstream library operator** (`SnapshotGatherOp`), not per-application code.
- *Snapshot source scope:* RD-neighborhood only. **Spatial smoothing** is applied downstream in the estimator (`spatial_smoothing` param), operating on the gathered channel axis. **Multi-frame / temporal** snapshots would need cross-tick state — out of scope for this stateless bridge (a future stateful variant could add them).
- *Neighbor assumption:* the window cells are assumed to share the detection's DOA (target energy straddles adjacent bins from windowing/straddle). Valid for point and modestly-extended targets over a few bins; a large window blends distinct sources and biases the covariance — the caller's footgun to size sensibly.
- *Channel axis intact:* wire from the per-channel `rd_maps` (which CFAR also reads with `nci_axes = {channel}`), not from an integrated map. `num_channels` flows through unchanged and must match the estimator's `array_geometry`.
- *Sizing:* `snapshot` (and `is_valid`) are sized to the actual `num_detections` and reallocated per tick — no fixed-capacity padding, no valid-count companion. `num_detections` is just the row count of the incoming `detection_idx`.
- *Beamscan:* the default `(0, 0)` window yields `num_snapshots = 1`, which Beamscan accepts (its 2D-or-3D-with-`num_snapshots=1` rule).
- *Carried over:* plain-tensor ports; precision by template instantiation; MatX + MatX memory spaces.

#### SubarraySelectionOp

Selects a uniform ULA/URA sub-array from a (possibly holey) MIMO virtual array and re-emits a reduced `(snapshot, array_geometry)` pair (S3) so gridless estimators and holey-array spatial smoothing can run. Host-planned selection is cached; the per-tick channel gather reuses the S1 gather (`P=1`, no forward-backward).

- **In:** `snapshot` — complex64 / complex128, `(num_items, N_virtual)` or `(num_items, N_virtual, num_snapshots)`. Holey virtual-array snapshot from `SnapshotGatherOp`; `N_virtual` must match `array_geometry.num_elements`.
- **In:** `is_valid` (optional) — bool or uint8, `(num_items, num_snapshots)`. Forwarded unchanged when present (channel selection leaves the mask intact).
- **Out:** `snapshot_out` — complex64 / complex128, `(num_items, L[, num_snapshots])`. Selected-channel snapshot (`L` = sub-array size). HSDK 5 requires unique port names within an operator (inputs and outputs share one namespace), so the physical output name is `snapshot_out` rather than repeating `snapshot` ([EA HSDK5 ONLY]).
- **Out:** `array_geometry` (optional, **latched**) — `WireArrayGeometry`, carrying the reduced logical `ArrayGeometry` for the estimator in the shared generated descriptor-Tensor schema. When consumed, it is emitted on the same tick as `snapshot_out` and directly wires to an angle operator's same-schema `[live] array_geometry` input; Params side-channel is not acceptable for runtime-varying geometry. A consumer using a separately configured static geometry must keep geometry/subarray overrides disabled or maintain equivalent coherence itself.
- **Out:** `is_valid_out` (optional, **latched**) — the input mask, forwarded. Physical name `is_valid_out` ([EA HSDK5 ONLY] unique in/out names).

**Parameters:**

- `array_geometry` **[live]** — ArrayGeometry (holey virtual positions + wavelength). Required. HSDK 5 physical live payload: generated descriptor-Tensor schema; logical Params type remains `ArrayGeometry`.
- `subarray` **[live]** — int64 vector of explicit channel indices; empty → auto (largest filled ULA/URA). HSDK 5 physical live payload: rank-1 pageable-host int64 Tensor with a ctor-captured element bound; logical Params type remains the int64 vector.
- `min_subarray_size` — int64. Auto-selection errors below this many channels. Static. Default: 2.

#### Sparse angle estimators (shared slot)

`BeamscanEstimatorOp`, `MVDREstimatorOp`, `MUSICEstimatorOp`, `RootMUSICEstimatorOp`, and `ESPRITEstimatorOp` fill the same pluggable sparse slot. They share one **mandatory input port** and one **mandatory output port** so any estimator drops into the slot without rewiring; estimator-specific capabilities (spectrum, multi-source, diagnostics) ride on **separate optional ports** that a common consumer can ignore.

**Shared inputs:**

- `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`. One `(num_channels, num_snapshots)` block per item (detection or cluster representative). The covariance estimators form `R̂ = (1/K) Σ_k x_k·x_kᴴ` over the valid snapshots per item; single-snapshot Beamscan uses one look. `num_channels` must match `array_geometry`. Produced by `SnapshotGatherOp` (an RD-neighborhood gather).
- `is_valid` (optional) — bool **or** uint8 (`0`/`1`), `(num_items, num_snapshots)`. Per-item mask of usable snapshot slots (from `SnapshotGatherOp`'s ragged `clamp` edge mode; that operator emits `uint8`, but a `bool` mask is also accepted). When present, the covariance / beamforming sums only over the marked slots (so the effective snapshot count `K` is the per-item valid count); absent → all `num_snapshots` are used.

**Shared mandatory output:**

- `aoa_estimates` — float32 / float64, `(num_items, 2)`. The single dominant DOA `(az, el)` per item (or `(u, v)` under the `direction_cosine` convention), row-aligned with the input items and named per `angle_convention`. This is the common output every estimator always provides.

**Optional ports (per estimator, where meaningful):**

- `angle_spectrum` — float32 / float64, `(num_items, num_az, num_el)`. Grid-scan estimators only (Beamscan, MVDR, MUSIC); the gridless pair emit none.
- `doa_multi` — float32 / float64, `(num_items, max_sources, 2)`, with a companion `source_count` — int, `(num_items,)`. All resolved DOAs for the multi-source estimators; fixed capacity (`max_sources`), unused rows flagged (NaN). `aoa_estimates` is the strongest of these.
- `eigenvalues` — float32 / float64, `(num_items, num_channels)`. Covariance eigenvalue spectrum for subspace / source-count diagnostics (covariance estimators).

**Shared parameters** (all five):

- `array_geometry` **[live]** — ArrayGeometry (element positions + wavelength). `num_elements` must match the snapshot `num_channels` (mismatch errors). Overriding (reconfigurable aperture / changed wavelength) rebuilds the steering weights and warns once; cached, keyed on the descriptor. Default: none (required).
- `angle_convention` — AngleConvention. How the scan grid (grid-scan estimators) and the output `(az, el)` map to unit vectors in the array frame; `direction_cosine` for sine-space `(u, v)`. **Static** — it fixes the *meaning* of the output angles for the whole graph, so it is not live-overridable; vary it by instantiating a separate estimator. Default: `radar_boresight`.

Grid-scan estimators (Beamscan, MVDR, MUSIC) additionally take a scan grid; the gridless pair (Root-MUSIC, ESPRIT) have none:

- `az_coordinates` **[live]** — float vector. First scan-coordinate grid: azimuth radians for angular conventions, or dimensionless `u` under `direction_cosine`. Overriding steers the scan and rebuilds weights (cached). Default: none (required for grid-scan estimators).
- `el_coordinates` **[live]** — float vector. Second scan-coordinate grid: elevation radians for angular conventions, or dimensionless `v` under `direction_cosine`; a single value (e.g. `[0]`) gives a 1-D azimuth scan. Default: none (required for grid-scan estimators).

Application YAML accepts either an explicit array or inclusive `{min: ..., max: ..., step: ...}` for each grid. The shorthand has the same finite, positive-step, ordered, integral-endpoint requirements as PhaseShift and is expanded before Params decoding; it does not change the vector Params or live Tensor ports.

Per-estimator deviations from the shared contract:

##### BeamscanEstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels)` **or** `(num_items, num_channels, num_snapshots)`. Beamscan is single-snapshot: a 3D input with `num_snapshots ≠ 1` is an error (a 2D input is the canonical form).
- **Out:** `aoa_estimates` — `(num_items, 2)`.
- **Out:** `angle_spectrum` (optional, **always**) — `(num_items, num_az, num_el)`. Single-source, single-snapshot — no `doa_multi`, no `eigenvalues`. `always` because the scan grid is the same pass that yields `aoa_estimates`.

**Parameters:** shared (`array_geometry`, `angle_convention`, `az_coordinates`, `el_coordinates`) plus:

- `array_taper` **[live]** — WindowSpec. Amplitude taper for spatial sidelobe control (aperture shading). A non-custom window is applied element-wise on a ULA or unrecognized geometry and as the separable product `w_y[iy] * w_z[iz]` on each recognized filled URA. A custom window remains an explicit per-element vector in geometry/channel order. Overriding rebuilds weights (cached). Default: none.
- `peak_interpolation` **[live]** — enum {none, parabolic}. Sub-grid peak refinement: `parabolic` fits a quadratic around the grid maximum for sub-bin accuracy; `none` returns the argmax cell. Default: none.

##### MVDREstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`. `num_snapshots` should be ≳ `num_channels` for a well-conditioned (invertible) covariance.
- **Out:** `aoa_estimates` — `(num_items, 2)`.
- **Out:** `angle_spectrum` (optional, **always**) — Capon spectrum, `(num_items, num_az, num_el)`. `always` because the Capon grid also yields `aoa_estimates`.
- **Out:** `doa_multi` + `source_count` (optional, **latched** — one shared peak-extraction pass, skipped only once both latch); `eigenvalues` (optional, **latched** — latching skips a batched Hermitian eigendecomposition run solely for this port).

**Parameters:** shared (`array_geometry`, `angle_convention`, `az_coordinates`, `el_coordinates`) plus:

- `diagonal_loading` **[live]** — float or `auto`. Diagonal loading added to the covariance for robustness / invertibility (`R + δI`); `auto` scales δ to a small multiple of the estimated noise floor. Default: auto.
- `spatial_smoothing` **[live]** — int-vector or none. Per-aperture-axis sub-array **counts** for forward spatial smoothing on a filled ULA/URA (`[P]` on a ULA, `[P_az, P_el]` on a URA; empty = off). Decorrelates coherent multipath and raises covariance rank at the cost of effective aperture (sub-array length `L_axis = M_axis − P_axis + 1`, so up to `∏P` — or `2∏P` with forward-backward — coherent sources). A non-uniform / holey virtual array must be reduced to a uniform sub-array upstream first (see S3). Default: none.
- `forward_backward` — bool. Include the conjugate-reversed looks (Pillai/Kwon forward-backward smoothing; needs a centro-symmetric ULA/URA). Applies only when `spatial_smoothing` is set. Default: true.
- `max_sources` **[live]** — int. Capacity of the optional `doa_multi` output — the number of Capon-spectrum peaks to report (`aoa_estimates` is always the strongest). Overriding reallocates `doa_multi` (minor). Default: 1.

##### MUSICEstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`.
- **Out:** `aoa_estimates` — `(num_items, 2)`.
- **Out:** `angle_spectrum` (optional, **always**) — MUSIC pseudo-spectrum, `(num_items, num_az, num_el)`. `always` because the pseudo-spectrum also yields `aoa_estimates` and feeds peak extraction.
- **Out:** `doa_multi` + `source_count` (optional, **latched** — one shared peak-extraction pass, skipped only once both latch); `eigenvalues` (optional, **always** — the eigendecomposition is required for the subspace projector, so the port is a re-export with no dedicated work).

**Parameters:** shared (`array_geometry`, `angle_convention`, `az_coordinates`, `el_coordinates`) plus:

- `num_sources` **[live]** — int or `auto`. Signal-subspace dimension (model order). `auto` = per-item **EFT** (Exponential Fitting Test — robust at the low, `M≈N` snapshot counts of automotive radar, where MDL/AIC collapse) on the **forward-backward-smoothed** covariance eigenvalue spectrum (the `eigenvalues` output), clamped to `[1, max_sources]`; an integer fixes the order uniformly. A rank-deficient covariance (smoothing off, `K < channels`) falls back to 1 and warns. Default: auto. *(Seeding from the CFAR/cluster detection count is **rejected** — that is the detection count, not the per-cell angular multiplicity `num_sources` denotes; RMT/Kritchman–Nadler, AIC, and an upstream-seed port are deferred. See STATUS.md S2.)*
- `source_count_pfa` **[live]** — float. EFT over-estimation (false-alarm) probability for `auto`; higher → more sources admitted. Coherent with CFAR `pfa`. Default: 1e-2.
- `spatial_smoothing` **[live]** — int-vector or none. As MVDR — per-aperture-axis sub-array counts on a filled ULA/URA (`[P]` / `[P_az, P_el]`; empty = off); decorrelates coherent multipath at the cost of aperture. Default: none.
- `forward_backward` — bool. Forward-backward smoothing (needs a centro-symmetric ULA/URA); applies only when `spatial_smoothing` is set. Default: true.
- `max_sources` **[live]** — int. Capacity of the optional `doa_multi` / `source_count` outputs (holds up to `min(num_sources, max_sources)` DOAs); also the K_max cap on `auto`. Default: 4.

##### RootMUSICEstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`. Requires a ULA **or** a filled URA (Vandermonde per aperture axis); the geometry is auto-detected from `array_geometry`.
- **Out:** `aoa_estimates` — `(num_items, 2)`. Gridless (DOAs from polynomial roots) — **no `angle_spectrum`**. A ULA resolves one coordinate (azimuth; elevation degenerate ≈ 0); a URA resolves native `(az, el)`.
- **Out:** `doa_multi` + `source_count` (optional, **always** — the same terminal kernel also produces `aoa_estimates`, so nothing is saved without splitting it); `eigenvalues` (optional, **always** — the eigendecomposition is required for the root polynomial).

**Parameters:** shared `array_geometry` + `angle_convention` (**no scan grid**) plus:

- `num_sources` **[live]** — int or `auto`. Model order (root count); `auto` = per-item EFT on the forward-backward-smoothed covariance eigenvalues, clamped to `[1, max_sources]` (`source_count_pfa` sets the rate); an integer fixes it. Default: auto. *(See MUSICEstimatorOp / STATUS.md S2.)*
- `source_count_pfa` **[live]** — float. EFT over-estimation probability for `num_sources: auto`. Default: 1e-2.
- `spatial_smoothing` **[live]** — int-vector or none. Per-aperture-axis sub-array counts on a filled ULA/URA (`[P]` / `[P_az, P_el]`; empty = off). A partly-filled / holey virtual array must first be reduced to a uniform sub-array upstream (see S3). Default: none.
- `forward_backward` — bool. Forward-backward smoothing (needs a centro-symmetric ULA/URA); applies only when `spatial_smoothing` is set. Default: true.
- `max_sources` **[live]** — int. `doa_multi` / `source_count` capacity. Default: 4.

**Algorithm (2-D URA).** A URA (element order `(iz` outer, `iy` inner), the
`make_ura_geometry` convention) is handled by **reduced-dimension per-axis
rooting with cross-axis pairing**: the full covariance yields the shared
noise-subspace projector; row- and column-averaged sub-covariances (an `Ny×Ny`
azimuth covariance averaged over the `Nz` rows, and an `Nz×Nz` elevation
covariance averaged over the `Ny` columns) each feed the existing 1-D rooting
path (coefficients → companion → `cusolverDnXgeev`) to yield candidate azimuth
cosines `u_i` and elevation cosines `v_k`; candidates are paired by evaluating
the full 2-D MUSIC subspace metric at each `(u_i, v_k)` and greedily selecting
the `d` best. There is no canonical single-polynomial 2-D Root-MUSIC; this
reduced-dimension + pairing method is a deliberate design decision (documented
in the operator's `OPEN_ITEMS.md`), trading some joint optimality for robustness
and reuse of the 1-D machinery. A ULA keeps the unchanged single-axis cone path.
Deferred: a true joint 2-D polynomial formulation and explicit
non-ULA-per-axis subarray specs.

Requires a ULA or a filled URA; a non-uniform / holey virtual array needs a uniform sub-array selection (or spatial smoothing) first.

##### ESPRITEstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`. Requires a shift-invariant geometry: a ULA (or two identical displaced subarrays) for azimuth-only, **or** a filled URA for native az+el; the geometry is auto-detected from `array_geometry`.
- **Out:** `aoa_estimates` — `(num_items, 2)`. Gridless (closed-form eigenvalue rotation) — **no `angle_spectrum`**. A ULA resolves one coordinate (azimuth; elevation degenerate ≈ 0); a URA resolves native `(az, el)`.
- **Out:** `doa_multi` + `source_count` (optional, **always** — the same terminal kernel also produces `aoa_estimates`, so nothing is saved without splitting it); `eigenvalues` (optional, **always** — the eigendecomposition is required for the subspace rotation).

**Parameters:** shared `array_geometry` + `angle_convention` (**no scan grid**) plus:

- `num_sources` **[live]** — int or `auto`. Model order; `auto` = per-item EFT on the forward-backward-smoothed covariance eigenvalues, clamped to `[1, max_sources]` (`source_count_pfa` sets the rate); an integer fixes it. Default: auto. *(See MUSICEstimatorOp / STATUS.md S2.)*
- `source_count_pfa` **[live]** — float. EFT over-estimation probability for `num_sources: auto`. Default: 1e-2.
- `spatial_smoothing` **[live]** — int-vector or none. Per-aperture-axis sub-array counts on a filled ULA/URA (`[P]` / `[P_az, P_el]`; empty = off); decorrelates coherent multipath at the cost of aperture. Default: none.
- `forward_backward` — bool. Forward-backward smoothing (needs a centro-symmetric ULA/URA); applies only when `spatial_smoothing` is set. Default: true.
- `max_sources` **[live]** — int. `doa_multi` / `source_count` capacity. Default: 4.
- `subarray_spec` — ShiftInvariantSubarrayPair. Ordered corresponding element-index vectors (`reference_indices`, `shifted_indices`) defining the two identical displaced subarrays. Their common displacement is derived and validated against `array_geometry` rather than stored redundantly. **Static** (structural). Empty vectors select `auto`: derive the maximum-overlap pair `[0…M−2]` / `[1…M−1]` for a ULA, or the two orthogonal maximum-overlap invariances for a URA. Default: auto.

**Algorithm (2-D URA).** On a URA (element order `(iz` outer, `iy` inner)),
`subarray_spec: auto` derives **two orthogonal shift invariances** — a y-shift
(azimuth) and a z-shift (elevation) — internally from the lattice. The signal
subspace yields two rotation operators `Ψ_y`, `Ψ_z` (LS). Because in the
noiseless model `Ψ_y` and `Ψ_z` share eigenvectors (they commute), the unitary
Schur basis that triangularizes `Ψ_y` simultaneously triangularizes `Ψ_z`; the
paired diagonals give azimuth cosine `u_i` and elevation cosine `v_i`
**automatically paired** (a joint-diagonalization realization — no separate
assignment step). A ULA (or an explicit single-pair `subarray_spec`) keeps the
unchanged single-shift 1-D path. Deferred: explicit two-pair 2-D
`subarray_spec`s and the real-valued Unitary-ESPRIT speedup.

An explicit `subarray_spec` and nonempty `spatial_smoothing` are mutually exclusive: smoothing changes the effective aperture and channel indexing. With smoothing enabled, leave `subarray_spec=auto`; the pair(s) are derived from the reduced uniform aperture each tick.

*(Geometry note: Root-MUSIC and ESPRIT constrain the valid `array_geometry` — the non-uniform / holey virtual arrays from `VirtualArrayAssemblyOp` may need a ULA/subarray-selection or spatial-smoothing step first. A ULA resolves azimuth only; a filled URA resolves native `(az, el)`. They do not change the port shapes above.)*

---

### MIMO Marshaling Operators

#### MIMODeinterleaveOp

- **In:** `adc_cube` — complex64 / complex128 (IQ IF) **or** float32 / float64 (real IF), `(N_rx, M·L, samples_per_chirp)`. Slow-time interleaves the M TX per the TX schedule.
- **Out:** `adc_cube_out` — complex64 / complex128 / float32 / float64 (dtype preserved), `(M_tx, N_rx, L, samples_per_chirp)`. TX split into an explicit axis. Pure marshaling (strided reshape/gather). HSDK 5 requires unique port names within an operator (inputs and outputs share one namespace), so the physical output name is `adc_cube_out` rather than repeating `adc_cube` ([EA HSDK5 ONLY] — same pattern as ClutterFilterOp `range_profiles_out`).

**Parameters:**

- `tx_schedule` — TXSchedule. Determines how interleaved `M·L` slow time splits into `(M_tx, L)`; `L = (M·L)/num_tx`. Static, TDM only, required. The first-wave chain rejects non-uniform / staggered schedules. A uniform-repeating `tx_order` permutation is accepted; pass the same schedule to `TDMDenseCompensationOp` or `TDMSparseCompensationOp` so phase uses firing slot, not TX id. Staggered TDM is deferred — see STATUS.md Roadmap (L12).

#### TDMDenseCompensationOp

- **In:** `rd_maps` — complex64 / complex128, `(M_tx, N_rx, num_doppler, num_range_bins)`. TX axis already explicit from de-interleave.
- **Out:** `rd_maps_out` — complex64 / complex128, same shape. Per-Doppler-bin inter-TX motion phase removed. [EA HSDK5 ONLY] unique in/out port names required; logical output channel remains `rd_maps`.

**Parameters:**

- `waveform` **[live]** — FMCWWaveformDescriptor. Per-TX PRI for the Doppler-dependent inter-TX correction. Default: none (required). HSDK 5 physical live payload: the shared generated waveform descriptor-Tensor schema; logical Params type remains `FMCWWaveformDescriptor`.
- `tx_schedule` — TXSchedule. TDM firing cycle used to index the inter-TX phase correction by **firing slot** (the same descriptor as `MIMODeinterleaveOp`). Static. Default: empty (identity round-robin: TX id == firing slot). Unused for DDM. When set, `num_tx` must match the input `M_tx` axis; uniformity rules match deinterleave (one repeating cycle; `chirp_tx_index` overrides `tx_order` when present).

No array geometry is needed. `num_tx` comes from the input axis. Correction tables are built lazily and cached. A non-uniform schedule is rejected.

#### DDMDenseDemultiplexOp

- **In:** `rd_maps` — complex64 / complex128, `(N_rx, num_doppler, num_range_bins)`. The M simultaneous TX occupy coded Doppler bands.
- **In:** `velocity_bins` (optional) — float32 / float64, `(num_doppler,)`. The physical center-zero axis matching `rd_maps`. It joins `rd_maps` by `Exact{}` + `SafeDefault{}` and is omitted when no physical axis is available.
- **Out:** `rd_maps_out` — complex64 / complex128, `(M_tx, N_rx, num_doppler_out, num_range_bins)`. `num_doppler_out = num_doppler_per_tx` for linear-phase band gather or `num_doppler` for nonlinear full-axis correlation. Linear `phase_codes` are additionally dephased by `conj(code[tx,0])`; offsets-only codes use unity.
- **Out:** `velocity_bins_out` (optional, **latched**) — float32 / float64, `(num_doppler_out,)`. Emitted only when `velocity_bins` is present. Linear demultiplexing takes the centered `num_doppler_per_tx`-element slice of the input axis, preserving physical bin spacing and discarding guard bins symmetrically in the reported center-zero interval; nonlinear correlation preserves the full axis. This sideband is published before `rd_maps_out`.

**Parameters:**

- `ddm_code` — DDMCodeDescriptor. Static TX code/band layout (carries `num_tx`). Exactly one of Doppler offsets or phase codes is required. Linear offsets must map to integer bins for the current Doppler extent; nonlinear phase codes use full-axis correlation. Static means frame-to-frame code rotation requires separate operator instances or a future live-code contract. Default: none (required).

No waveform is required: the demultiplexing math is defined by `ddm_code` and tensor extents, and the optional physical axis is transformed from the upstream Doppler output rather than regenerated. Packed M-band DDM retains the reduced, aliased Doppler interval; this operator does not resolve the ambiguity fold.

#### VirtualArrayAssemblyOp

- **In:** `rd_maps` — complex64 / complex128, `(M_tx, N_rx, num_doppler, num_range_bins)`. Separated `(tx, rx)` block. (TX/RX element positions are parameters.)
- **Out:** `rd_maps_out` — complex64 / complex128, `(num_virtual, num_doppler, num_range_bins)`. `(tx, rx)` flattened to a single virtual channel axis (`c = tx·N_rx + rx`), overlapped virtual elements deduped, so `num_virtual ≤ M·N`. [EA HSDK5 ONLY] physical port name `rd_maps_out` (unique in/out namespace).
- **Out:** `array_geometry` — `WireArrayGeometry`, carrying the logical virtual `ArrayGeometry` in the shared generated descriptor-Tensor schema: `element_positions` (`p_virtual = p_tx + p_rx`), wavelength, and optional geometry type. It directly wires to angle operators' same-schema `[live] array_geometry` inputs.

**Parameters:**

- `tx_positions` — float, `(num_tx, 3)`. TX element positions (m); length must match the input `M_tx` axis. Static (fixed aperture). Default: none (required).
- `rx_positions` — float, `(num_rx, 3)`. RX element positions (m); length must match `N_rx`. Static. Default: none (required).
- `wavelength` **[live]** — float. Operating wavelength (m) for the emitted `array_geometry` (`= c / carrier_frequency`); `[live]` so it tracks a changing carrier. The virtual `element_positions` themselves are wavelength-independent. Default: none (required).
- `overlap_policy` — enum {dedup, average, keep_all}. How coincident virtual elements (`p_virtual = p_tx + p_rx` collisions) are handled: `dedup` keeps one representative (default), `average` combines their samples, `keep_all` retains duplicates. Sets `num_virtual`. Static. Default: dedup.

#### TDMSparseCompensationOp

Sparse TDM bridge for detect-before-compensate workflows. It replaces the dense `TDMDenseCompensationOp → VirtualArrayAssemblyOp → SnapshotGatherOp` segment by gathering only detected RD cells, applying each cell's Doppler-dependent TDM phase correction, and assembling its virtual-channel snapshot.

- **In:** `rd_maps` — complex64 / complex128, `(M_tx, N_rx, num_doppler, num_range_bins)`. Uncompensated, center-zero post-Doppler TDM maps with an explicit TX axis. No opaque leading batch.
- **In:** `detection_idx` — int64, `(num_detections, 2)`. Detected `(doppler_bin, range_bin)` cells in the uncompensated map. Both coordinates must be in bounds. The exact row count is `num_detections`.
- **Out:** `snapshot` — complex64 / complex128, `(num_detections, num_virtual, num_snapshots)`. Per-detection virtual-array snapshots, phase-corrected at each gathered neighborhood cell and assembled according to `overlap_policy`. `num_snapshots` has the same definition as `SnapshotGatherOp`.
- **Out:** `is_valid` (optional, **always**) — uint8 (`0`/`1`), `(num_detections, num_snapshots)`. Same edge-mask semantics as `SnapshotGatherOp`; emitted before `snapshot` when connected.
- **Out:** `array_geometry` — `WireArrayGeometry`, carrying the virtual positions generated from `tx_positions + rx_positions` and the wavelength derived from the effective `waveform.carrier_frequency`. It is emitted before `snapshot`, so the estimator's `Exact{}` join sees geometry and data from the same tick.

**Parameters:**

- `waveform` **[live]** — FMCWWaveformDescriptor. Per-TX PRI and carrier frequency for the TDM correction; carrier frequency also determines the emitted geometry wavelength. Default: none (required). HSDK 5 physical live payload uses the shared generated waveform descriptor-Tensor schema.
- `tx_schedule` — TXSchedule. Static TDM firing cycle used to index correction by firing slot. Default: empty (identity round-robin). Uniformity and `num_tx` validation match `TDMDenseCompensationOp`.
- `tx_positions` — float, `(num_tx, 3)`. Static TX element positions (m); length must match `M_tx`. Default: none (required).
- `rx_positions` — float, `(num_rx, 3)`. Static RX element positions (m); length must match `N_rx`. Default: none (required).
- `overlap_policy` — enum {dedup, average, keep_all}. Same virtual-element grouping and ordering as `VirtualArrayAssemblyOp`. Static. Default: dedup.
- `snapshot_window` **[live]** — int per axis `(doppler, range)`. Same normalization, output-shape behavior, and one-tick override semantics as `SnapshotGatherOp`. Every neighborhood cell uses its own Doppler bin for phase correction. Default: `(0, 0)`.
- `doppler_edge` — enum {wrap, clamp, shift}. Same semantics as `SnapshotGatherOp`. Static. Default: wrap.
- `range_edge` — enum {wrap, clamp, shift}. Same semantics as `SnapshotGatherOp`. Static. Default: shift.

**Algorithm and detection boundary:**

1. For each `detection_idx` row, resolve the configured RD neighborhood and edge behavior.
2. For each valid neighborhood cell `(d, r)`, multiply every `(tx, rx)` sample by the cached TDM correction for `(tx, d)`.
3. Apply the same virtual assembly plan as `VirtualArrayAssemblyOp`, including overlap handling, and write the virtual vector into the estimator-facing snapshot.
4. Derive `array_geometry.wavelength = c / waveform.carrier_frequency` from the same effective build-time or live waveform used for correction.

TDM correction is unit magnitude, so pre-compensation CFAR with non-coherent integration over both TX and RX (`nci_axes = {0, 1}`) is power-equivalent when channels remain independent (`keep_all`, or no coincident virtual elements). It is not generally byte- or statistic-equivalent to CFAR after `dedup` or coherent `average`, because those policies change the channel set before detection.

#### DDMPowerFoldOp

Power-domain detector preparation for sparse DDM. It folds every usable TX band onto the reduced Doppler axis and non-coherently integrates RX without materializing a demultiplexed cube.

- **In:** `rd_maps` — complex64 / complex128, `(N_rx, num_doppler, num_range_bins)`. Pre-demux, center-zero Doppler maps.
- **Out:** `power_map` — float32 / float64, `(num_doppler_per_tx, num_range_bins)`, where `B = num_doppler/M_tx` and `num_doppler_per_tx = B - guard_bands`.

**Parameters:**

- `ddm_code` — DDMCodeDescriptor. Required and static. Sparse v1 accepts only linear M-band codes whose offsets map to integer Doppler bins; nonlinear codes and fractional-bin offsets reject.

For each folded cell, sum `|rd_maps[rx, mapped_bin(tx,d),r]|²` over all `M_tx·N_rx` looks using the exact table shared with `DDMDenseDemultiplexOp`. CFAR consumes this real map with `nci_axes: []` and `preintegrated_looks: M_tx·N_rx`. That Gamma model assumes independent looks; correlated channels require empirical calibration. `G` discarded bins per width-`B` band create a blind-velocity fraction `G/B`; applications should derive `G` from the slow-time window main-lobe/transition width.

#### DDMSparseDemultiplexOp

Sparse DDM bridge for detect-before-demux workflows. It gathers each TX band only around folded CFAR detections, applies linear-code dephasing and a supplied velocity-fold correction, and assembles estimator-ready virtual snapshots.

- **In:** `rd_maps` — complex64 / complex128, `(N_rx, num_doppler, num_range_bins)`. Same pre-demux cube as `DDMPowerFoldOp`.
- **In:** `detection_idx` — int64, `(num_detections, 2)`. `(folded_doppler_bin, range_bin)` on the reduced `(W,R)` surface.
- **In:** `fold_idx` — int64, `(num_detections)`. Doppler ambiguity integer in `[0,M_tx)`. Unconditional Exact side input with `SafeDefault{0}`; when connected its row count must equal `num_detections`.
- **Out:** `snapshot` — complex64 / complex128, `(num_detections, num_virtual, num_snapshots)`.
- **Out:** `is_valid` (optional, **always**) — uint8, `(num_detections, num_snapshots)`.
- **Out:** `array_geometry` — `WireArrayGeometry`, coherent with snapshot ordering and emitted before `snapshot`.

**Parameters:**

- `ddm_code` — DDMCodeDescriptor. Required, static, linear integer-bin codes only.
- `tx_positions`, `rx_positions`, `overlap_policy` — same static geometry and virtual grouping semantics as `TDMSparseCompensationOp`.
- `wavelength` **[live]** — float. Required operating wavelength for emitted geometry; it does not affect DDM demultiplexing.
- `snapshot_window` **[live]**, `doppler_edge`, `range_edge` — same neighborhood and edge semantics as `TDMSparseCompensationOp`, applied on the folded `W` axis.

`fold_idx` undoes the cyclic TX permutation for an aliased target. The default zero is correct only in the principal reduced interval `[-PRF/(2M), PRF/(2M))`; outside it, unresolved zero can produce a plausible but wrong virtual ordering and angle. Callers needing wider velocity coverage must supply `fold_idx` from an external staggered-CRI, code-rotation, or other observation. An M+1 empty-band layout is a research item; no generic resolver operator is defined until a specific observable and algorithm are selected. Sparse DDM v1 does not support clutter filtering; the existing slow-time clutter filter remains valid only after dense demultiplexing.

---

## Supporting data structures

Shared types the operators depend on; expanded as each operator is locked.

### ArrayGeometry

Describes the (virtual) antenna array for beamforming and DOA estimation. Shared across `PhaseShiftBeamformerOp` and the sparse DOA estimators to keep array conventions consistent. Passed as a parameter (config), not a port.

- `element_positions` — float, `(num_elements, 3)`. Element coordinates (meters). General form (ULA / URA / UCA / Conformal). Whatever the channel axis represents: physical RX positions (SIMO) or synthesized virtual positions (`p_tx ⊕ p_rx`, MIMO). `num_elements` must match the operator's `num_channels` axis.
- `wavelength` — float. Operating wavelength (m), `c / carrier_frequency`. Sets the steering-vector phase per element.
- `geometry_type` (optional) — enum {ULA, URA, UCA, Conformal}. Informational; enables optimized kernels (e.g. the spatial-FFT fast path).

### ShiftInvariantSubarrayPair

Defines two ordered, corresponding element-index sets with one common nonzero displacement, for ESPRIT and future shift-invariant array methods. Passed as a static parameter, never a data port. The displacement is derived from `ArrayGeometry.element_positions`; storing it separately would permit contradictory geometry.

- `reference_indices` — int vector. Element indices for the reference subarray.
- `shifted_indices` — int vector of the same length. `shifted_indices[i]` corresponds to `reference_indices[i]`; every position difference must equal the same displacement within geometry tolerance.
- Both vectors must be nonempty, equal length, unique within each vector, and in range. The two vectors may overlap (the standard ULA maximum-overlap pair does).
- Both vectors empty is the `auto` sentinel. For a **single** shift-invariant pair (the generic case, and the 1-D ESPRIT path) `auto` derives `[0, …, M−2]` / `[1, …, M−1]` on a ULA with `M` elements, and rejects geometry that is not a uniformly spaced line in snapshot channel order. `ESPRITEstimatorOp` additionally accepts `auto` on a filled URA, where it derives the *two* orthogonal maximum-overlap invariances (y-shift and z-shift) from the lattice rather than a single pair — see the ESPRITEstimatorOp algorithm note. An explicit (nonempty) `subarray_spec` always specifies exactly one pair.

### AngleConvention

Defines the azimuth/elevation (AER) parameterization at the API boundary — the `(az, el) ↔ unit-vector` mapping — for every operator that consumes a scan grid or emits `(az, el)`. Passed as a standalone parameter, defaulting to `radar_boresight`; *not* folded into `ArrayGeometry`. Axis references interpreted in the array-local Cartesian frame; angles in radians; applied only as a boundary relabeling (kernels run in the fixed canonical convention).

- `preset` — enum {radar_boresight, ros_rep103, iso8855_flu, aerospace_ned, physics_spherical, geodetic_enu, geodetic_ned, direction_cosine, custom}. `radar_boresight` = the internal canonical convention (identity/no-op). Other presets map to the cross-domain table rows. `direction_cosine` expresses the grid and `doa` in sine-space `(u, v)` — convention- and singularity-free; non-physical points (`u²+v² > 1`) are masked (`−inf` spectrum, NaN `doa`). `custom` builds the mapping generically from the axis fields below (an orthonormal frame + `zenith` polar step); the three axes must be **distinct principal axes** (mutually orthogonal), else the config is rejected. Each named preset is a specific `custom` configuration. **Default: `radar_boresight`.**
- `boresight_axis` — signed-axis enum {±x, ±y, ±z}. Axis that `(az=0, el=0)` points along. Used only when `preset = custom`.
- `azimuth_positive_axis` — signed-axis enum. Axis increasing azimuth rotates toward (fixes azimuth plane, sense, handedness). Orthogonal to `boresight_axis`. `custom` only.
- `elevation_positive_axis` — signed-axis enum. Axis increasing elevation rotates toward (`+z` = up, `−z` = NED-style down). Orthogonal to both other axes. `custom` only.
- `elevation_reference` — enum {horizon, zenith}. Elevation up from the azimuth plane (radar/robotics/aviation) vs. polar angle from boresight/pole (physics). `zenith` carries `el ↔ π/2 − polar` (a genuine angle transform). Default: horizon (`custom` only).

*Scan-grid ranges.* Every authored scan-grid coordinate must be finite. For angular conventions, grid coordinates may otherwise be any real value: the boundary→canonical map is periodic in azimuth (period `2π`), so `az` and `az ± 2πk` are equivalent and no normalization is required. Under `direction_cosine`, every authored `u` and `v` axis value must lie in `[-1, 1]`; Cartesian-product corners with `u² + v² > 1` remain valid grid construction and are intentionally masked per beam (`−inf` spectrum, NaN `doa`). Emitted `doa`/`(az, el)` are returned in the **principal range**: `az ∈ (−π, π]` and `el ∈ [−π/2, π/2]` (or `el ∈ [0, π]` under `zenith`). At a pole (`el = ±π/2`) azimuth is undefined and reported as `0`.

### WindowSpec

Specifies a window function for the FFT-based operators (range / Doppler windowing) and the array taper.

- `type` — enum {none, Hann, Hamming, Blackman, Blackman-Harris, Kaiser, Chebyshev, Taylor, flat_top, Tukey, custom}. Window function. `flat_top` — wide mainlobe but flat passband, the choice when amplitude accuracy matters more than resolution (calibration, amplitude estimation). `Tukey` — tapered-cosine edge control.
- `sidelobe_attenuation_db` — float. Target sidelobe attenuation (dB); used by `Kaiser`, `Chebyshev`, `Taylor`. Default: 30.
- `kaiser_beta` — float. Kaiser shape parameter; used only when `type = Kaiser`. Derived from `sidelobe_attenuation_db` when unset.
- `taylor_nbar` — int. Number of near-constant sidelobes; used only when `type = Taylor`. Default: 4.
- `tukey_alpha` — float. Taper ratio (fraction of the window that is cosine-tapered; 0 = rectangular, 1 = Hann); used only when `type = Tukey`. Default: 0.5.
- `coefficients` — float vector. User-supplied window; used only when `type = custom` (length must match the windowed axis).
- `coherent_gain_normalization` — bool. Normalize the window for unit coherent gain. Default: `true`.

Used by: `FMCWRangeProcessorOp` / `FMCWDopplerProcessorOp` / `FMCWRangeDopplerProcessorOp` (`range_window`, `doppler_window`) and `PhaseShiftBeamformerOp` / `BeamscanEstimatorOp` (`array_taper`).

### FMCWWaveformDescriptor

Parameters of the transmitted FMCW chirp, shared across operators to avoid parameter drift. Immutable value type: built from an `FMCWWaveformSpec` (the construction input), which is validated once at construction — every derived quantity (slope ↔ bandwidth, wavelength) is precomputed there, all accessors are plain non-throwing reads, and a constructed descriptor cannot hold inconsistent fields. To change a value, construct a new descriptor from an updated spec.

- `carrier_frequency` — float (Hz). Sets the wavelength `λ = c / carrier_frequency` (Doppler/velocity axis and the steering-vector phase).
- `bandwidth` **or** `chirp_slope` — float. Define one, derive the other (`slope = bandwidth / chirp_duration`). Sets range resolution and the beat-frequency → range mapping.
- `chirp_duration` — float (s). Active sweep duration.
- `sample_rate` — float (Hz). Fast-time (ADC) sampling rate.
- `chirp_repetition_interval` (PRI) — float (s); `PRF = 1/PRI`. Slow-time sampling period between successive chirps (≥ `chirp_duration`; includes any inter-chirp idle). **For MIMO this is the per-TX effective interval** (TDM: `M ×` the raw inter-chirp time), so the Doppler/velocity axis reflects the per-TX unambiguous velocity.
- `reference_samples` (optional) — complex vector, length `samples_per_chirp`. Explicit samples of a **single** reference chirp; used directly as the dechirp reference (measured / non-ideal case). When absent, the reference is synthesized from the parameters above.

Used by: `FMCWRangeProcessorOp` (dechirp reference; slope + sample rate for `range_bins`), `FMCWDopplerProcessorOp` (carrier → λ, PRI → PRF for `velocity_bins`), and the fused operator. Passed as a static parameter and/or delivered in-flight on the `waveform` override port.

### DDMCodeDescriptor

Describes the Doppler-division layout consumed by
`DDMDenseDemultiplexOp`, `DDMPowerFoldOp`, and
`DDMSparseDemultiplexOp`: how the M TX are separated after the Doppler FFT.

- `num_tx` — int. M, the number of TX multiplexed into distinct Doppler sub-bands.
- `phase_codes` (optional) — float or complex, `(num_tx, num_chirps)`. The per-chirp phase (or unit-magnitude code) applied to each TX; the general form, demuxed by conjugate correlation. Provide this **or** `doppler_offsets`. **Implemented:** *linear-phase* codes (constant per-chirp phase increment per TX) are reduced to `doppler_offsets` and demuxed by Doppler-band gather; *non-linear* codes use the correlate path (full Doppler axis out, `guard_bands` must be 0, `num_chirps` must equal `num_doppler`). Algorithm detail: the operator README.
- `doppler_offsets` (optional) — float, `(num_tx,)`. Per-TX Doppler sub-band offset **as a fraction of PRF (cycles/chirp) in `[0, 1)`** (e.g. `m/M` for uniform DDM); the compact form for standard linear-phase DDM. Fractions are canonical because the descriptor is built before runtime `num_doppler` is known. Linear M-band consumers require `offset · num_doppler` to be an exact integer band origin and require every band to be occupied exactly once. Provide this **or** `phase_codes`.
- `guard_bands` — int. Guard bins reserved between adjacent TX Doppler bands to keep them separable. Default: 0.

Notes: the `[0, PRF)` Doppler span is shared across the M bands, so unambiguous velocity is reduced by M; band width `num_doppler_per_tx` follows the offset/guard layout. There is no unfold / disambiguation step — the demuxed axis is that reduced span.

**Imposed layout (current slice):** `num_doppler` must be divisible by `num_tx`; each band is `num_doppler / num_tx` bins wide, and `num_doppler_per_tx = num_doppler / num_tx − guard_bands` (the top `guard_bands` fft-order bins of each band are discarded as the inter-band guard). The demuxed output Doppler axis is center-zero over the reduced `[−PRF/2M, PRF/2M)` span, with every TX lane un-shifted to a common reduced-Doppler axis so the emitted `(tx, rx)` block is coherent. Standard linear-phase demux is a waveform-independent Doppler-bin gather.

### TXSchedule

Describes the TDM slow-time firing pattern for `MIMODeinterleaveOp` and
`TDMDenseCompensationOp` / `TDMSparseCompensationOp`: how interleaved slow
time splits into a TX axis and which firing slot each TX id occupied.

- `num_tx` — int. M, the number of time-multiplexed TX.
- `tx_order` (optional) — int, `(num_tx,)`. TX firing order within one repetition cycle; the schedule repeats it `L` times. Default: round-robin `[0, 1, …, M−1]` (the common case). De-interleave gathers per this order; compensation indexes the TDM phase by the inverse (firing slot of each TX id).
- `chirp_tx_index` (optional) — int, `(M·L,)`. Explicit per-chirp TX index for irregular / non-repeating schedules; overrides `tx_order` when present.

Notes: `chirps_per_tx` (L) is inferred as `(M·L) / M` from the input's slow-time length; the per-TX effective PRI is `M ×` the raw inter-chirp interval (carried in `FMCWWaveformDescriptor.chirp_repetition_interval`).
