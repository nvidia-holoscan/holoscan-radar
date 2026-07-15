# Holoscan Radar Module — FMCW Operator API Lockdown

## Purpose

An **FMCW-focused** operator API lockdown. Scope is derived from the FMCW workflows doc: the operators here are exactly those the SISO / SIMO / MIMO-TDM / MIMO-DDM chains require, plus the MIMO data-marshaling operators and the I/O bridges those chains lean on. Each operator is specified by its **Inputs / Outputs / Parameters / Algorithm / Notes**.

Design conventions (from the workflows doc): plain-tensor ports (bridge with small explicit add-on operators, never shared record/struct message types); precision fixed by compile-time template instantiation (single + double), not a runtime flag or dtype inference; MatX implementation accepting tensors in any CUDA-accessible memory space (device, managed, pinned host — pageable-host input is rejected; amended 2026-07-10) and producing device memory; rank-agnostic batching (operate on the meaningful axes, treat leading axes as opaque batch dims, infer shapes from tensors); industry-standard operator names; and angle operators that make the az/el convention explicit at the API boundary (required, no default).

**Precision (all operators).** Every operator supports both **single** (`complex64` / `float32`) and **double** (`complex128` / `float64`) precision floating-point I/O. Precision is fixed at compile time by the operator's template instantiation — it is neither a runtime parameter nor inferred from the data — and an operator's input and output tensors must match that templated precision. The per-operator dtype labels below list both forms (e.g. `complex64 / complex128`, `float32 / float64`); a given operator instance uses one, fixed by its instantiation, and its inputs and outputs share it. Any operator that must fix precision for a qualitative reason is called out in its own Notes.

**Proposed amendments.** Text marked **PROPOSED — needs team review (not locked)** (or tagged **[PROPOSED]**) is a working proposal only. It is not accepted lockdown until the team explicitly adopts it. Today that applies to the `CFARDetectorOp` calibration / `alpha_lut` block under Detection & Clustering.

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
  - It is a **required, standalone parameter (no default)** on operators that consume or emit angles, so the frame is never silently inherited.
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

Out of scope here. **CDM / BPM** (code-division / binary-phase multiplexing) are **under consideration for a future wave** — architecturally a small extension of the DDM path (all TX on, per-chirp phase code, separated after the Doppler stage), reusing `MIMOCoherentCompensationOp` in a code-correlation mode and the rest of the MIMO chain unchanged — but not implemented here. **FDM** (frequency-division) and **PMCW** (fast-time code division — a different waveform, not FMCW) are excluded.

### Cross-cutting structure

- **Two angle paths.** *Dense* — `PhaseShiftBeamformerOp` forms an angle for every RD cell → RDA cube → `CFARDetectorOp` on the cube. `PhaseShiftBeamformerOp` is the only dense estimator (forming a full MVDR/MUSIC cube over every RD cell is prohibitive), so the dense slot is hardcoded. *Sparse* — `CFARDetectorOp` detects first (non-coherently integrating the channel axis), then a per-detection channel snapshot is gathered and a sparse angle estimator resolves one DOA per detection.
- **The sparse angle estimator is a pluggable slot** (`⟨sparse angle estimator⟩` in the chains below), orthogonal to the antenna/multiplexing modality — whatever produced the snapshots (physical RX for SIMO, assembled virtual array for MIMO) feeds the same slot. Any estimator fills it: `BeamscanEstimatorOp` (conventional/Bartlett, the default), `MVDREstimatorOp`, `MUSICEstimatorOp`, `RootMUSICEstimatorOp`, or `ESPRITEstimatorOp`. The choice affects only the estimator's own I/O (single- vs. multi-snapshot input, source count, geometry constraint), not the surrounding graph — see the Angle-estimation enumeration.
- **Clustering placement.** `ClusterDBSCANOp` is orthogonal to modality and attaches at the detection/angle stage in one of two patterns: *angle-then-cluster* (default; cluster in range/velocity/az/el — the only option on the dense path) or, sparse path only, *cluster-then-angle* (cluster in range-Doppler, reduce to one representative per cluster, then estimate angle once per representative).
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
  → MIMOCoherentCompensationOp → (M_tx, N_rx, num_doppler, num_range)  [per-Doppler-bin TX phase correction]
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
  → MIMOCoherentCompensationOp → (M_tx, N_rx, num_doppler_per_tx, num_range)  [demux bands → TX axis, remove code phase, align]
  → [ClutterFilterOp]          → (M_tx, N_rx, num_doppler_per_tx, num_range)  [after demux]
  → VirtualArrayAssemblyOp     → (num_virtual, num_doppler, num_range) + virtual element_positions
  → angle path (dense or sparse) + CFARDetectorOp
```

No `MIMODeinterleaveOp` (nothing interleaved in slow-time). `ClutterFilterOp` runs *after* demux here (the per-chirp TX codes ride on the raw chirps, so a slow-time clutter filter must follow demux). Clustering as in SIMO/TDM.

### Workflow comparison

| Modality   | TX × RX | `MIMODeinterleaveOp` | `MIMOCoherentCompensationOp`  | `VirtualArrayAssemblyOp` | Angle path      | Channel axis      |
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
- **RootMUSICEstimatorOp** — subspace, gridless (polynomial rooting); ULA/uniform geometry; no scan grid, no spectrum.
- **ESPRITEstimatorOp** — subspace, gridless (rotational invariance); shift-invariant geometry; closed-form DOAs.

#### Covariance estimator open items

All covariance estimators share open items: multi-snapshot / spatial-smoothing input form, source-count source (fixed vs. MDL/AIC vs. CFAR-cluster), and — for the gridless pair — a ULA/subarray-selection step on non-uniform virtual arrays.

### MIMO marshaling

Three operators, one per responsibility, spanning TX separation through virtual-array assembly.

- **MIMODeinterleaveOp** — TDM only: split the interleaved slow-time axis into an explicit TX axis (front of chain). Pure data marshaling.
- **MIMOCoherentCompensationOp** — TX separation + phase-aware coherent compensation (TDM: per-Doppler-bin phase correction; DDM: band demux + code removal + alignment). The one genuine MIMO compute step.
- **VirtualArrayAssemblyOp** — flatten the separated `(tx, rx)` block into a virtual channel axis and emit the matching virtual `ArrayGeometry` (`p_virtual = p_tx + p_rx`, with overlap dedup).

### I/O bridges

The small add-on operators that connect detection, clustering, and the sparse angle estimators via plain-tensor ports. Each is filed under its functional group above; their contracts are in *Operator APIs*.

- **`SnapshotGatherOp`** — reads RD maps + `detection_idx` → per-detection channel **snapshot** matrix feeding the sparse angle estimators (sparse angle path; SIMO and both MIMO modalities).
- **`DetectionReductionOp`** — reads cluster labels + `detection_idx` (+ features) → one representative detection per cluster, for the cluster-then-angle (pre-angle) sparse path.
- **`FeatureAssemblyOp`** — maps `detection_idx` (+ bin axes, `detection_score`, angle) → the `points` feature matrix for `ClusterDBSCANOp` / `DetectionReductionOp`.

### Explicitly out of scope for this FMCW pass

**Not** required by the FMCW workflows, so excluded from this lockdown: Pulse-Doppler operators, SAR image-formation operators, and PVA SDK operators. Multiplexing scope (see Workflows → Scope): **CDM / BPM** are under consideration for a future wave (would reuse `MIMOCoherentCompensationOp` in a code-correlation mode, rest of the MIMO chain unchanged); **FDM** and **PMCW** are excluded.

---

## Operator APIs

Per-operator **Inputs**, **Outputs**, and **Parameters**. Every port carries a dtype + shape; every parameter carries a `Default:` (required parameters shown as `Default: none (required)`). Algorithm and Notes are added when each operator is fully locked. Each port lists both precision forms (`complex64 / complex128`, `float32 / float64`); an operator instance uses one, fixed by its template instantiation, with inputs and outputs matching. Leading `...` axes are opaque, preserved batch dims.

**Parameter override convention.** Parameters are configuration set at graph build.

- A parameter tagged **[live]** may additionally be overridden in-flight via a same-named non-blocking input port (unconnected → the parameter value is used).
- Overrides are **sticky**: once a value arrives on the override port it remains the effective value on later ticks until another override arrives; the parameter supplies the initial effective value. (Re-emitting configuration every tick would make the override port a per-tick data port, contradicting "parameters are configuration".)
- The philosophy is to not over-constrain a caller who knows what they are doing: allow a reasonable set of footguns, and warn where relevant.
- Output shapes are inferred from the input tensors each tick (never assumed fixed downstream), so a **[live]** change that alters an output shape is allowed — consumers re-infer.
- Where a **[live]** override incurs a genuine cost — an FFT-plan build, a window-coefficient build, a CFAR window-kernel build — the operator **warns** (once per new configuration) and **caches** prior configurations so alternating values do not thrash.
- A few parameters are genuinely fixed for an instance's lifetime (e.g. the CFAR variant, which selects the kernel) and are **static**; instantiate separate operators to vary those.
- A **[live]** parameter is listed once — here under Parameters, with the tag — never duplicated under Inputs; only true data inputs (the tensors consumed anew each tick) appear under Inputs.

**Bridge operators.** The workflows also use small add-on operators to connect detection, clustering, and the sparse angle estimators. Per the plain-tensor-ports convention, bridging is done by explicit add-on operators (never shared structs). Each carries a **delivery-model** decision — a mainstream library operator, a per-application implementation, or a shared helper called inside another operator's `compute()` — recorded per bridge. Three exist — `SnapshotGatherOp`, `DetectionReductionOp`, and `FeatureAssemblyOp` — grouped into the functional sections below alongside the operators they connect (each flagged in its own Notes).

---

### Signal Processing Operators

#### FMCWRangeProcessorOp

- **In:** `adc_cube` — complex64 / complex128 (IQ IF) **or** float32 / float64 (real IF), `(..., num_chirps, samples_per_chirp)`. Range FFT along the last (fast-time) axis.
- **Out:** `range_profiles` — complex64 / complex128, `(..., num_chirps, num_range_bins)`. `num_range_bins = samples_per_chirp` (complex IF) or `samples_per_chirp//2 + 1` (real IF); grows to `nfft` when set. No range-axis `fftshift` (bin 0 = zero range).
- **Out:** `range_bins` (optional) — float32 / float64, `(num_range_bins,)`. Per-bin range (m) relative to zero beat; emitted only when a waveform descriptor is available.

**Parameters:**

- `range_window` **[live]** — WindowSpec {none, Hann, Hamming, Blackman, Blackman-Harris, Kaiser, Chebyshev, Taylor, flat_top, Tukey, custom}, coherent-gain normalized. Window applied along fast-time before the range FFT. Overriding to an unseen window rebuilds its coefficients and warns once; prior windows are cached. Default: none (rectangular — best resolution, ~−13 dB sidelobes).
- `apply_dechirp` **[live]** — bool. Conjugate-multiply the input against the reference chirp along fast-time before windowing/FFT (for data not dechirped in hardware). Requires a `waveform` and complex-IF input when `true` — software dechirp produces complex samples, which is inconsistent with the real-IF `samples_per_chirp//2 + 1` output size, so real-IF input with `apply_dechirp = true` is an error. Default: `false`.
- `waveform` **[live]** — FMCWWaveformDescriptor. Shared chirp descriptor; source for the dechirp reference and the `range_bins` axis. Overriding supports a waveform that changes between frames (the fused reference is cached, rebuilt only when the descriptor changes). Default: none (required when `apply_dechirp = true` and/or `range_bins` is used; unused otherwise).
- `fft_norm` **[live]** — enum {backward, ortho, forward}. FFT normalization; `backward` (unnormalized forward) is the radar-standard, keeping coherent-integration gain visible. Default: backward.
- `propagation_speed` **[live]** — float. Wave speed for the beat-frequency → range conversion (`range_bins` only). Default: speed of light ≈ 2.998e8 m/s.
- `nfft` **[live]** — int, ≥ `samples_per_chirp`. Fast-time FFT length; larger zero-pads for a finer interpolated range grid over the same span (also the fast-length escape hatch for a large-prime `samples_per_chirp`). The output range-bin count follows `nfft` (downstream infers it). Overriding to an unseen length builds a new FFT plan and warns; seen lengths are reused from the MatX plan cache. Default: none (= `samples_per_chirp`, exact N-point).

#### ClutterFilterOp

- **In:** `range_profiles` — complex64 / complex128, `(..., num_chirps, num_range_bins)`. Slow-time (chirp) axis second-to-last.
- **Out:** `range_profiles` — complex64 / complex128, `(..., num_chirps_out, num_range_bins)`. `num_chirps_out = num_chirps` (`mean_subtraction`), `num_chirps − 1` (`single_delay`), `num_chirps − 2` (`double_delay`).

**Parameters:**

- `method` **[live]** — enum {mean_subtraction, single_delay, double_delay}. Slow-time clutter suppression: `mean_subtraction` nulls zero-Doppler with no chirp loss; the delay cancellers are 1st/2nd-order MTI high-pass (wider near-DC notch, and drop 1–2 chirps). Switching to/from a delay canceller changes `num_chirps_out`; the downstream Doppler stage infers the chirp count, so this is allowed. No setup cost. Default: mean_subtraction.

#### FMCWDopplerProcessorOp

- **In:** `range_profiles` — complex64 / complex128, `(..., num_chirps, num_range_bins)`. Doppler FFT along the `num_chirps` (slow-time) axis.
- **Out:** `rd_maps` — complex64 / complex128, `(..., num_doppler, num_range_bins)`. `num_doppler = num_chirps` (or `nfft`); zero-Doppler centered by default.
- **Out:** `velocity_bins` (optional) — float32 / float64, `(num_doppler,)`. Per-bin radial velocity; emitted only when a descriptor is available.

**Parameters:**

- `doppler_window` **[live]** — WindowSpec. Window along the chirp (slow-time) axis; sidelobe control trades against zero-Doppler clutter leaking into moving-target bins. Overriding to an unseen window rebuilds its coefficients and warns once; prior windows cached. Default: none.
- `center_zero_doppler` **[live]** — bool. `fftshift` the Doppler axis so zero-Doppler is centered (conventional RD layout); else DC at bin 0. Default: `true`.
- `waveform` **[live]** — FMCWWaveformDescriptor. Source for the `velocity_bins` axis (carrier → λ, chirp repetition interval → PRF). Override for a changing waveform. Default: none (required only when `velocity_bins` is used).
- `fft_norm` **[live]** — enum {backward, ortho, forward}. Same convention and rationale as FMCWRangeProcessorOp. Default: backward.
- `nfft` **[live]** — int, ≥ `num_chirps`. Doppler FFT length; larger interpolates the velocity grid over the same span (also the large-prime escape hatch). The output Doppler-bin count follows `nfft` (downstream infers it). Overriding to an unseen length builds a new FFT plan and warns; seen lengths reused from the MatX plan cache. Default: none (= `num_chirps`, exact N-point).

#### FMCWRangeDopplerProcessorOp (fused)

- **In:** `adc_cube` — complex64 / complex128 (IQ IF) **or** float32 / float64 (real IF), `(..., num_chirps, samples_per_chirp)`.
- **Out:** `rd_maps` — complex64 / complex128, `(..., num_doppler, num_range_bins)`. `num_range_bins` per FMCWRangeProcessorOp, `num_doppler = num_chirps` (reduced by the delay-canceller chirp loss when `clutter_filter` is a delay canceller). No intermediate range-profile output; the inline clutter filter is fused into the Doppler pre-FFT expression (no materialized intermediate), bit-identical to `ClutterFilterOp` → `FMCWDopplerProcessorOp`.
- **Out:** `range_bins` (optional) — float32 / float64, `(num_range_bins,)`.
- **Out:** `velocity_bins` (optional) — float32 / float64, `(num_doppler,)`.

**Parameters:** (fused; union of the split operators, same semantics)

- `range_window` **[live]** — WindowSpec. Window along fast-time (range FFT). Overriding to an unseen window rebuilds its coefficients and warns once; cached. Default: none.
- `doppler_window` **[live]** — WindowSpec. Window along the chirp axis (Doppler FFT). Overriding to an unseen window rebuilds its coefficients and warns once; cached. Default: none.
- `apply_dechirp` **[live]** — bool. Software dechirp before the range FFT. Requires a `waveform` and complex-IF input when `true` (same constraint as FMCWRangeProcessorOp). Default: `false`.
- `clutter_filter` **[live]** — enum {none, mean_subtraction, single_delay, double_delay}. Optional inline clutter stage between the range and Doppler stages (same semantics as ClutterFilterOp `method`); the delay cancellers reduce `num_doppler` (downstream infers it). No setup cost. Default: none.
- `waveform` **[live]** — FMCWWaveformDescriptor. Dechirp reference and the `range_bins` / `velocity_bins` axes. Override for a changing waveform. Default: none (required for software dechirp and/or those outputs).
- `fft_norm` **[live]** — enum {backward, ortho, forward}. One normalization convention for both transforms. Default: backward.
- `center_zero_doppler` **[live]** — bool. Center zero-Doppler on the Doppler axis. Default: `true`.
- `propagation_speed` **[live]** — float. For the `range_bins` output. Default: speed of light ≈ 2.998e8 m/s.
- `nfft_range`, `nfft_doppler` **[live]** — int, each ≥ its natural length. Per-axis FFT lengths (interpolation / large-prime escape hatch); the output bin counts follow them (downstream infers). Overriding to an unseen length builds a new FFT plan and warns; seen lengths reused from the plan cache. Default: none (= exact `samples_per_chirp` / `num_chirps`).

---

### Detection & Clustering Operators

#### CFARDetectorOp

- **In:** `rd_maps` — complex64 / complex128, **or** a real float32 / float64 power map, `(..., num_doppler, num_range_bins)`. CFAR plane = last two axes; leading axes batched/integrated per `nci_axes`. The path is auto-selected by input dtype (no parameter, mirroring FMCWRangeProcessorOp): **complex** → form the square-law power `|·|²`; **real** → treat as an already-formed power map and threshold it directly.
- **Out:** `detection_mask` — bool, `(..., num_doppler, num_range_bins)` (axes in `nci_axes` reduced; other leading axes preserved). `True` where detected — equivalently `detection_score > 1`.
- **Out:** `detection_score` — float32 / float64, same shape as `detection_mask`. The CFAR test statistic normalized by its per-cell adaptive threshold: `detection_score = P / (threshold_scale · α · noise_estimate)`, where `P` is the cell's power statistic (`|·|²`, or the NCI sum). `> 1` ⇔ detected; the value is the detection margin, and since the threshold tracks local noise it is proportional to the cell SNR — so it doubles as the per-detection **strength / ranking score** used downstream (gathered by feature-assembly into `points` for `ClusterDBSCANOp` / `DetectionReductionOp`). Replaces the former `detection_threshold` output (the absolute threshold is recoverable from `noise_estimate` and the params).
- **Out:** `noise_estimate` — float32 / float64, same shape as `detection_mask`. Per-cell noise from the training cells (`N`); kept for absolute calibration / diagnostics.
- **Out:** `detection_idx` — int, `(num_detections, 2 [+ n_batch])` — `(doppler_bin, range_bin)` per row (widening to include any preserved leading batch axes, or emitted per batch slice). Bare indices only. **Always emitted**, sized to the actual detection count and reallocated per tick — the row count *is* `num_detections`, so there is no companion scalar. If `max_detections` is set and the scene exceeds it, the surplus is dropped and a warning is raised.

**Parameters:**

- `cfar_variant` — enum {CA, OS, GO, SO}. Noise-estimation rule: CA (cell-averaging), GO (greatest-of), SO (smallest-of), OS (order-statistic). **Static** — selects the kernel; to run different variants, instantiate separate detectors. Default: CA.
- `os_rank` **[live]** — int. Order-statistic rank into the sorted training cells; used only when `cfar_variant = OS`. Default: none (required when OS; ≈ 0.75·N training cells is typical).
- `guard_cells` **[live]** — int per axis `(doppler, range)`. Guard extent each side of the cell under test (excluded from the noise estimate). Overriding to an unseen window footprint may rebuild the CFAR window kernel and warns; up to `max_cached_kernels` footprints are retained. Default: none (required).
- `training_cells` **[live]** — int per axis `(doppler, range)`. Training / reference extent each side, for the noise estimate. Also updates the training-cell count `N` (and thus α — a cheap scalar recompute). Same window-kernel rebuild / warn / cache behavior as `guard_cells`. Default: none (required).
- `pfa` **[live]** — float. Probability of false alarm; sets the per-variant threshold multiplier α (CA mean: `α = N·(pfa^(−1/N) − 1)`). Cheap scalar recompute. Default: 1e-6.
- `threshold_scale` **[live]** — float. Empirical multiplier on top of the Pfa-derived threshold (`threshold = threshold_scale · α · noise_estimate`); `1.0` = pure CFAR. Cheap; hand-tuning without changing the CFAR model. Default: 1.0.
- `nci_axes` **[live]** — set/vector of leading-axis indices to non-coherently integrate before thresholding; other leading axes stay batched. Changes the output shape (integrated axes reduced — downstream infers) and the integration fold count `K = ∏ sizes of nci_axes`, which sets the cell statistic to Gamma(K); α takes `K` and is recomputed by closed form (cheap). Default: `[]` (no integration; `K = 1`, exponential cell statistic).
- `max_detections` (optional) **[live]** — int. Upper cap on the number of detections emitted in `detection_idx`; if the scene produces more, the surplus is dropped and a warning raised. `detection_idx` is always sized to the actual count (≤ this cap, reallocated per tick). Unset → no cap (emit all detections). Default: none (uncapped).
- `max_cached_kernels` — int. How many distinct guard/training window-footprint kernels to retain before evicting (and rebuilding on reuse). Static. Default: 4.

> ### PROPOSED — needs team review (not locked)
>
> The block below is a **proposed contract modification** for `CFARDetectorOp`
> calibration. It is **not** accepted lockdown text until the team reviews and
> explicitly adopts it. Implementation on `dpc/detection` already follows this
> proposal so correctness/perf work can proceed; do not treat presence in this
> file as approval. Tracking:
> [`operators/detection/cfar_detector_op/OPEN_ITEMS.md`](../../operators/detection/cfar_detector_op/OPEN_ITEMS.md)
> §7.
>
> **What changes if adopted** (replaces / narrows the locked bullets above for
> the named parameters; adds `alpha_lut`):
>
> - `pfa` **[live]** **[PROPOSED]** — float. Probability of false alarm; sets
>   the per-variant threshold multiplier α when `alpha_lut` is unset. CA and
>   OS with `K = 1` calibrate α by closed-form inversion (cheap). GO, SO, and
>   OS with `K > 1` may trigger Monte Carlo LUT calibration on cache miss
>   (seconds on large footprints unless a precomputed `alpha_lut` is supplied
>   or `blocking_alpha_lut_rebuild = false` serves a stale LUT while the new
>   table builds). Default: 1e-6.
> - `alpha_lut` **[live]** **[PROPOSED — new parameter]** — optional float
>   vector. Precomputed threshold multiplier LUT. When non-empty, used
>   directly and `pfa` is ignored for α calibration. Indexing matches the
>   runtime stencil:
>   - **CA / OS:** `alpha_lut[n]` where `n` is the in-bounds training-cell
>     count at the cell under test.
>   - **GO / SO:** `alpha_lut[i]` where `i = go_lut_index(range_neg,
>     range_pos, doppler_neg, doppler_pos)` — the compact side-count key used
>     by the GOGO/SOSO stencil (`cfar_alpha.hpp`). Per axis: if both half-sides
>     are alive use `min(neg, pos)`, else the alive side’s count; then if both
>     axes are alive use `min(range_side, doppler_side)`, else the alive axis.
>     Distinct border geometries can share one `i`; a single α serves all of
>     them (`pfa`-derived builds take the max α across aliased geometries so
>     every cell at that index meets the target Pfa). Offline tables must use
>     this same `go_lut_index` — not raw per-side length and not CA-style `N`.
>   Slot 0 is unused. Length must exceed the maximum LUT index for the
>   configured variant, footprint, and map size (`len > max_n`). CA/OS require
>   every index in `1..max_n` to be finite and strictly positive (slot 0 and
>   any past `max_n` may be zero). GO/SO allow zeros at unreachable indices
>   but require every reachable `go_lut_index` to be finite and strictly
>   positive; negative or non-finite entries are always rejected. Cheap:
>   validation and host→device upload only. Default: `[]` (unset →
>   `pfa`-derived calibration). Precedence: `alpha_lut` if provided, else
>   `pfa`-derived calibration.
> - `blocking_alpha_lut_rebuild` **[PROPOSED — new parameter]** — bool. When
>   `true`, a `pfa`-derived LUT cache miss blocks the tick until calibration
>   finishes. When `false` (default), an unseen key starts a background build
>   and the previous LUT is served for the tick (warn once); if no previous
>   LUT exists yet, the tick still blocks on the first build. Has no effect
>   while an external `alpha_lut` is supplied. Static. Default: `false`.
> - `threshold_scale` **[live]** **[PROPOSED wording]** — float. Empirical
>   multiplier on top of the active α source
>   (`threshold = threshold_scale · α · noise_estimate`); `1.0` = pure CFAR.
>   Cheap; hand-tuning without changing the CFAR model. Default: 1.0.
> - `nci_axes` **[live]** **[PROPOSED wording]** — set/vector of leading-axis
>   indices to non-coherently integrate before thresholding; other leading
>   axes stay batched. Changes the output shape (integrated axes reduced —
>   downstream infers) and the integration fold count
>   `K = ∏ sizes of nci_axes`, which sets the cell statistic to Gamma(K); α
>   takes `K` and is recomputed from `pfa` (closed form for CA; Monte Carlo
>   for GO/SO/OS when `K > 1` and `alpha_lut` is unset). Default: `[]` (no
>   integration; `K = 1`, exponential cell statistic).
> - `training_cells` **[live]** **[PROPOSED wording note]** — the locked
>   “cheap scalar recompute” claim for α is false for GO/SO and OS with
>   `K > 1` under `pfa`-derived calibration; adopt the narrowed `pfa` /
>   `alpha_lut` wording above instead.
>
> **[PROPOSED] Future calibration fast paths.** If `alpha_lut` is adopted as
> the v1 explicit fast path (same per-index threshold semantics as
> `pfa`-derived calibration), other user-facing fast paths remain deferred and
> are **not** part of this proposal:
>
> - an interior-cell scalar `alpha` for homogeneous maps where border exactness
>   is not required;
> - scalar `alpha` plus simplified edge-handling modes, such as cropping the
>   CFAR-valid region, treating border cells as non-detections, or accepting
>   approximate border thresholds;
> - hybrid scalar/table modes, such as using scalar `alpha` for fully
>   in-bounds interior cells and a small border LUT for reduced-`N` edge
>   cells;
> - hybrid scalar/policy modes, such as scalar `alpha` for the interior plus
>   an explicit edge policy that crops, suppresses, approximates, or
>   delegates border handling to a downstream stage;
> - helper utilities that generate `alpha_lut` tables offline from
>   `(variant, pfa, K, os_rank, guard_cells, training_cells)`;
> - domain-specific table packaging or deployment conventions outside the
>   operator API.
>
> **End proposed block.** Until team review lands, treat the locked parameter
> bullets above this banner as the contract source of truth for review of
> non-calibration topics; treat the proposed block as the working text the
> implementation is targeting.

#### ClusterDBSCANOp

- **In:** `points` — float32 / float64, `(num_points, num_features)`. Caller-assembled feature matrix (e.g. `[range, velocity, az, el]`); radar-agnostic.
- **Out:** `labels` — int, `(num_points,)`. Cluster id per point, noise = `-1`; row-aligned with `points`.
- **Out:** `sizes` (optional) — int, `(num_clusters,)`. Point count per cluster; sized to the actual cluster count and reallocated per tick. If `max_clusters` is set and exceeded, surplus clusters are dropped and a warning raised.

**Parameters:**

- `epsilon` **[live]** — float scalar or `(num_features,)` vector. Neighborhood radius per feature axis; a per-axis vector gives an axis-aligned (anisotropic) neighborhood — `q` is a neighbor of `p` when `Σ_i ((p_i − q_i) / epsilon_i)² ≤ 1` — which handles mixed-unit features (m, rad, m/s). A scalar applies isotropically. Overridable for runtime tuning. Default: none (required).
- `min_points` **[live]** — int. Minimum neighbors (including the point itself) for a core point (DBSCAN `minPts`). Overridable. Default: `2 · num_features` (common heuristic).
- `max_clusters` (optional) **[live]** — int. Upper cap on the number of clusters reported in `sizes`; surplus dropped and a warning raised if exceeded. `sizes` is sized to the actual cluster count (≤ this cap). Unset → no cap. Default: none (uncapped).

#### FeatureAssemblyOp

Assembles the `points` feature matrix for clustering (and `DetectionReductionOp`) by gathering each detection's physical attributes from its cell — mapping CFAR's bare bin indices to physical range / velocity, gathering the detection strength, and appending angle when available. Pure gather / lookup — the feature-engineering step between detection and clustering.

- **In:** `detection_idx` — int, `(num_points, 2 [+ n_batch])`. The `(doppler_bin, range_bin)` cells (CFAR). Row order sets the output row order (so `ClusterDBSCANOp.labels` join back positionally).
- **In:** `range_bins` (optional) — float32 / float64, `(num_range,)`. Physical range (m) per range bin (`FMCWRangeProcessorOp.range_bins`); indexed by the `range_bin` column for the range feature. Absent → the raw bin index is used.
- **In:** `velocity_bins` (optional) — float32 / float64, `(num_doppler,)`. Physical velocity (m/s) per Doppler bin (`FMCWDopplerProcessorOp.velocity_bins`); indexed by `doppler_bin`. Absent → the raw bin index.
- **In:** `detection_score` (optional) — float32 / float64, `(..., num_doppler, num_range)`. CFAR's per-cell score map; gathered at each detection cell → the strength feature.
- **In:** `aoa_estimates` (optional) — float32 / float64, `(num_points, 2)`. Per-detection `(az, el)` (post-angle path), row-aligned with `detection_idx`; supplies the angle features.
- **Out:** `points` — float32 / float64, `(num_points, num_features)`. Assembled feature matrix, columns in the configured order, sized to the detection count. Feeds `ClusterDBSCANOp.points` and `DetectionReductionOp.points`.

**Parameters:**

- `features` — ordered list over {range, velocity, score, az, el} (plus raw {doppler_bin, range_bin} as escape hatches). Selects which columns to assemble and their order, fixing `num_features` and the layout. Each token draws from its matching input (range/velocity from the bin-axis lookups, score from `detection_score`, az/el from `aoa_estimates`); selecting a token whose input is absent is an error. Static (defines the output layout). Default: none (required).

**Algorithm:**

1. For each `detection_idx` row, read `(doppler_bin, range_bin)`.
2. Build each configured feature: `range = range_bins[range_bin]` (or the raw bin), `velocity = velocity_bins[doppler_bin]` (or raw), `score =` gather `detection_score` at the cell, `az` / `el` `= aoa_estimates[row]`.
3. Stack the columns in `features` order → `points`. Pure gather / 1-D
   lookup (custom CUDA; MatX is a poor fit for sparse irregular gathers);
   no scaling.

**Notes:**

- *Delivery model — mainstream thin operator:* the gather + bin→physical lookup + column stack is mechanical and reusable, so it ships as a **mainstream operator with a configurable `features` list**, reusing the same gather primitive as `SnapshotGatherOp`. It is the most application-specific of the three bridges (which columns to cluster on is a modeling choice), so a per-application implementation remains a reasonable alternative; the configurable op is the default.
- *No scaling here:* `points` carries raw physical columns; per-axis scaling for clustering / reduction lives in `ClusterDBSCANOp.epsilon` and `DetectionReductionOp.feature_weights`, so mixed units are handled once, downstream — not baked into the matrix.
- *Placement:* pre-angle (cluster-then-angle) uses `features = [range, velocity, score]` (no `aoa_estimates`); post-angle (angle-then-cluster) adds `[az, el]`. The `features` list + optional `aoa_estimates` cover both without a variant.
- *Shared gather primitive:* the same per-detection gather as `SnapshotGatherOp` (which pulls channel vectors); here it pulls scalars and does the 1-D bin-axis lookups. One primitive, two bridges.
- *Row alignment:* `points` rows are 1:1 with `detection_idx` rows, so `labels` positionally join back and `DetectionReductionOp.rep_idx` resolves to real cells.
- *Carried over:* plain-tensor ports; precision by template instantiation;
  device tensors on the Holoscan CUDA stream (custom CUDA gather kernel).

#### DetectionReductionOp

Reduces each cluster to one **representative detection** so the cluster-then-angle path estimates angle once per cluster instead of once per detection. Selects a real cluster member (a detected cell) by a configurable policy; a thin wrapper over the shared group-by-label reduction primitive.

- **In:** `labels` — int, `(num_points,)`. Cluster id per detection (noise = `-1`), row-aligned with `detection_idx`. From `ClusterDBSCANOp.labels`.
- **In:** `detection_idx` — int, `(num_points, 2 [+ n_batch])`. The detected cells, row-aligned with `labels` (CFAR's `detection_idx`).
- **In:** `points` — float32 / float64, `(num_points, num_features)`. The per-detection feature matrix — the same one clustered on, augmented with any columns to select by (power, SNR, …). Both the `peak` score and the `medoid` distance are derived from these columns (via `feature_weights` / `score_method`); `first` ignores it. From the feature-assembly bridge (the same matrix as `ClusterDBSCANOp.points`). Required unless `representative = first`.
- **Out:** `rep_idx` — int, `(num_clusters, 2 [+ n_batch])`. One representative cell per cluster — a subset of `detection_idx` rows, sized to the actual cluster count (reallocated per tick). Feeds `snapshot-gather.detection_idx`. Noise points (label `-1`) form no cluster and yield no representative.

**Parameters:**

- `representative` **[live]** — enum {peak, medoid, first}. Which member to keep: `peak` = the member maximizing a feature-derived score (see `score_method`; the recommended default — weight it toward power/SNR for the best-SNR snapshot); `medoid` = the member nearest the cluster's feature centroid; `first` = lowest-index member. Always a real member (never a synthetic centroid). Default: peak.
- `score_method` **[live]** — enum {weighted_sum, weighted_l2, weighted_max, custom}. For `peak`, how the weighted feature columns collapse to the scalar score that is maximized: `weighted_sum` = Σ wᵢ·featureᵢ; `weighted_l2` = ‖w ⊙ feature‖₂; `weighted_max` = maxᵢ wᵢ·featureᵢ; `custom` = the caller's `score_fn` functor. A one-hot `feature_weights` reduces `weighted_sum` to "max of a single column" (e.g. peak power). Used only by `peak`. Default: weighted_sum.
- `feature_weights` **[live]** — float `(num_features,)`. Per-feature weights `w`. For `peak` they weight the columns before `score_method` combines them; for `medoid` they scale the per-axis centroid distance (mixed-unit weighting — the same idea as `ClusterDBSCANOp.epsilon`, best matched to it). Default: none (required for `peak` / `medoid`; mixed-unit features have no universal default).
- `score_fn` — custom scoring functor, used only when `score_method = custom`. A **compile-time** device functor (a template parameter, like precision), mapping a length-`num_features` feature row to a scalar — not a runtime value. Default: none.

**Algorithm:**

1. Group the detection rows by `labels`, excluding noise (`-1`).
2. Per cluster, select the representative row by `representative`, via the shared group-by-label reduction primitive: `peak` = argmax of the per-row score (`score_method` applied to `feature_weights ⊙ points`, or the `score_fn` functor); `first` = lowest index; `medoid` = form the centroid as the (segmented) mean of the `feature_weights`-scaled `points`, then take the argmin scaled-distance member (two segmented passes — a mean, then an argmin).
3. Emit `rep_idx` = the selected rows of `detection_idx`, one per cluster, sized to the cluster count. Pure gather + segmented reduction (custom CUDA atomics / compact in the current implementation; CUB/MatX segmented reduce is a staged replacement).

**Notes:**

- *Delivery model — thin operator over a shared library primitive:* the generic group-by-label reduction (segmented argmax/argmin/medoid, and aggregate stats) is a **public library function** any operator can call — the reusable core.
  - Cluster-reduction ships as a **thin standalone operator** wrapping it, **not** folded into `ClusterDBSCANOp`: it combines `labels` + radar `detection_idx` + a `points`-derived score and emits representative *cell* indices, which would pollute the deliberately radar-agnostic clusterer (a `points`-only contract), and the representative *policy* is consumer-specific ("no universal representative").
  - Keeping it a node also makes the cluster-then-angle path wireable out of the box. The primitive stays public, so a future clustering op could call it directly if a strong case appears.
- *Representative is a real cell:* every policy selects an actual cluster member, so `rep_idx` addresses a real RD cell for `SnapshotGatherOp`. A mean/centroid is deliberately excluded — it is not a detected cell and cannot be gathered from.
- *Score source:* the `peak` score is derived from `points` columns (via `feature_weights` / `score_method`), so any selection attribute — power, SNR — must be present as a feature column. Supplying it is the feature-assembly bridge's job (it builds `points` from detection attributes), which settles where the score comes from: a column of the same matrix used for clustering, not a separate input. The natural strength column is CFAR's `detection_score` (SNR-proportional), gathered per detection by feature-assembly — but the operator is not beholden to it: it reduces over whatever columns `points` carries, so any feature (or `score_fn` of several) can drive selection.
- *Custom scoring:* `score_method = custom` uses a compile-time device functor (`score_fn`) — the module's template-based customization, GPU-callable — not a runtime callback. The built-in methods cover weighted feature combinations; the functor covers anything else (nonlinear scores, ratios, etc.).
- *Feature centroid (medoid only):* the centroid is computed **inside the operator** — the per-cluster mean of the `feature_weights`-scaled `points` — and the medoid is the member nearest it (min scaled distance). It is never emitted (not a real cell). Matching `feature_weights` to the upstream DBSCAN `epsilon` keeps "nearest to centroid" consistent with the metric that formed the cluster. Both `peak` and `medoid` read `points` now; only `first` is feature-free.
- *Carried over:* plain-tensor ports; precision by template instantiation; MatX + MatX memory spaces.

---

### Angle Estimation Operators

#### PhaseShiftBeamformerOp (dense)

- **In:** `rd_maps` — complex64 / complex128, `(..., num_channels, num_doppler, num_range_bins)`. The channel axis sits immediately before the `(doppler, range)` plane and is reduced into the beam axes; `num_channels` must match `array_geometry`.
- **Out:** `rda_cube` — complex64 / complex128, `(..., num_az, num_el, num_doppler, num_range_bins)`. Channel axis replaced by the `az × el` beam grid (or `u × v` under the `direction_cosine` convention); feeds CFARDetectorOp's dense path directly.

**Parameters:**

- `array_geometry` **[live]** — ArrayGeometry (virtual-element positions + wavelength). `num_elements` must match the `num_channels` axis (mismatch errors). Overriding (reconfigurable aperture / changed wavelength) rebuilds the beam weights and warns once; cached, keyed on the descriptor. Default: none (required).
- `angle_convention` — AngleConvention. How `az_angles` / `el_angles` and the output beam axes map to unit vectors in the array frame, or `direction_cosine` to scan in sine-space `(u, v)`. **Static** — it fixes the *meaning* of the output angles for the whole graph; vary it by instantiating a separate operator. Default: none (required).
- `az_angles` **[live]** — float vector. Azimuth scan grid (the `u` grid under `direction_cosine`). Overriding steers the scan and rebuilds weights (cached). Default: none (required).
- `el_angles` **[live]** — float vector. Elevation scan grid (the `v` grid under `direction_cosine`); a single value (e.g. `[0]`) gives a 1-D azimuth scan. Default: none (required).
- `array_taper` **[live]** — WindowSpec. Amplitude taper across the elements for spatial sidelobe control (aperture shading); Taylor / Chebyshev are canonical. Overriding rebuilds weights (cached). Default: none.

*(The beam weights W depend on `array_geometry`, `az_angles`, `el_angles`, and `array_taper`; overriding any of them rebuilds W — cached keyed on the inputs, so an unchanged override is free and a continuously-steered grid rebuilds as its expected per-tick cost.)*

#### SnapshotGatherOp

Gathers, for each detection, the per-channel complex snapshot(s) from the RD maps — the channel vector at the detected cell plus a configurable range-Doppler neighborhood — into the `snapshot` matrix the sparse angle estimators consume. Pure indexed gather; no arithmetic.

- **In:** `rd_maps` — complex64 / complex128, `(..., num_channels, num_doppler, num_range)`. The **per-channel** RD maps (SIMO: physical RX; MIMO: the assembled virtual array), channel axis immediately before the `(doppler, range)` plane, leading axes batched. Must be the pre-integration maps — *not* the NCI-integrated maps CFAR thresholded on (the channel axis has to be intact to gather from).
- **In:** `detection_idx` — int, `(num_detections, 2 [+ n_batch])`. Detected `(doppler_bin, range_bin)` cells — CFAR's `detection_idx` contract (sized to the actual count; its row count *is* `num_detections`).
- **Out:** `snapshot` — complex64 / complex128, `(num_detections, num_channels, num_snapshots)`. Row *i* is detection *i*'s `(num_channels, num_snapshots)` block; sized to the actual detection count and reallocated per tick (no fixed capacity / padding). `num_snapshots = (2·doppler_half + 1)·(2·range_half + 1)` from `snapshot_window` (1 at the default) — the full window product for every detection regardless of edge mode; under a `clamp` edge, out-of-bounds slots are left unfilled and flagged in `is_valid`. Feeds `⟨sparse angle estimator⟩.snapshot`.
- **Out:** `is_valid` (optional) — bool, `(num_detections, num_snapshots)`. Per-detection mask of which snapshot slots hold a real gathered cell — all true except where a `clamp` edge truncated the window (ragged). Consumed by the sparse estimators to restrict the covariance / beamforming to valid snapshots; emit it whenever either axis uses `clamp`.

**Parameters:**

- `snapshot_window` **[live]** — int per axis `(doppler, range)`, the neighborhood half-extent gathered around each detection. `(0, 0)` → one snapshot (the detection cell itself). `num_snapshots = (2·doppler_half + 1)·(2·range_half + 1)`. Overriding changes `num_snapshots` (an output-shape change — downstream infers); cheap (a gather, no setup). Default: `(0, 0)` (single-snapshot — matches Beamscan).
- `doppler_edge` — enum {wrap, clamp, shift}. Neighborhood behavior at the Doppler-axis edges. `wrap`: periodic — bin 0 is adjacent to bin `num_doppler − 1` in the aliased spectrum (physically correct for the Doppler FFT axis). `clamp`: truncate the window at the axis bound, yielding fewer real snapshots near the edge (ragged) — the missing slots are flagged in `is_valid`. `shift`: slide the window inward to keep `num_snapshots` constant using real distinct cells (asymmetric about the detection near an edge). Static. Default: wrap.
- `range_edge` — enum {wrap, clamp, shift}, the same three behaviors as `doppler_edge`. (`wrap` is offered for symmetry but is rarely physical for range — range bins are not periodic.) Static. Default: shift.

**Algorithm:**

1. Take the `detection_idx` rows (its row count is `num_detections`).
2. For each detection `(doppler_bin, range_bin)`, enumerate the neighborhood cells per `snapshot_window`, resolving each axis's out-of-bounds indices by its edge mode (`wrap` / `clamp` / `shift`).
3. Gather the length-`num_channels` complex vector at each in-bounds neighborhood cell from `rd_maps` and stack into a `(num_channels, num_snapshots)` block.
4. Write it to `snapshot[i]`, marking any `clamp`-truncated slots `false` in `is_valid` (all-true otherwise). The output is sized to `num_detections` (allocated per tick). Pure MatX gather/indexing; leading batch axes handled per `detection_idx`'s index width.

**Notes:**

- *Delivery model — mainstream operator:* the gather is generic, plain-tensor in / plain-tensor out, and reusable across any CFAR + any sparse estimator, so it ships as a **mainstream library operator** (`SnapshotGatherOp`), not per-application code.
- *Snapshot source scope:* RD-neighborhood only. **Spatial smoothing** is applied downstream in the estimator (`spatial_smoothing` param), operating on the gathered channel axis. **Multi-frame / temporal** snapshots would need cross-tick state — out of scope for this stateless bridge (a future stateful variant could add them).
- *Neighbor assumption:* the window cells are assumed to share the detection's DOA (target energy straddles adjacent bins from windowing/straddle). Valid for point and modestly-extended targets over a few bins; a large window blends distinct sources and biases the covariance — the caller's footgun to size sensibly.
- *Channel axis intact:* wire from the per-channel `rd_maps` (which CFAR also reads with `nci_axes = {channel}`), not from an integrated map. `num_channels` flows through unchanged and must match the estimator's `array_geometry`.
- *Sizing:* `snapshot` (and `is_valid`) are sized to the actual `num_detections` and reallocated per tick — no fixed-capacity padding, no valid-count companion. `num_detections` is just the row count of the incoming `detection_idx`.
- *Beamscan:* the default `(0, 0)` window yields `num_snapshots = 1`, which Beamscan accepts (its 2D-or-3D-with-`num_snapshots=1` rule).
- *Carried over:* plain-tensor ports; precision by template instantiation; MatX + MatX memory spaces.

#### Sparse angle estimators (shared slot)

`BeamscanEstimatorOp`, `MVDREstimatorOp`, `MUSICEstimatorOp`, `RootMUSICEstimatorOp`, and `ESPRITEstimatorOp` fill the same pluggable sparse slot. They share one **mandatory input port** and one **mandatory output port** so any estimator drops into the slot without rewiring; estimator-specific capabilities (spectrum, multi-source, diagnostics) ride on **separate optional ports** that a common consumer can ignore.

**Shared inputs:**

- `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`. One `(num_channels, num_snapshots)` block per item (detection or cluster representative). The covariance estimators form `R̂ = (1/K) Σ_k x_k·x_kᴴ` over the valid snapshots per item; single-snapshot Beamscan uses one look. `num_channels` must match `array_geometry`. Produced by `SnapshotGatherOp` (an RD-neighborhood gather).
- `is_valid` (optional) — bool, `(num_items, num_snapshots)`. Per-item mask of usable snapshot slots (from `SnapshotGatherOp`'s ragged `clamp` edge mode). When present, the covariance / beamforming sums only over the marked slots (so the effective snapshot count `K` is the per-item valid count); absent → all `num_snapshots` are used.

**Shared mandatory output:**

- `aoa_estimates` — float32 / float64, `(num_items, 2)`. The single dominant DOA `(az, el)` per item (or `(u, v)` under the `direction_cosine` convention), row-aligned with the input items and named per `angle_convention`. This is the common output every estimator always provides.

**Optional ports (per estimator, where meaningful):**

- `angle_spectrum` — float32 / float64, `(num_items, num_az, num_el)`. Grid-scan estimators only (Beamscan, MVDR, MUSIC); the gridless pair emit none.
- `doa_multi` — float32 / float64, `(num_items, max_sources, 2)`, with a companion `source_count` — int, `(num_items,)`. All resolved DOAs for the multi-source estimators; fixed capacity (`max_sources`), unused rows flagged (NaN). `aoa_estimates` is the strongest of these.
- `eigenvalues` — float32 / float64, `(num_items, num_channels)`. Covariance eigenvalue spectrum for subspace / source-count diagnostics (covariance estimators).

**Shared parameters** (all five):

- `array_geometry` **[live]** — ArrayGeometry (element positions + wavelength). `num_elements` must match the snapshot `num_channels` (mismatch errors). Overriding (reconfigurable aperture / changed wavelength) rebuilds the steering weights and warns once; cached, keyed on the descriptor. Default: none (required).
- `angle_convention` — AngleConvention. How the scan grid (grid-scan estimators) and the output `(az, el)` map to unit vectors in the array frame; `direction_cosine` for sine-space `(u, v)`. **Static** — it fixes the *meaning* of the output angles for the whole graph, so it is not live-overridable; vary it by instantiating a separate estimator. Default: none (required).

Grid-scan estimators (Beamscan, MVDR, MUSIC) additionally take a scan grid; the gridless pair (Root-MUSIC, ESPRIT) have none:

- `az_angles` **[live]** — float vector. Azimuth scan grid (the `u` grid under `direction_cosine`). Overriding steers the scan and rebuilds weights (cached). Default: none (required for grid-scan estimators).
- `el_angles` **[live]** — float vector. Elevation scan grid (the `v` grid under `direction_cosine`); a single value (e.g. `[0]`) gives a 1-D azimuth scan. Default: none (required for grid-scan estimators).

Per-estimator deviations from the shared contract:

##### BeamscanEstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels)` **or** `(num_items, num_channels, num_snapshots)`. Beamscan is single-snapshot: a 3D input with `num_snapshots ≠ 1` is an error (a 2D input is the canonical form).
- **Out:** `aoa_estimates` — `(num_items, 2)`.
- **Out:** `angle_spectrum` (optional) — `(num_items, num_az, num_el)`. Single-source, single-snapshot — no `doa_multi`, no `eigenvalues`.

**Parameters:** shared (`array_geometry`, `angle_convention`, `az_angles`, `el_angles`) plus:

- `array_taper` **[live]** — WindowSpec. Amplitude taper across the elements for spatial sidelobe control (aperture shading). Overriding rebuilds weights (cached). Default: none.
- `peak_interpolation` **[live]** — enum {none, parabolic}. Sub-grid peak refinement: `parabolic` fits a quadratic around the grid maximum for sub-bin accuracy; `none` returns the argmax cell. Default: none.

##### MVDREstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`. `num_snapshots` should be ≳ `num_channels` for a well-conditioned (invertible) covariance.
- **Out:** `aoa_estimates` — `(num_items, 2)`.
- **Out:** `angle_spectrum` (optional) — Capon spectrum, `(num_items, num_az, num_el)`.
- **Out:** `doa_multi` + `source_count` (optional); `eigenvalues` (optional).

**Parameters:** shared (`array_geometry`, `angle_convention`, `az_angles`, `el_angles`) plus:

- `diagonal_loading` **[live]** — float or `auto`. Diagonal loading added to the covariance for robustness / invertibility (`R + δI`); `auto` scales δ to a small multiple of the estimated noise floor. Default: auto.
- `spatial_smoothing` **[live]** — int or none. Forward (sub-array averaging) spatial smoothing to decorrelate coherent multipath and raise the covariance rank, at the cost of effective aperture; requires a uniform sub-array geometry. Default: none.
- `max_sources` **[live]** — int. Capacity of the optional `doa_multi` output — the number of Capon-spectrum peaks to report (`aoa_estimates` is always the strongest). Overriding reallocates `doa_multi` (minor). Default: 1.

##### MUSICEstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`.
- **Out:** `aoa_estimates` — `(num_items, 2)`.
- **Out:** `angle_spectrum` (optional) — MUSIC pseudo-spectrum, `(num_items, num_az, num_el)`.
- **Out:** `doa_multi` + `source_count` (optional); `eigenvalues` (optional).

**Parameters:** shared (`array_geometry`, `angle_convention`, `az_angles`, `el_angles`) plus:

- `num_sources` **[live]** — int or `auto`. Signal-subspace dimension (model order). `auto` estimates it from the covariance eigenvalues (MDL/AIC); an integer fixes it; it may also be seeded from the CFAR/cluster count upstream. Default: auto.
- `spatial_smoothing` **[live]** — int or none. As MVDR — decorrelates coherent multipath at the cost of aperture; requires a uniform sub-array geometry. Default: none.
- `max_sources` **[live]** — int. Capacity of the optional `doa_multi` / `source_count` outputs (holds up to `min(num_sources, max_sources)` DOAs). Default: 4.

##### RootMUSICEstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`. Requires a ULA / uniform (Vandermonde) array geometry.
- **Out:** `aoa_estimates` — `(num_items, 2)`. Gridless (DOAs from polynomial roots) — **no `angle_spectrum`**.
- **Out:** `doa_multi` + `source_count` (optional); `eigenvalues` (optional).

**Parameters:** shared `array_geometry` + `angle_convention` (**no scan grid**) plus:

- `num_sources` **[live]** — int or `auto`. Model order (root count); `auto` via MDL/AIC on the covariance eigenvalues. Default: auto.
- `spatial_smoothing` **[live]** — int or none. Also the standard way to apply Root-MUSIC on a partly-filled / holey virtual array — select uniform sub-arrays. Default: none.
- `max_sources` **[live]** — int. `doa_multi` / `source_count` capacity. Default: 4.

Requires a ULA / uniform (Vandermonde) geometry; a non-uniform virtual array needs a uniform sub-array selection (or spatial smoothing) first, and native 2-D az/el needs a 2-D-capable variant.

##### ESPRITEstimatorOp

- **In:** `snapshot` — complex64 / complex128, `(num_items, num_channels, num_snapshots)`. Requires a shift-invariant geometry (ULA or two identical displaced subarrays).
- **Out:** `aoa_estimates` — `(num_items, 2)`. Gridless (closed-form eigenvalue rotation) — **no `angle_spectrum`**.
- **Out:** `doa_multi` + `source_count` (optional); `eigenvalues` (optional).

**Parameters:** shared `array_geometry` + `angle_convention` (**no scan grid**) plus:

- `num_sources` **[live]** — int or `auto`. Model order; `auto` via MDL/AIC on the covariance eigenvalues. Default: auto.
- `spatial_smoothing` **[live]** — int or none. Decorrelates coherent multipath at the cost of aperture; requires a uniform sub-array geometry. Default: none.
- `max_sources` **[live]** — int. `doa_multi` / `source_count` capacity. Default: 4.
- `subarray_spec` — the two shift-invariant sub-arrays / displacement defining the rotational invariance. **Static** (structural). Default: derived from `array_geometry` (maximum-overlap sub-array pair for a ULA).

Requires a shift-invariant geometry; native 2-D az/el needs a 2-D variant (e.g. Unitary / 2-D ESPRIT).

*(Geometry note: Root-MUSIC and ESPRIT constrain the valid `array_geometry` — the non-uniform / holey virtual arrays from `VirtualArrayAssemblyOp` may need a ULA/subarray-selection or spatial-smoothing step first, and native 2-D az/el needs a 2-D-capable variant. They do not change the port shapes above.)*

---

### MIMO Marshaling Operators

#### MIMODeinterleaveOp

- **In:** `adc_cube` — complex64 / complex128 (IQ IF) **or** float32 / float64 (real IF), `(..., N_rx, M·L, samples_per_chirp)`. Slow-time interleaves the M TX per the TX schedule.
- **Out:** `adc_cube` — complex64 / complex128 / float32 / float64 (dtype preserved), `(..., M_tx, N_rx, L, samples_per_chirp)`. TX split into an explicit axis. Pure marshaling (strided reshape/gather).

**Parameters:**

- `tx_schedule` — TXSchedule. The TDM firing pattern (carries `num_tx`); determines how the interleaved `M·L` slow-time axis splits into `(M_tx, L)`. The split is not inferable from the input alone — the `M·L` axis carries no marker of the boundaries — so this is required; `chirps_per_tx` (L) is then `(M·L)/num_tx`. Static (a fixed waveform / hardware property). TDM only. Default: none (required). The first-wave TDM chain (the downstream per-TX phase compensation) assumes **uniform** per-TX spacing, so a non-uniform / non-round-robin schedule is rejected with an error.

#### MIMOCoherentCompensationOp

- **In:** `rd_maps` — complex64 / complex128. **TDM mode:** `(..., M_tx, N_rx, num_doppler, num_range_bins)` (TX axis already explicit from de-interleave). **DDM mode:** `(..., N_rx, num_doppler, num_range_bins)` (the M TX sit in distinct Doppler sub-bands). Rank/interpretation set by the `multiplexing_scheme` parameter.
- **Out:** `rd_maps` — complex64 / complex128, `(..., M_tx, N_rx, num_doppler_out, num_range_bins)`. Separated, phase-aligned `(tx, rx)` block. `num_doppler_out = num_doppler` (TDM) or `num_doppler_per_tx` (DDM band width).

**Parameters:**

- `multiplexing_scheme` — enum {TDM, DDM}. Selects the compensation path — TDM: per-Doppler-bin inter-TX phase correction; DDM: Doppler-band demux + per-chirp code removal + alignment — and how the `rd_maps` input rank is interpreted. Static (selects the kernel; instantiate separate ops to switch). Default: none (required).
- `waveform` **[live]** — FMCWWaveformDescriptor. Per-TX PRI and carrier frequency for the Doppler-dependent phase math (the inter-TX correction under TDM; band alignment under DDM). Default: none (required).
- `ddm_code` — DDMCodeDescriptor. The Doppler-band / phase-code layout to demux (carries `num_tx`). Static. Default: none (required for DDM; unused for TDM).

*(No array/element geometry is needed here — this is a phase/demux operation, not a spatial one; `element_positions` enter only downstream at VirtualArrayAssemblyOp. `num_tx` is not a separate parameter — TDM reads it from the input's `M_tx` axis, DDM from `ddm_code`; anything sized by `num_tx` (e.g. the TDM per-Doppler-bin correction table) is built lazily on the first tick and cached, as with FFT plans, so it needs no init-time `num_tx`. TDM assumes uniform per-TX spacing; a non-uniform schedule is rejected upstream (see MIMODeinterleaveOp). Doppler-ambiguity / ÷M velocity handling lives here.)*

#### VirtualArrayAssemblyOp

- **In:** `rd_maps` — complex64 / complex128, `(..., M_tx, N_rx, num_doppler, num_range_bins)`. Separated `(tx, rx)` block. (TX/RX element positions are parameters.)
- **Out:** `rd_maps` — complex64 / complex128, `(..., num_virtual, num_doppler, num_range_bins)`. `(tx, rx)` flattened to a single virtual channel axis (`c = tx·N_rx + rx`), overlapped virtual elements deduped, so `num_virtual ≤ M·N`.
- **Out:** `array_geometry` — ArrayGeometry on an output port (a descriptor exception, since the virtual geometry is computed at runtime): virtual `element_positions` (`p_virtual = p_tx + p_rx`) + wavelength, wired directly into the angle operator's `array_geometry` override port (same descriptor type).

**Parameters:**

- `tx_positions` — float, `(num_tx, 3)`. TX element positions (m); length must match the input `M_tx` axis. Static (fixed aperture). Default: none (required).
- `rx_positions` — float, `(num_rx, 3)`. RX element positions (m); length must match `N_rx`. Static. Default: none (required).
- `wavelength` **[live]** — float. Operating wavelength (m) for the emitted `array_geometry` (`= c / carrier_frequency`); `[live]` so it tracks a changing carrier. The virtual `element_positions` themselves are wavelength-independent. Default: none (required).
- `overlap_policy` — enum {dedup, average, keep_all}. How coincident virtual elements (`p_virtual = p_tx + p_rx` collisions) are handled: `dedup` keeps one representative (default), `average` combines their samples, `keep_all` retains duplicates. Sets `num_virtual`. Static. Default: dedup.

---

## Supporting data structures

Shared types the operators depend on; expanded as each operator is locked.

### ArrayGeometry

Describes the (virtual) antenna array for beamforming and DOA estimation. Shared across `PhaseShiftBeamformerOp` and the sparse DOA estimators to keep array conventions consistent. Passed as a parameter (config), not a port.

- `element_positions` — float, `(num_elements, 3)`. Element coordinates (meters). General form (ULA / URA / UCA / Conformal). Whatever the channel axis represents: physical RX positions (SIMO) or synthesized virtual positions (`p_tx ⊕ p_rx`, MIMO). `num_elements` must match the operator's `num_channels` axis.
- `wavelength` — float. Operating wavelength (m), `c / carrier_frequency`. Sets the steering-vector phase per element.
- `geometry_type` (optional) — enum {ULA, URA, UCA, Conformal}. Informational; enables optimized kernels (e.g. the spatial-FFT fast path).

### AngleConvention

Defines the azimuth/elevation (AER) parameterization at the API boundary — the `(az, el) ↔ unit-vector` mapping — for every operator that consumes a scan grid or emits `(az, el)`. Passed as a **standalone, required parameter (no default)**; *not* folded into `ArrayGeometry`. Axis references interpreted in the array-local Cartesian frame; angles in radians; applied only as a boundary relabeling (kernels run in the fixed canonical convention).

- `preset` — enum {radar_boresight, ros_rep103, iso8855_flu, aerospace_ned, physics_spherical, geodetic_enu, geodetic_ned, direction_cosine, custom}. `radar_boresight` = the internal canonical convention (identity/no-op). Other presets map to the cross-domain table rows. `direction_cosine` expresses the grid and `doa` in sine-space `(u, v)` — convention- and singularity-free; non-physical points (`u²+v² > 1`) are masked (`−inf` spectrum, NaN `doa`). `custom` uses the axis fields below. **Default: none (required).**
- `boresight_axis` — signed-axis enum {±x, ±y, ±z}. Axis that `(az=0, el=0)` points along. Used only when `preset = custom`.
- `azimuth_positive_axis` — signed-axis enum. Axis increasing azimuth rotates toward (fixes azimuth plane, sense, handedness). Orthogonal to `boresight_axis`. `custom` only.
- `elevation_positive_axis` — signed-axis enum. Axis increasing elevation rotates toward (`+z` = up, `−z` = NED-style down). Orthogonal to both other axes. `custom` only.
- `elevation_reference` — enum {horizon, zenith}. Elevation up from the azimuth plane (radar/robotics/aviation) vs. polar angle from boresight/pole (physics). `zenith` carries `el ↔ π/2 − polar` (a genuine angle transform). Default: horizon (`custom` only).

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

Describes the Doppler-division multiplexing scheme for `MIMOCoherentCompensationOp` (DDM mode) — how the M TX are separated after the Doppler FFT. (The forward-looking CDM/BPM extension would carry its orthogonal code here in a code-correlation mode; out of scope for now.)

- `num_tx` — int. M, the number of TX multiplexed into distinct Doppler sub-bands.
- `phase_codes` (optional) — float or complex, `(num_tx, num_chirps)`. The per-chirp phase (or unit-magnitude code) applied to each TX; the general form, demuxed by conjugate correlation. Provide this **or** `doppler_offsets`.
- `doppler_offsets` (optional) — float, `(num_tx,)`. Per-TX Doppler sub-band offset (fraction of `[0, PRF)`, or bins); the compact form for standard linear-phase DDM, from which `phase_codes` are synthesized. Provide this **or** `phase_codes`.
- `guard_bands` — int. Guard bins reserved between adjacent TX Doppler bands to keep them separable. Default: 0.

Notes: the `[0, PRF)` Doppler span is shared across the M bands, so unambiguous velocity is reduced by M; band width `num_doppler_per_tx` follows the offset/guard layout. Doppler-ambiguity handling lives in `MIMOCoherentCompensationOp`.

### TXSchedule

Describes the TDM slow-time firing pattern for `MIMODeinterleaveOp` — how the interleaved slow-time axis (`M·L` chirps) splits into an explicit TX axis.

- `num_tx` — int. M, the number of time-multiplexed TX.
- `tx_order` (optional) — int, `(num_tx,)`. TX firing order within one repetition cycle; the schedule repeats it `L` times. Default: round-robin `[0, 1, …, M−1]` (the common case). De-interleave gathers per this order.
- `chirp_tx_index` (optional) — int, `(M·L,)`. Explicit per-chirp TX index for irregular / non-repeating schedules; overrides `tx_order` when present.

Notes: `chirps_per_tx` (L) is inferred as `(M·L) / M` from the input's slow-time length; the per-TX effective PRI is `M ×` the raw inter-chirp interval (carried in `FMCWWaveformDescriptor.chirp_repetition_interval`).
