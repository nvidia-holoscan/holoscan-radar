# Holoscan Radar Module — Testing & Simulation Requirements

## Purpose

Defines what the FMCW operators must be tested against: (1) the **minimal simulator** — an analytic point-target signal model that produces raw ADC cubes with known ground truth — and (2) the **high-level, I/O-level tests** each operator must pass. This document stays at the interface level: it specifies *what property* each test asserts, not the fixtures, tolerances, or framework. Lower-level unit tests (kernel internals, numerical tolerances, boundary indexing) are deferred.

## Test tiers

- **Unit** — one operator, minimal synthetic input with a closed-form answer (a tone, a steering vector, a DC-across-chirps sequence). Most operators are fully checkable here without the simulator.
- **Integration** — a chain (or full modality) fed by the simulator; assert that targets in produce the correct detections / velocities / angles out, within resolution.
- **Statistical** — Monte-Carlo properties (CFAR false-alarm rate, DOA error vs. the Cramér-Rao bound). Called out per operator where relevant.

This document lists the **high-level checks** across these tiers; exact fixtures and tolerances are a later step.

---

## Minimal simulator (analytic point-target signal model)

The simulator is a **signal model, not a physics/EM solver**: it synthesizes the raw ADC cube by summing each point target's closed-form contribution over (tx, rx, chirp, fast-time sample), then adds noise. Cheap, deterministic, and its own correctness is easy to reason about.

### Inputs it must accept

- **Scene** — a list of point targets, each with a **Cartesian position** `(x, y, z)` (m) and **Cartesian velocity** `(vx, vy, vz)` (m/s) in the array-local frame, plus a complex amplitude (or RCS + a range law). The simulator derives range `R = ‖position‖`, radial velocity `v_r` (the velocity vector projected onto the line of sight), and look direction `û = position / ‖position‖`. Stationary clutter is just zero-velocity targets.
- **Array geometry** — TX element positions and RX element positions (meters), in the array-local frame that the operators use.
- **Waveform** — the fields of `FMCWWaveformDescriptor` (carrier frequency, bandwidth/slope, chirp duration, sample rate, PRI), plus `num_chirps` and `samples_per_chirp`.
- **TX multiplexing** — SISO / SIMO / MIMO-TDM (a `TXSchedule`) / MIMO-DDM (a `DDMCodeDescriptor`).
- **Noise** — additive complex Gaussian, configurable by SNR or noise power, with a seed.
- **Output dtype** — complex IF (analytic) or real IF, to exercise both front-end paths.

### Outputs it must produce

- The **raw ADC cube** in the exact layout the front-end expects for the chosen modality — interleaved slow-time for TDM, per-chirp coded for DDM, `(…, N_rx, num_chirps, samples)` etc.
- **Ground truth** alongside it: each target's derived **physical** quantities — range (m), radial velocity (m/s), look direction (Cartesian unit vector / direction cosines) — plus the expected virtual-array geometry for MIMO.
  - **Not bin indices:** those depend on the processing configuration (FFT lengths, clutter-canceller chirp loss, `center_zero_doppler`, real-vs-complex range axis), which the simulator does not know.
  - Tests therefore assert in the **physical domain** — map an operator's output bin back to physical units via its own `range_bins` / `velocity_bins` outputs and compare to ground truth within resolution (this also validates those axis outputs).
  - A shared "forward bin model" helper, parameterized by the specific operator config, is available where an exact-bin assertion is wanted.
  - Comparing an estimator's `(az, el)` to the ground-truth direction goes through that estimator's `AngleConvention`, or the estimator is run in the canonical / `direction_cosine` convention for a direct compare.

### The signal model (per target, summed)

With `R`, `v_r`, and `û` derived from each target's Cartesian position and velocity, the contribution for each (tx `m`, rx `n`, chirp `k`, sample `s`) is the product of:

- a **fast-time beat phase** ∝ `R` (→ beat frequency via the sweep slope and sample rate),
- a **slow-time Doppler phase** ∝ `v_r` (advances by `2π·f_d·PRI` per chirp, evaluated at that TX's firing time within the cycle),
- a **spatial phase** `∝ (p_tx,m + p_rx,n)·û / λ` — the two-way array phase, with `û` the target's Cartesian look direction (no angle naming — it dots directly with element positions; this is what makes the virtual-array and angle operators testable),
- the **TX modulation** — TDM: placed in its time slot; DDM: the per-chirp code phase,
- the target's **complex amplitude** (× an optional range law).

Sum over targets, add noise.

### Scene frame and angle conventions

The scene is specified in the **array-local Cartesian frame**, so the simulator is **convention-free** — like the operator kernels, it only ever dots direction vectors with element positions and never names an angle. It therefore takes **no `AngleConvention`**. Supplying targets by `(az, el)` would reintroduce the naming ambiguity; converting `(az, el)` + an `AngleConvention` into a Cartesian direction is an **upstream helper's** job, and the resulting Cartesian direction is what feeds the simulator. This keeps the sim aligned with the module principle that the internal math is convention-free and `AngleConvention` is only a boundary relabeling — which also means the simulator can validate the angle operators' convention handling without itself depending on it.

### Modeling assumptions (state these explicitly)

- **Stop-and-hop.** The target is frozen *within* a chirp (fast-time) and advances in phase *between* chirps (slow-time). This models motion-between-chirps at the Doppler-phase level and **omits intra-CPI range migration** (range-Doppler coupling) — matching the operators' scope (range-Doppler coupling is out of the first wave). Valid while range·velocity products are modest; breaks for very fast targets × long CPIs × wide bandwidth.
- **Ego motion frozen over a CPI.** The platform sits at one position/orientation for the CPI (ranges and angles constant across chirps). Ego motion is **not** dropped from velocity — it folds into each target's *relative* velocity vector (target world velocity minus platform velocity), from which the radial component is derived; otherwise-static clutter then carries a nonzero ego-Doppler. Cross-frame ego motion and world-frame placement are the caller's responsibility (upstream frame transforms), consistent with the coordinate-conventions stance.
- **Far-field plane wave**, **monostatic**, **single-slope sweeps**, **ideal hardware**.

### Explicitly out of scope (first wave)

EM propagation / scattering; multipath; antenna element patterns and mutual coupling; hardware impairments (phase noise, IQ imbalance, ADC quantization); intra-CPI range migration; SAR aperture synthesis; near-field; bistatic geometry. These mirror the operator-lockdown exclusions and can be added later as optional layers.

### Cross-cutting simulator requirements

- **Aliasing modeled.** Targets beyond the unambiguous range / velocity must fold correctly, so wrap-around behavior (and the ÷M MIMO velocity reduction) can be tested.
- **Determinism.** A fixed seed reproduces the cube bit-for-bit.
- **Guard against tautology.** Derive the signal model independently of the operator implementations, and cross-check a few canonical cases against an independent reference (a textbook closed form or an established external radar model). Tests assert against **analytic ground truth** (target at 12.3 m / 4.1 m/s / +7°), not merely that the generator round-trips through the pipeline.
- **No kinematic driver (yet).** The simulator is **per-frame**: velocity is a frame parameter (it lives in the slow-time phase within one CPI), so a static generator exercises range, Doppler, angle, and all MIMO paths. A time-stepping driver (targets evolving across frames) is deferred until a frame-spanning consumer exists (a tracker, multi-frame integration, or clutter-map CFAR — all out of scope now). Inter-frame coverage is instead obtained by parameter-sweeping the static generator across many independent scenes. The generator's per-frame API should be shaped so a driver is a thin future add.

---

## Per-operator high-level tests

Each operator lists the I/O-level properties to validate. Unless noted, all also carry the module-wide checks in *Cross-cutting tests* (precision, memory space, rank-agnostic batching, determinism).

### Signal processing

#### FMCWRangeProcessorOp

- Single target at known range → the `range_profiles` peak maps (via `range_bins`) to the target's range within one bin; `range_bins` itself matches the analytic range axis (slope, sample rate, propagation speed) for the configured `nfft` / dtype.
- Both front-end paths: complex IF → full FFT bins; real IF → RFFT non-negative-beat bins.
- `apply_dechirp` on synthetic pre-mixer data reproduces the hardware-dechirped result.
- Windowing lowers peak sidelobe level vs. rectangular; coherent gain preserved (peak magnitude as expected under the chosen `fft_norm`).
- `nfft` oversampling adds bins and interpolates the peak location — no added resolution or extent.
- Leading axes (rx, or tx×rx) preserved and processed independently.

#### ClutterFilterOp

- `none` preserves shape and values exactly through the operator-owned output.
- Zero-Doppler (DC-across-chirps) input → `mean_subtraction` nulls it (≈ 0 at those cells); a moving target is preserved.
- `single_delay` / `double_delay`: output chirp count drops by 1 / 2; response is the binomial high-pass (DC notch).
- Leading axes preserved.

#### FMCWDopplerProcessorOp

- Target at known velocity → the `rd_maps` peak maps (via `velocity_bins`) to the target's radial velocity within one bin; `velocity_bins` matches `v = −f_d·λ/2`, accounting for `center_zero_doppler` and any clutter chirp loss.
- `center_zero_doppler`: zero-Doppler centered (true) vs. DC at bin 0 (false).
- Windowing / `nfft` behave as in the range operator.

#### FMCWRangeDopplerProcessorOp (fused)

- **Parity:** identical `rd_maps` (within tolerance) to the split `FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp` chain for matching parameters.
- Inline `clutter_filter` reduces `num_doppler` consistently with the standalone op; `range_bins` / `velocity_bins` match the split.

### Detection & clustering

#### CFARDetectorOp

- Target injected above threshold over known noise → detection at the target's cell (verified in physical units via the RD axes); `detection_score > 1` there and `< 1` elsewhere; `detection_mask == (detection_score > 1)`.
- **Statistical:** over noise-only trials, the realized false-alarm rate ≈ configured `pfa` for each variant {CA, OS, GO, SO}.
- `nci_axes` integration raises detectability and the realized Pfa still matches (Gamma(K) threshold with the correct `K`).
- Real-power input (float dtype) thresholds directly and yields the same detections as the complex `|·|²` path.
- `detection_idx` lists exactly the detected cells, sized to the count; a set `max_detections` caps and warns on overflow.
- `peak_mode` covers `none`, per-plane `range_doppler`, and full rank-2/3/4
  `all` NMS; detected-neighbor comparisons use raw CFAR power, do not wrap at
  axis boundaries, and break equal-power ties by full row-major index.
- Peak selection changes only `detection_mask` / `detection_idx`; dense
  `detection_score`, `noise_estimate`, and `cfar_power` remain unchanged.
- Selecting OS with the `-1` unset `os_rank` sentinel rejects with a targeted
  required-parameter error; explicit rank 0 remains valid.
- Sparse external α LUTs for GO/SO and OS `K > 1` may contain zero only at
  unreachable indices; zero at any reachable index rejects.
- Every `[live]` family is exercised through Graph with an Exact mid-stream
  match and a following miss that reverts to Params; malformed cell-window and
  host-Tensor payloads reject.
- Every output is optional/latched; leaving either the four non-index outputs
  or `detection_idx` unwired does not fail active sibling outputs or later
  ticks.
- Extra leading rank beyond the ctor-admitted named axes is rejected (no
  opaque module batch).

#### ClusterDBSCANOp

- Synthetic blobs in feature space → correct membership; noise points labeled `−1`.
- Per-axis `epsilon` correctly handles mixed-unit (anisotropic) features; `min_points` separates core vs. noise.
- A zero per-axis `epsilon` excludes that feature completely, including non-finite pass-through values; all-zero and negative encodings reject, and default `min_points` counts only active features.
- A non-core point neighboring two disconnected core components joins the
  component with the numerically smallest root.
- `labels` row-aligned with `points`; `sizes` counts per cluster; `max_clusters` caps.
- Every `[live]` family is exercised through Graph with an Exact match or
  skewed miss and a following miss that reverts to Params.
- `sizes` is optional/latched; leaving it unwired does not fail `labels` or
  disturb descending-size cluster numbering on later ticks.
- Rank-3 input is rejected; the contract admits no opaque leading batch.

#### FeatureAssemblyOp (bridge)

- `points` columns equal the configured physical lookups / sparse estimates /
  raw indices at each detection, in `features` order.
- Rows 1:1 with `detection_idx`.
- Selecting a feature whose source input is absent → error.
- `range` / `velocity` require their physical axes; raw coordinates are
  selected explicitly with `range_bin` / `doppler_bin`.
- Dense `az` / `el` perform angle-grid lookups; sparse
  `az_estimate` / `el_estimate` read `aoa_estimates`; raw preserved angle
  coordinates use `az_bin` / `el_bin` and require no grid inputs.
- Static `features` determines the compiled join: raw-only layouts need only
  `detection_idx`; selected source ports are required exact-aligned members;
  unselected gather ports may remain unwired.
- Extra leading axes on `detection_idx` are rejected (input rank is exactly 2; no module batch).

#### DetectionReductionOp (bridge)

- `peak` policy → `rep_idx` is the max-score member of each cluster; `first` / `medoid` select the expected member (`medoid` respects `feature_weights`).
- `weighted_mean` with empty `feature_weights` emits the ordinary feature mean; one-hot score weighting emits the analytic score-weighted mean. Its `rep_idx` is the real member nearest the same weighted centroid over all detection-index columns.
- `reduced_points` gathers the selected member for `first` / `peak` / `medoid`, is synthetic only for `weighted_mean`, and may be left unwired without suppressing `rep_idx`.
- `weighted_mean` rejects a negative or non-finite row weight and a non-positive per-cluster weight sum.
- Every `rep_idx` addresses a real detected cell (a subset of `detection_idx`); one per cluster; noise (`−1`) excluded.
- `points` is always a required exact-aligned input, including for `representative=first`; an unwired graph is rejected for every policy.
- A caller-instantiated compile-time `ScoreFn` is exercised through the Graph harness from a CUDA translation unit.
- Exact peak/medoid ties choose the lowest input row; opaque leading batch rank is rejected.

### Angle estimation

#### PhaseShiftBeamformerOp (dense)

- RD maps synthesized as a steering vector for a known (az, el) → `rda_cube` peak at the expected beam bin.
- `angle_convention`: default-constructed Params select `radar_boresight`; the same physical direction is reported with the expected (az, el) labels under different presets; `direction_cosine` masks the invisible region (`u²+v² > 1`).
- `array_taper` lowers spatial sidelobes; recognized URAs use a separable
  per-axis product and must suppress azimuth and elevation sidelobes
  independently, while custom coefficients remain per-element.
- `[live]` overrides (`array_geometry` / `az_angles` / `el_angles` / `array_taper`) change the output and reuse cached weights when unchanged.
- Optional `az_grid` / `el_grid` outputs equal the effective scan grids in template precision; live grid overrides reshape both the corresponding cube axis and grid output for one tick, then a missing override reverts both to Params.
- Input rank is exactly 3; an opaque leading batch axis is rejected, and Doppler/range or scan-grid extents above `MaxTensorExtents` fail before output allocation.
- Distinct `custom` angle frames and distinct floating-point grids produce distinct steering-cache identities; degenerate custom frames reject.

#### SnapshotGatherOp (bridge)

- `snapshot_window (0,0)` → `snapshot` equals the per-channel vector at each detected cell, shape `(num_detections, num_channels, 1)`.
- A non-trivial window gathers the correct neighborhood; `num_snapshots = (2·d+1)(2·r+1)`.
- Edge modes: `wrap` wraps the Doppler axis; `shift` slides inward keeping the count; `clamp` truncates and marks the missing slots in `is_valid`.
- Channel axis preserved (gathers per-channel, not the NCI-integrated map).
- Invalid preserved-batch and Doppler/range coordinates fail predictably; `shift` rejects a requested window longer than its axis.
- Leaving optional/always `is_valid` unwired does not suppress required `snapshot`.
- A matched `[live] snapshot_window` applies for one tick and a subsequent miss reverts to Params.

#### SubarraySelectionOp

- Auto-selection returns the longest filled ULA run or largest filled URA rectangle, including the larger-azimuth-aperture tie-break.
- A filled but non-lattice-ordered input is still gathered into lattice order; equal input/output channel counts do not imply pass-through.
- Explicit indices are range/uniqueness checked, reordered into lattice order, and rejected when they leave a hole.
- A 3-D aperture or selected size below `min_subarray_size` fails predictably.
- Leaving optional/latched `array_geometry` unwired does not suppress required `snapshot_out`; a wired geometry emits on the same tick.
- `snapshot_out` and `WireArrayGeometry` directly drive an angle estimator; the Exact join preserves their shared capture time and the reduced geometry yields the expected steering result.

#### Sparse angle estimators (shared slot)

Common check (all five): default-constructed Params select `radar_boresight`; `snapshot = a(θ)` (+ noise) for a known DOA → `aoa_estimates ≈ θ` within grid/interpolation or the CRLB under both the default and at least one non-default convention; an `is_valid` mask correctly excludes masked snapshots.

- **BeamscanEstimatorOp** — single-source recovery; accepts 2-D or 3-D-with-`num_snapshots=1`; errors on 3-D with `num_snapshots ≠ 1`.
- **MVDREstimatorOp** — multi-snapshot covariance nulls an interferer better than Beamscan; `diagonal_loading` stabilizes the low-snapshot / ill-conditioned case.
- **MUSICEstimatorOp** — resolves two closely-spaced sources that Beamscan cannot, given the source count (fixed or EFT `auto`).
- **RootMUSICEstimatorOp / ESPRITEstimatorOp** — gridless DOA ≈ truth, with **no** `angle_spectrum` output. On a **1-D ULA** / shift-invariant geometry they resolve azimuth; on a **filled URA** (`subarray_spec: auto`) they resolve native (az, el) — assert single- and two-source recovery with sources distinct in **both** azimuth and elevation so mispairing is detectable (Root-MUSIC per-axis rooting + pairing; ESPRIT dual-invariance + joint-diagonalization pairing). On a non-uniform virtual array they either use the subarray/smoothing step or fail predictably.
- Multi-source estimators fill `doa_multi` / `source_count` up to `max_sources`.

### MIMO marshaling

#### MIMODeinterleaveOp

- An interleaved cube with TX-tagged content → output correctly split into `(M_tx, N_rx, L, samples)` for a round-robin `tx_schedule` and for a custom order.
- A non-uniform schedule is rejected with an error (first-wave TDM assumes uniform spacing).
- Dtype preserved.
- Extra leading axes are rejected (input rank is exactly 3; no module batch).

#### TDMDenseCompensationOp

- **TDM:** a moving target → after compensation the inter-TX motion phase is removed, so the assembled virtual array yields an **unbiased** DOA (whereas skipping compensation biases it) — this is the key regression.
- `num_tx` is derived from the input axis; round-robin and custom uniform schedules index phase by firing slot.

#### DDMDenseDemultiplexOp

- A multi-TX coded cube → demux recovers each TX into the `M_tx` axis at its band with linear-code initial phase removed.
- Offset-only, linear phase-code, guard-band, nonuniform integer-offset permutation, and nonlinear full-axis correlation cases retain the existing dense behavior.
- Fractional-bin linear offsets fail predictably; `num_tx` is derived from `ddm_code`.

#### TDMSparseCompensationOp

- For identical detections and configuration, output snapshots and geometry match the corresponding cells from `TDMDenseCompensationOp → VirtualArrayAssemblyOp → SnapshotGatherOp` under every overlap policy.
- A moving target's gathered virtual snapshot yields an unbiased DOA; omitting the sparse correction biases it by the same amount as omitting dense TDM compensation.
- Each cell in a nonzero `snapshot_window` uses its own resolved Doppler bin for phase correction. Edge modes, `is_valid`, exact-sized zero/nonzero detection outputs, and `[live] snapshot_window` override/revert match `SnapshotGatherOp`.
- Round-robin and custom uniform `tx_schedule` mappings produce the same firing-slot correction as `TDMDenseCompensationOp`; malformed shapes, indices, extents, schedules, and live waveforms fail predictably.
- The emitted `WireArrayGeometry` matches the virtual snapshot ordering and overlap policy, derives wavelength from the same effective waveform used for correction, and directly drives an angle estimator under an `Exact{}` join.

#### DDMPowerFoldOp

- Folded power matches an independent host sum over the dense demux table for every TX/RX look, including nonuniform integer offsets and guard bands.
- Output shape is exactly `(D/M-G, R)`; bins discarded by `G` are absent and the documented blind fraction is `G/(D/M)`.
- Fractional-bin offsets, nonlinear codes, non-divisible Doppler extents, malformed shapes, and invalid guard widths fail predictably.
- Float/double agree; output is device-resident real power.
- White-noise Monte Carlo through `CFARDetectorOp(preintegrated_looks=M·N_rx)` reaches the requested Pfa. A correlated-noise characterization demonstrates that nominal `M·N_rx` is not exact and a lower empirically calibrated effective look count improves agreement.

#### DDMSparseDemultiplexOp

- For identical detections and linear codes, snapshots and geometry match `DDMDenseDemultiplexOp → VirtualArrayAssemblyOp → SnapshotGatherOp` under every overlap, window, and edge policy.
- Offset-only and phase-code descriptors agree; nonzero initial code phase is removed consistently by dense and sparse demux.
- A principal-fold target (`k=0`) recovers with the missing-input `SafeDefault{0}`. An aliased target with unresolved zero exhibits the documented cyclic TX permutation, while the same target with supplied nonzero `fold_idx` recovers the reference snapshot and unbiased DOA.
- `fold_idx` values outside `[0,M)` or with a row count different from `detection_idx` fail predictably.
- Nonlinear and fractional-bin codes reject. Exact-sized zero/nonzero outputs, `is_valid`, `[live]` snapshot-window/wavelength override/revert, and `WireArrayGeometry` coherence match `TDMSparseCompensationOp`.

#### VirtualArrayAssemblyOp

- `(tx, rx)` block → virtual channel axis with `p_virtual = p_tx + p_rx`; emitted `array_geometry.element_positions` match the analytic virtual layout.
- `overlap_policy`: `dedup` collapses coincident positions (`num_virtual` = unique count); `keep_all` retains `M·N`; `average` combines coincident samples.
- `num_virtual ≤ M·N`.
- Extra leading axes are rejected (input rank is exactly 4; no module batch).
- Emitted `WireArrayGeometry` directly drives an angle estimator's Exact live input and yields the expected steering result without an adapter.

---

## Cross-cutting tests

- **End-to-end per modality.** SISO / SIMO / MIMO-TDM / MIMO-DDM: a scene with known targets → detections at the correct range / velocity (and angle for the angle-bearing modalities), within resolution. The MIMO end-to-end runs are the primary validation of the marshaling + angle chain together.
- **Sparse-slot interchangeability.** With the wiring fixed, swapping the estimator in `⟨sparse angle estimator⟩` (Beamscan ↔ MVDR ↔ MUSIC ↔ …) requires no graph change and each recovers the DOA — validating the shared contract.
- **`[live]` override behavior.** An Exact-matched override mid-stream changes the output accordingly; a following tick with no match reverts to the Params value, since an override applies to one tick only. Warn only when a genuine rebuild is triggered (FFT plan, window coefficients, CFAR window kernel); previously-seen configurations are reused from cache.
- **Precision.** Single- and double-precision instantiations agree within their respective tolerances.
- **Memory spaces.** Operators accept tensors in each CUDA-accessible memory space (device, managed, pinned host) and reject pageable-host input with a clear error; outputs are always device memory. `test_tensor` directly covers the shared `require_cuda_accessible()` rejection and diagnostic. HSDK 5 Graph sources currently materialize data through device-bound output loans, so do not duplicate an unreachable pageable-host case as a skipped test in every operator suite; see `HSDK5_PORT.md` §3.
- **Determinism.** A fixed seed reproduces identical outputs across runs.

---

## CI notes

- Unit and integration tests should be fast and deterministic (seeded), suitable for per-commit CI; the statistical tests (Pfa, DOA-vs-CRLB) are heavier Monte-Carlo runs better suited to a nightly job.
- The simulator is a first-class test dependency and should be versioned and tested itself (its analytic outputs cross-checked against an independent reference), since every integration test relies on it.
