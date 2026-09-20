# Open items

This is the single backlog for deferred Holoscan Radar work. Each section
names the deleted source file whose live contents are preserved below.
Items explicitly marked fixed, completed, or superseded in those source
files are omitted.

## Application backlog

_Original source: `applications/OPEN_ITEMS.md`_

### 1. GUI visual refinement

**Status:** open — low-priority polish; agreed to defer until core signal
chain is stable.

The HSDK 5 dashboards across the standalone SISO, SIMO, and MIMO applications are
functional but unpolished. Candidate improvements:

- Layout and panel sizing (consistent margins, panel proportions).
- Color-map choices for RD maps and cluster overlays.
- Label / legend clarity (axis tick density, units).
- Window title and headless vs. GUI mode indication.

No specific design has been chosen; track against the `radar_app` viz helpers
in `applications/common/radar_app/` so improvements land once and propagate to
all apps.

### 2. GUI dynamic range should be more static

**Status:** open — affects all Holoviz RD-map panels.

The RD-map color scale currently auto-ranges per frame (or per a short rolling
window), causing the display to "breathe" as targets and noise floor fluctuate.
A fixed or slowly-adapting dynamic range would give a more stable visual
reference.

Candidate approaches (not chosen):

- Fixed `[vmin, vmax]` parameters in the YAML config, passed through to the
  Holoviz image widget.
- Percentile-clipped running statistics (e.g. 1st–99th percentile over a
  configurable history window) so the scale adapts slowly rather than
  frame-by-frame.
- Operator-level `DynamicRangeOp` that normalizes the RD map before
  visualization, decoupling the scale policy from the Holoviz widget.

Whichever approach, the dynamic-range policy should be configurable via YAML
and shared across apps via `radar_app` so it does not need to be re-implemented
per demo.

### 3. HDF5 ADC replay source

**Status:** open — shared application input option; simulator sources remain
the default.

Each application should be able to select recorded HDF5 ADC data instead of
the analytic simulator without changing its downstream processing graph. Add
a reusable source under `applications/common/radar_app/` that reads framed ADC
tensors, validates the application-specific shape and complex dtype, uploads
each frame into a device Tensor pool, and emits the same `adc_cube` contract
and cadence/timestamps as the corresponding simulator source.

The application-owned YAML boundary should select `simulator` or `hdf5` and
carry only replay concerns such as file path, dataset path, end-of-file policy,
and optional frame range. Operator parameter blocks remain unchanged. SISO, SIMO, MIMO-TDM, and MIMO-DDM must each validate the recorded tensor against
their existing named ADC axes before the graph starts; file metadata must not
silently override waveform, array, schedule, or DDM-code configuration.

Before implementation, define and document:

- the HDF5 dataset layout and complex-number encoding;
- fixed-shape versus per-frame datasets and supported precisions;
- stop, loop, and bounded-frame behavior at end of file;
- timestamp/cadence behavior needed by Exact side inputs;
- HDF5 build, container, and package dependencies;
- focused reader tests with small deterministic fixtures plus one smoke per
  application family.

This remains a GPU application path: host file I/O is expected, but emitted
ADC tensors are CUDA-accessible and there is no CPU signal-processing fallback.

### Interactive multi-profile dashboard selection

_Original source: `docs/planning/STATUS.md`_

The fixed `app.visualization.profile` path prevents rotating-axis flicker.
Runtime arrow-key or numeric-key switching requires public keyboard callbacks
from the HSDK 5 Holoviz operator. Do not add X11 interception or a radar-owned
presentation loop as a substitute.

## Operator-library backlog

_Original source: `operators/OPEN_ITEMS.md`_

### 1. Add automatic C++ formatting checks

**Status:** open; the current pre-commit configuration runs `cpplint` but not
`clang-format`.

Add a checked-in `.clang-format` policy and a pinned `clang-format`
pre-commit hook, then run the same formatter check in CI. This would make
equivalent formatting choices deterministic across operator sources instead
of relying on manual consistency.

Before landing, choose and document the formatter version and verify the
initial repository-wide formatting diff is acceptable. Avoid mixing the
bootstrap formatting sweep with behavioral changes.

### 2. Support staggered and non-uniform TDM timing

**Status:** deferred 2026-07-22; uniform repeating TDM is the v1 scope.

`TXSchedule`, `MIMODeinterleaveOp`, `TDMDenseCompensationOp`, and
`TDMSparseCompensationOp` currently require a repeating cycle in which each TX
appears once per cycle with uniform per-TX spacing. A permutation of the TX
order is supported; varying dwell, count, or inter-firing interval is rejected.

Supporting staggered TDM requires a per-TX timing representation, generalized
deinterleave indexing, Doppler/phase correction based on actual firing times,
simulator support, and dense/sparse parity tests. Do not reinterpret
`FMCWWaveformDescriptor.chirp_repetition_interval`: it remains the effective
per-TX slow-time sample period for the current uniform workflow. The future
contract must define how timing is represented and kept coherent across both
TDM operators before relaxing `TXSchedule` validation.

### 3. Support M+1 empty-band DDM ambiguity resolution

**Status:** research item deferred 2026-09-09; current DDM uses M occupied
bands in an M-band packed layout.

Packed M-band DDM provides no empty-band observation from which one frame can
identify the cyclic Doppler ambiguity. `DDMSparseDemultiplexOp.fold_idx`
therefore remains an external escape hatch: missing input uses
`SafeDefault{0}`, which is correct only in the principal reduced velocity
interval. This is a DDM-wide waveform limitation, not a sparse-demultiplex
implementation gap; dense demultiplexing does not inherently resolve it.

Investigate an M+1 layout in which M TX comb lines occupy M of M+1 Doppler
bands and the empty band's location supplies ambiguity information. Before
locking a contract, establish behavior for multiple targets and comb
collisions in one range bin, guard-band loss, noise/clutter, and band-boundary
crossings. Compare the usable velocity and detection performance against
packed M-band DDM and external staggered-CRI or code-rotation observations.

Implementation would affect `DDMCodeDescriptor`, dense and sparse demultiplex
tables, `DDMPowerFoldOp` shape/index semantics, simulators, applications, and
cross-path tests. Do not introduce a generic `DDMFoldResolverOp` until a
specific observable and resolution algorithm are selected. Keep `fold_idx`
available for applications that resolve the ambiguity externally.

### 4. Add code-aware DDM clutter suppression

**Status:** corrected and deferred 2026-09-16; dense and sparse DDM v1
explicitly do not support clutter filtering.

The existing `ClutterFilterOp` is not valid after `DDMDenseDemultiplexOp`:
demultiplexed outputs are already Doppler-domain maps, so mean subtraction acts
across Doppler bins and creates a range-aligned pedestal. Applying that filter
directly to coded pre-demux chirps is not equivalent either, while power folding
concentrates zero-Doppler clutter into the folded zero bin. Both DDM workflows
therefore currently require scenes where omitting clutter suppression is
acceptable.

Design a code-aware pre-demux or folded-domain suppression method and validate
its Pfa/Pd impact before adding it to either workflow, or define a dedicated
post-demux operator that notches physical zero velocity in the RD domain. Any
new operator or parameter is a contract change.

### 5. Extend sparse DDM beyond linear integer-bin codes

**Status:** deferred 2026-09-09; dense DDM retains broader code support.

`DDMPowerFoldOp` and `DDMSparseDemultiplexOp` currently require a linear
M-band code whose offsets map exactly to Doppler bins. Fractional-bin offsets
and nonlinear codes reject. `DDMDenseDemultiplexOp` continues to support
nonlinear full-axis correlation.

Fractional offsets require an interpolation/leakage model rather than the
current exact gather table. Nonlinear codes require detection and sparse
recovery definitions based on correlation, not band folding. Treat these as
separate algorithm and contract extensions; retain dense-versus-sparse
snapshot parity as the acceptance oracle wherever both paths are defined.

### 6. Support frame-to-frame DDM code rotation

**Status:** deferred 2026-09-09; `ddm_code` is static in all current DDM
operators.

Code rotation across frames can mitigate comb collisions between targets in
the same range bin and can contribute an external observation for fold
resolution. One operator instance cannot currently consume a new
`DDMCodeDescriptor` each tick; callers must use separate statically configured
instances.

A future live-code contract must keep `DDMPowerFoldOp`,
`DDMDenseDemultiplexOp`, and `DDMSparseDemultiplexOp` temporally coherent,
invalidate/reuse mapping and phase caches safely, and warn once per unseen
configuration. It must also define how detections and `fold_idx` stay paired
with the code that produced them.

### 7. Add a sparse DDM application workflow

**Status:** deferred 2026-09-09; intentionally excluded from the first
operator implementation.

Add an `applications/fmcw/mimo_ddm` sparse-DDM graph/config path that wires
`DDMPowerFoldOp → CFARDetectorOp → DDMSparseDemultiplexOp → angle estimator`,
including `preintegrated_looks = M_tx * N_rx`, geometry and validity sidebands,
and performance/reporting integration. The example must label
`SafeDefault{0}` fold handling as principal-interval-only; it must not present
unresolved full-velocity operation as generally correct. Add deterministic
smoke coverage and compare its recovered snapshots against the dense DDM
pipeline for supported linear codes.

## Angle-estimation backlog

_Original source: `operators/angle_estimation/OPEN_ITEMS.md`_

### EFT source-count threshold fidelity

MUSIC, Root-MUSIC, and ESPRIT share the EFT source-count helper. Its
closed-form noise-eigenvalue profile uses a first-order Gaussian threshold
rather than Monte-Carlo-calibrated η_P. Consequently, `source_count_pfa` is a
target rather than a guaranteed false-alarm rate; rank-deficient inputs fall
back to one source.

Keep this as shared algorithm research rather than resolving it during an
operator cleanup. Any replacement needs a validated statistical study and
documented accuracy expectations across supported channel, snapshot, SNR, and
precision ranges.

### MUSIC denominator floor

MUSIC scanning and Root-MUSIC candidate pairing use
`kMusicDenomFloor<T> = 1e-6` before evaluating `1 / (aᴴP_n a)`. The absolute
floor prevents division by zero and preserves a genuine near-zero MUSIC peak,
but it is a first-cut constant shared by float and double precision. It caps
the pseudospectrum at `1e6` and has not been calibrated across channel counts,
conditioning, source separation, or precision.

Do not tune the constant from a single scenario. Evaluate absolute versus
scale- and precision-aware floors against the analytic simulator, including
near-coincident sources, noisy and rank-deficient covariance matrices, varied
aperture sizes, and float/double agreement. Any change must preserve invisible
direction-cosine blanking, deterministic peak ordering, and both MUSIC and
Root-MUSIC behavior.

### Subspace model-order performance sweeps

MUSIC and Root-MUSIC now measure fixed source counts 1, 2, and
`min(4, num_channels - 1)` independently from their live-override matrices.
ESPRIT also has model-order-dependent work but does not yet carry an equivalent
sweep. Add estimator-specific fixed-order scenarios when its performance
harness is next revised; do not multiply the full live-override matrix by every
model order.

## ESPRIT backlog

_Original source: `operators/angle_estimation/esprit_estimator_op/OPEN_ITEMS.md`_

### Estimator variants

Version 1 implements LS-ESPRIT. TLS-ESPRIT and Unitary ESPRIT remain accuracy
and performance follow-ups. The filled-URA 2-D path is landed; explicit
two-pair non-ULA subarray specifications remain deferred.

## MUSIC backlog

_Original source: `operators/angle_estimation/music_estimator_op/OPEN_ITEMS.md`_

### Cross-item steering-manifold reuse

After the noise-subspace projector was conditionally staged in shared memory,
profiling identified per-beam steering-manifold reads as the remaining
`music_reduce_kernel` L2-bandwidth pressure. The manifold is immutable for a
cached geometry and scan grid and is reused by every item, but the current
kernel independently reads it while evaluating each item.

Further optimization is deferred because sharing manifold tiles across items
changes the kernel mapping and may trade lower L2 traffic for additional
shared-memory use, synchronization, register pressure, or reduced occupancy.
Before changing the kernel, re-profile current HSDK 5 builds on representative
small and large scenarios. Compare candidate item/beam tilings against the
existing occupancy-gated projector staging, and retain the change only if it
improves steady-state latency without regressing smaller grids or either
precision.

## MVDR backlog

_Original source: `operators/angle_estimation/mvdr_estimator_op/OPEN_ITEMS.md`_

### Automatic diagonal-loading calibration

Auto diagonal loading uses
`1e-2 * trace(covariance) / num_channels`; the factor remains a first cut.

### Capon reduction performance

Capon reduction remains latency-bound; warp-cooperative or GEMM paths are
deferred.

## Root-MUSIC backlog

_Original source: `operators/angle_estimation/root_music_estimator_op/OPEN_ITEMS.md`_

### Batched polynomial root solver

Serial per-item `cusolverDnXgeev` dominates latency. A batched or
device-resident replacement remains postponed by team decision.

## Snapshot-gather backlog

_Original source: `operators/angle_estimation/snapshot_gather_op/OPEN_ITEMS.md`_

### Dense-CFAR index projection for sparse AoA refinement

`CFARDetectorOp` on an RDA cube emits rows shaped
`(az_bin, el_bin, doppler_bin, range_bin)`. Gathering a refinement snapshot
from the original SIMO channel cube `(num_channels, num_doppler, num_range)`
should use only the trailing Doppler/range coordinates. The leading angle-bin
columns are candidate annotations, not axes of that input cube.

`SnapshotGatherOp` currently interprets every column before Doppler/range as a
preserved-batch coordinate and requires one matching leading `rd_maps` axis
per column. It therefore rejects this otherwise valid dense-detect followed by
sparse-AoA refinement path.

Add an explicit static index-projection or batch-column mapping that lets the
operator distinguish columns which address real `rd_maps` leading axes from
annotation columns ignored during gathering. Keep the current strict
all-preserved-columns behavior as the default; do not infer or silently discard
columns from tensor rank alone. Output rows must remain aligned with the
original full `detection_idx` rows so coarse `az_bin`/`el_bin` values stay
available to reporting, feature assembly, and future local-grid refinement.

The implementation must cover:

- rank-3 SIMO `rd_maps` with dense four-column RDA detections;
- existing batched `rd_maps` behavior without projection;
- invalid, duplicate, and out-of-range column mappings;
- empty detections and all three snapshot edge modes;
- graph integration through a sparse AoA estimator using a shared refinement
  grid.

This is ordinary deferred feature work, not a team-blocked contract decision.

## CFAR backlog

_Original source: `operators/detection/cfar_detector_op/OPEN_ITEMS.md`_

### 1. Add a generic detection and training exclusion mask

**Status:** deferred 2026-09-16; not required by the initial
`DopplerNotchOp` contract.

Investigate a generic validity/exclusion input for cells that CFAR must not use
as CUTs or training samples. Potential producers and uses include a future
`DopplerNotchOp.notch_mask`, known interference or saturation excision,
calibration-invalid bins, sensor blind zones, and algorithm-owned guard bands.
The surface must remain mechanism-neutral rather than naming Doppler notches.

Before locking a contract, settle:

- accepted shapes and named-axis broadcasting, including rank-1 Doppler masks
  versus masks matching the post-NCI CFAR plane;
- same-tick temporal pairing and missing-mask behavior;
- whether masked cells are excluded as CUTs, training cells, or both;
- dynamic training-cell counts and Pfa-derived alpha selection for CA;
- independent side counts for GO/SO and rank semantics when OS training cells
  are removed;
- edge behavior, minimum surviving training count, NMS interaction, and output
  values for excluded CUTs;
- mask dtype and whether multiple exclusion producers require an explicit
  composition operator.

Characterize realized Pfa at mask boundaries and ordinary map edges before
implementation. Any new input is an additive contract change to
`OPERATOR_API.md`.

## Doppler-notch backlog

_Original source: `operators/fmcw/doppler_notch_op/OPEN_ITEMS.md`_

### 1. Determine whether the operator should emit a notch mask

**Status:** deferred from the initial operator contract on 2026-09-16. The
first pass emits only the filtered RD tensor.

A hard or tapered post-Doppler notch creates a deterministic low-power region.
CFAR training windows that cross its boundary may underestimate interference
and raise realized Pfa. Before adding a public `notch_mask` output, measure
hard-null and cosine-transition behavior at the notch interior and edges for
CA, GO, SO, and OS CFAR. Compare against the existing pre-Doppler
`ClutterFilterOp` methods so the decision is based on observed statistical
impact rather than assuming post-Doppler filtering is unique.

If a mask is justified, decide whether it marks only the full-rejection
stopband or also some/all of the transition, whether it is a rank-1 Doppler
mask or a broadcastable RD mask, and whether it is a causal sideband emitted
before `rd_maps_out`. Coordinate the contract with
`CFARDetectorOp`'s generic exclusion-mask investigation; do not add a
notch-specific CFAR port.

## Doppler-processor backlog

_Original source: `operators/fmcw/fmcw_doppler_processor_op/OPEN_ITEMS.md`_

### 1. GPU-native window construction

**Class:** Performance follow-up; no contract change.

**Current behavior:** On a window cache miss, `doppler_stage.hpp` generates
taps on the host (`generate_window`) and uploads them through
`upload_coefficients`, which synchronizes the compute stream so its host
staging buffer can be released safely.

**Question:** Should window taps be generated directly on the GPU? The GPU
path could avoid host-side generation, staging, and synchronization for
unseen window configurations.

**Acceptance:** Benchmark cold ticks and unseen configuration changes across
the Doppler performance size ladder before choosing a CUDA kernel, a MatX
expression, or the current host-build path. Preserve cache reuse, stream
ordering, and numerical parity with `generate_window`.

## Fused range-Doppler backlog

_Original source: `operators/fmcw/fmcw_range_doppler_processor_op/OPEN_ITEMS.md`_

### 1. Benchmark, then revisit whether the fused operator earns its keep

**Status:** open — **benchmark harness landed**; team keep/kill decision still
pending (do not encode a verdict in CI).

The fused operator now fuses at the only seam the FFT barriers allow
(cuFFT in MatX v0.9.4): the clutter expression composes into the Doppler
pre-FFT pass (`with_clutter_expr` → the `run_doppler_stage` expression
overload), saving one full-cube write+read, one kernel launch, and one
async alloc/free per tick versus the split chain — plus the graph-level
savings of one node (no inter-operator message passing or intermediate
Holoscan tensor). The FFTs still dominate the tick, so the computational
win is expected to be modest.

What it costs to keep: a third operator to maintain (union-of-parameters
API surface, override plumbing, its own test suite including the
`UlpParity*` gates), and the expression-overload machinery exists
primarily for its benefit.

#### Decision aid (run this; do not treat as an automatic gate)

**Primary tool:** `performance/cpp/perf_fmcw_split_vs_fused` (Python mirror:
`performance/python/perf_fmcw_split_vs_fused.py`).

Times two end-to-end Holoscan graphs with identical I/O:

| Variant | Graph |
| --- | --- |
| `split` | `adc → range → [clutter] → Doppler → sink` |
| `fused` | `adc → FMCWRangeDopplerProcessorOp → sink` |

Default matrix: `{noclt, msub} × {xs, md, lg, xl}` plus `md_ch4` noclt.
`--full` adds `sdel` and channel batches. Emits per-graph rows plus a
**comparison table** (`split_ms`, `fused_ms`, `delta_ms`, `pct_saved`,
`speedup`). Suggested artifact paths:

```bash
./build/performance/cpp/perf_fmcw_split_vs_fused \
  --output perf_results/perf_fmcw_split_vs_fused.json \
  --text   perf_results/perf_fmcw_split_vs_fused.txt
./build/performance/cpp/perf_fmcw_split_vs_fused --full \
  --output perf_results/perf_fmcw_split_vs_fused.full.json \
  --text   perf_results/perf_fmcw_split_vs_fused.full.txt
```

**Do not** decide from a naive sum of
`perf_fmcw_range_processor` + `perf_clutter_filter` +
`perf_fmcw_doppler_processor` medians — that omits inter-op Holoscan cost on
the split path (and was the gap this tool closes).

**Decision to make after reviewing the table** (and optional `nsys` of a representative split-versus-fused application graph):

- If the end-to-end gain is negligible at realistic workloads, consider
  retiring the operator (or demoting it to an example of stage
  composition) rather than carrying the API surface forward.
- If the gain is real, or once a MatX upgrade with `-DMATX_EN_MATHDX=ON`
  (cuFFTDx) lands — which would let the expression-fed FFTs fuse their
  elementwise pre-ops and widen the gap — keep it and paste the numbers
  into this section.

Related: the split-vs-fused parity contract and stage-sharing design are
enforced by `tests/cpp/fmcw/test_fmcw_range_doppler_processor.cpp`
(`UlpParity*`, agreement to a few ULP rather than bit-exact — the
`mean_subtraction` reduction's summation order is not guaranteed to match
between the two paths).

## Range-processor backlog

_Original source: `operators/fmcw/fmcw_range_processor_op/OPEN_ITEMS.md`_

### 1. GPU-native coefficient construction

**Class:** Performance follow-up; no contract change.

**Current behavior:** On a premultiplier or window cache miss,
`range_stage.hpp` generates synthesized dechirp references, window taps, and
the combined premultiplier on the host, then uploads the coefficient tensor to
the GPU. The upload path synchronizes the compute stream so its host staging
buffer can be released safely.

**Question:** Should synthesized references, window taps, and the combined
premultiplier be generated directly on the GPU? Caller-provided
`reference_samples` would still require an upload, but the GPU path could
avoid host-side generation, staging, and synchronization for synthesized
configurations.

**Acceptance:** Benchmark cold ticks and unseen configuration changes across
the range performance size ladder before choosing a CUDA kernel, a MatX
expression, or the current host-build path. Preserve cache reuse, stream
ordering, and numerical parity with `radar_common/sim/`.

## Performance backlog

_Original source: `performance/OPEN_ITEMS.md`_

### 1. C++ / Python scenario-matrix sync

**Open question.** How should Python `perf_*` scripts stay aligned with the
C++ matrices in `performance/cpp/perf_*.cpp`?

Candidate approaches (not chosen):

- Manual mirror (current interim for all ops)
- Shared declarative matrix (JSON/YAML) loaded by both drivers
- Codegen from a single source into C++ and Python

There is no binary-inspection path that can derive the C++ matrix.

**Interim:** Every C++ `perf_*.cpp` has a Python counterpart under
`performance/python/`. Matrices (default / `--full` / `--sizes`) are
hand-mirrored. Default CLI matches C++ (`--full` off). CFAR mid/large
GO/SO/OS Monte Carlo cold builds remain `--full`-only on both sides.

**Status:** open — decide later; do not invent a sync mechanism in the meantime.

## Test and simulator backlog

_Original source: `tests/OPEN_ITEMS.md`_

### 1. MatX / GPU path for the analytic point-target simulator?

**Open question (2026-07-19).** Should C++ demos synthesize ADC cubes on the
GPU via MatX, instead of (or in addition to) the host STL
`radar_common/sim/point_target_sim.hpp` path?

**Context.**

- `point_target_sim.hpp` is **host-only by design** (`STATUS.md` settled:
  simulator + window generation). STL-only, no MatX/CUDA — so tests assert
  against a signal model independent of the operator stack (`TESTING.md`
  tautology guard; fixed `mt19937_64` seed → bit-identical cubes).
- Python demos already synthesize on GPU (CuPy) and emit device tensors;
  C++ demos still call host `simulate()` then upload.
- Dense / MIMO cubes are large enough that host synthesis can matter for
  demo cadence; operators themselves are not the bottleneck in those cases.

**Constraints (do not silently violate).**

- Do **not** replace or MatX-ify the shared SoT header used by unit /
  integration tests. Ground truth must stay derived from the signal model,
  not from MatX code under test.
- A GPU path, if added, must mirror the same far-field dechirped formula and
  noise-power convention; bit-identical noise vs host `mt19937_64` is not
  required (CuPy already differs) unless someone explicitly needs cross-lang
  cube dumps.

**Candidate answers (not chosen).**

1. **Status quo** — keep host SoT + host C++ demo sources; Python stays on
   CuPy. Only revisit if profiling shows host sim blocking a target frame
   period.
2. **Demo-only MatX synthesizer** — e.g. `applications/common/...` or a
   non-SoT helper that emits device tensors; tests keep calling
   `point_target_sim.hpp`.
3. **Shared MatX helper + host SoT** — same as (2), but factored for reuse
   across C++ demos; still never the tautology source.

**Interim.** Leave host SoT alone. Measure dense-SIMO / MIMO C++ host-sim
cost before building a MatX path. Python CuPy already covers the “GPU-side
demo sim” experiment on that side.

**Status:** open — needs a deliberate pick (likely after a cheap timing
diagnostic), not a contract blocker.

### 2. Simulator should regenerate noise each frame

**Status:** open — low-impact change; performance hit needs to be studied before landing.

The host `point_target_sim.hpp` path draws noise once at construction with a
fixed `mt19937_64` seed, so every frame gets identical noise. For demo realism
the simulator should reseed or draw fresh noise on each `simulate()` call.

**Constraints before landing.**

- Unit / integration tests depend on bit-identical cubes across runs (fixed
  seed is a tautology guard; see `TESTING.md`). The fresh-noise path must be
  opt-in (e.g. a `per_frame_noise` flag or a separate entry point) so the
  deterministic SoT header is untouched.
- The Python / CuPy demo path already draws per-frame noise; the concern is
  the C++ host path only.
- Measure the host noise-generation cost at demo cube sizes (dense-SIMO / MIMO
  xl) before landing — noise draw is O(N) random samples and may be negligible
  relative to upload, or may matter at high frame rates.

<!--
Recently landed:
- Mirror the operator tree under `tests/cpp/` (common/, fmcw/, detection/,
  clustering/; integration label for cross-family tests).
- Split `test_common.cpp` along `radar_common` layers into common/test_window,
  test_descriptors, test_sdk, test_sim, test_tensor.
-->

## Additional coverage identified during consolidation

_Source directories: `tests/cpp/fmcw/`_

### Remaining live-override coverage

Add multi-tick Graph tests for `VirtualArrayAssemblyOp.wavelength` and
`FMCWRangeDopplerProcessorOp.clutter_filter`. The former must verify that the
emitted geometry matches the override for exactly one tick; the latter must
exercise an override whose method changes the required clutter workspace and
then verify reversion to Params on the following unmatched tick.
