# Testing requirements

Use the host workflow in [DEVELOPER.md](../DEVELOPER.md). A CUDA-capable NVIDIA GPU is required for operators, applications, and the C++ suite. Host-only simulators and checkout-tool tests are deliberately separate; they never substitute for GPU validation. The CTest driver runs all registered tests by default, including statistical and provider tests; supplied include/exclude label filters narrow that selection.

## Independent analytic ground truth

Derive simulator models independently of processing stages. Assert physical predictions, not a previous output or merely a generator/reader round trip. Keep canonical closed-form self-checks in `tests/cpp/common/test_sim.cpp`; model changes require updating those checks. The [simulator guide](../operators/common/radar_common/sim/README.md) owns signal-model details and assumptions.

For FMCW, use known target range, velocity, angle, amplitude and noise to assert peak/bin locations, FFT signs and shifts, coherent gain, window sidelobes, real/complex paths, and split/fused agreement. Test multi-RX/TX geometry, TDM timing and compensation, DDM coding/demultiplexing, aliasing, and explicit convention conversions. Shared production kernels cannot serve as an independent oracle.

For SAR, derive exact per-pulse differential ranges in double precision. Map image peaks through the declared geometry into physical positions, and compare resolution and sidelobes to bandwidth/aperture/window predictions. Cover off-centre targets, odd/even transforms, the centred stored-sample phase reference, nontrivial frame transforms, accumulation/eviction, collection resets, scrolling overlap, and final partial-aperture flush. Stop-and-hop monostatic point fixtures omit real scattering, distributed clutter, moving targets, atmosphere, and bistatic propagation. They cannot establish real-data fidelity.

## Coverage and ownership

- Exercise float and double, including SAR float-sample/double-geometry pairs and forbidden precision combinations. Choose tolerances from error budgets and conditioning, including ECEF phase sensitivity.
- Test both build-time Params and `[live]` Exact matches, silent/unwired/missed overrides, no latch, return to Params, revisited configurations, shape changes, and pool bounds. Structural options need separate instances.
- Cover named axes, zero/empty detection sets, edges, noncontiguous/malformed shape/dtype/memory-kind inputs, missing CUDA streams, invalid enums/values, and overflow checks. Assert named errors rather than accepting any failure.
- Retain outputs across ticks to expose aliasing and lifetime errors. Exercise optional outputs wired and unwired, full queues, sample-time propagation, aligned joins, and sideband-before-driver publication.
- Cover schema round trips field by field, logical/schema drift, metadata and Tensor bounds, empty optional arrays, invalid descriptors, and Graph publication/receipt. Package tests validate installed headers, exports, dependencies, and Python parity.
- Verify aperture reform at image level against direct formation and boundary behavior through splitting/coalescing stages. More real-scene reform and backend comparisons remain qualification work; do not imply they are all covered by fixture smokes.

Use the SDK's native fixture for a single operator; connect sinks for outputs under assertion and exercise optional outputs intentionally. Broad graph tests must check data, not merely successful completion. For behavior-preserving app refactors, compare deterministic logs but omit `rep_*` medoid representative fields whose near-ties can change with floating accumulation order.

## Statistical assertions

Monte Carlo checks cover noise statistics, detector false-alarm probability, detection probability, estimator bias/variance and multi-source separation where applicable. Use independent seeded trials, justified sample counts and confidence intervals; never tune assertions to one random draw. Preserve the C++ `statistical` tier in ordinary full validation. CFAR calibration is expensive and retains its existing C++/Python library APIs; calibrating and then testing the same training samples is not independent validation.

Autofocus recovery is scored after removing constant and linear phase, which PGA cannot observe. Test asymmetric/odd and even errors: quadratic-only truth hides phase reversal and centre seams. Exercise both axes, support edges, refusal provenance, accepted iteration count, guard monotonicity, and transfer to differently sampled products. The diagnostic sweeps use π/8 residual as their target and π/4 as a floor; neither a convergence code nor image sharpness proves accuracy on a real scene.

## Fixtures and qualification limits

Generated C++ CPHD fixtures make smoke and parity tests reproducible without downloading collections. Assert analytic properties within tolerances, never a cross-platform file checksum: bitwise fixture stability across toolchains remains unproven. Public-data integrity pins instead protect the expected source bytes and metadata, not image quality.

Real-data qualification still needs focus, geometry, amplitude, radiometry, seams, long-aperture autofocus, and all selected backends across the supported SAR modes. Historical parser surveys and GUI visual checks are evidence for those revisions and presets only. They do not qualify this binary or every CPHD format/domain/sign/channel combination. Preserve offline data/network/integrity/progress/checkpoint/viewer and autofocus-report coverage when relocating tools.

## Interactive diagnostic coverage

Received pulse counts follow successful publication; GPU-completed counts follow completion events. Signed evictions and replay must not inflate acquisition. Cover sliced ticks with the same stamp, loop generations, early/final partial cadences, history overflow/recovery, paced first-batch release and overlapping acquisition/formation ETA. Unknown or changed-grid timing stays unknown. Hold a real CUDA stream pending and exhaust the fixed event ring to establish that polling neither waits nor claims completion. Test wide/compact text and actual C++/Python X11 bars through first-image handoff, with numerical/dump parity, independent main-frame uploads, delayed-display recovery and bounded host resources checked separately.
