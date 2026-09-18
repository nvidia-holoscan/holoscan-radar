# Simulator Range-Dependent Path Loss

## Goal

Add an optional relative monostatic free-space range law to the analytic FMCW
simulator so a distant or CRI-folded target is attenuated according to its full
physical range. Preserve the existing constant-amplitude model for analytic
operator tests that isolate phase, filtering, or aliasing behavior.

This models the range-dependent term of the radar equation. It does not turn
the point-target simulator into a calibrated RF or electromagnetic scene
simulator.

## Fidelity boundary

For full physical range R, use the voltage-domain gain

```text
g_range(R) = (reference_range / R)^2
```

so received power scales as

```text
|g_range(R)|^2 = (reference_range / R)^4.
```

TX power, wavelength, antenna gains, target RCS, receiver gain, and fixed
system loss remain absorbed into the target complex amplitude at the reference
range.

Absolute radar-equation calibration, aspect-dependent RCS, antenna patterns,
radome and atmospheric loss, multipath, clutter, AGC, saturation, quantization,
RF nonlinearities, near-field propagation, and element-dependent spherical
spreading remain out of scope.

## Scene contract

Add pure-model configuration in radar_common/sim/scene.hpp:

```cpp
enum class RangeLossModel {
  kNone,
  kMonostaticFreeSpace,
};

struct RangeLossConfig {
  RangeLossModel model{RangeLossModel::kNone};
  double reference_range{1.0};  // m
};
```

Add RangeLossConfig range_loss to SimScene. This is simulator-internal
configuration, not a shared descriptor, Tensor port, or operator contract.

PointTarget amplitude has these explicit semantics:

- kNone: the current constant complex ADC-voltage coefficient.
- kMonostaticFreeSpace: the complex received-voltage coefficient the target
  would produce at reference_range before the analog IF response.

kNone remains the SimScene default, preserving existing analytic fixtures and
callers. Applications claiming realistic moving-target strength opt in.

Validate reference_range as finite and positive in free-space mode. Continue
rejecting targets at the origin. Reject a non-finite or non-positive modeled
within-CPI range instead of silently clamping the singularity.

## Signal behavior

For each target and slow-time slot, use the existing stop-and-hop physical
center range:

```text
R_k = R_0 + v_r * k * physical_PRI
```

The clean contribution becomes:

```text
target.amplitude
  * g_range(R_k)
  * H_IF(slope * residual_delay)
  * TX_excitation
  * exp(j * phase(full_delay, residual_delay)).
```

Requirements:

- Compute range gain from full physical R_k, never CRI-folded apparent range.
- Apply gain before target summation, pre-mixer remodulation, and the
  post-solver noise additive.
- Preserve carrier, Doppler, spatial, TDM-slot, and DDM-code phase behavior.
- Use one scalar center-range gain for every TX/RX channel, consistent with the
  current far-field plane-wave approximation.
- Update gain between chirps as R_k changes; retain stop-and-hop within a chirp.

The existing noise convention remains relative. With loss enabled, snr_db is
the SNR of a unit-amplitude target at reference_range. A target with amplitude
A at R has signal-power offset

```text
20 log10(|A|) + 40 log10(reference_range / R) dB.
```

## Application configuration

Add an optional application-owned block to the shared SISO, SIMO, and MIMO
configuration schemas:

```yaml
app:
    range_loss:
        model: monostatic_free_space
        reference_range: 100.0
```

Accepted models are none and monostatic_free_space. Omission selects none. The
application loader warns when none is selected explicitly or by omission because
target amplitude then remains independent of physical range. Reject unknown
models and invalid or missing reference ranges.

All normal application presets explicitly enable free-space loss with a 100 m
reference. Short smoke presets retain none as deterministic constant-amplitude
analytic fixtures and therefore emit the warning.

## Independent implementations

Implement the equation independently in:

- tests/cpp/utils/sim_reference.hpp for SISO/SIMO;
- tests/cpp/utils/mimo_reference.hpp for TDM and both DDM encodings;
- operators/common/radar_common/sim/simulator.cu for the unified GPU solver.

Do not share the production range-gain helper with the CPU oracle.

## Verification

### CPU self-checks

- kNone preserves current constant-amplitude output exactly.
- A unit target at reference_range has unity gain.
- Doubling range divides complex amplitude by four and power by sixteen,
  or -12.0412 dB.
- Complex target phase is preserved while magnitude is scaled.
- Moving-target chirp magnitudes follow the analytic inverse-square voltage
  ratio while Doppler phase is unchanged.
- Dechirped and pre-mixer paths have identical attenuation after software
  dechirp.
- Invalid reference range and within-CPI range crossing zero reject.

### CRI ambiguity

- Equal targets at R and q*c*CRI/2 + R land at the same apparent range when
  their residual beats are in band, but their amplitude ratio follows their
  full physical ranges.
- Compare 40 m with 3787.405725 m for the 25 us CRI case. The far target has
  the same 40 m apparent range and must be approximately 79.05 dB below the
  physical 40 m target before IF loss and noise.
- Cover exact-CRI and just-beyond-CRI targets; folding must never restore
  near-range amplitude.
- Cover q greater than one independently of cyclic TDM/DDM excitation shifts.

### MIMO and GPU parity

- Cover SIMO, TDM, uniform-offset DDM, and arbitrary phase-coded DDM.
- All far-field RX channels receive the same scalar gain while retaining
  spatial phase.
- Compare GPU float output with independent CPU double references at the
  reference range, twice that range, and a CRI-folded full range.
- Include filtered, unfiltered, and pre-mixer cases to lock operation ordering.

### Noise and processing integration

- With a fixed seed, measured target SNR changes by the analytic
  40*log10(R1/R2) ratio while noise variance is unchanged.
- Feed near and CRI-folded targets through range/Doppler processing and compare
  peak-power ratios in physical bins with the analytic range law.
- Add deterministic CFAR coverage where the reference-range target is detected
  and an otherwise-identical folded target is below threshold. Keep the
  peak-power ratio as the primary oracle.
- Test SISO config decoding for both modes and invalid inputs, plus focused
  shared-schema decoding in SIMO and MIMO.

CI constructs static scenes at the required physical ranges. It must not wait
for a target to move for hundreds of seconds.

## Documentation

- Update TESTING.md to require relative monostatic loss while retaining
  absolute link-budget and EM effects out of scope.
- Update the simulator README with amplitude and SNR reference semantics,
  equations, the full-range-before-folding rule, and an example.
- Update SISO documentation and YAML comments for the manual wrap demo.
- Do not change OPERATOR_API.md; no operator or shared descriptor changes.

## Implementation order

1. Add scene types, validation, and amplitude/noise semantics.
2. Update independent CPU references and self-checks.
3. Update the unified GPU solver and parity tests.
4. Add application schema decoding and config tests.
5. Enable and document the production SISO demonstration.
6. Add range/Doppler and CFAR integration coverage.
7. Run simulator CPU/GPU suites, affected processing and config tests, SISO
   smoke, the complete non-performance C++ suite, affected Python tests, and
   full pinned pre-commit.

## Acceptance

- kNone is behaviorally unchanged.
- Free-space mode produces analytic inverse-square complex-amplitude and
  inverse-fourth-power scaling from full physical range in every modality.
- CRI folding changes apparent beat range but never resets propagation loss.
- IF filtering, pre-mixer conversion, excitation coding, and noise ordering
  remain correct.
- CPU and GPU results agree within existing tolerances.
- The SISO demonstration shows a distant wrapped target strongly attenuated.
- Documentation distinguishes relative range law from an absolute link budget.
