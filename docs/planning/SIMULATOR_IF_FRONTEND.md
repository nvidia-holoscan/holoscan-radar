# Simulator IF Front End and CRI Range Ambiguity

## Goal

Make the analytic FMCW simulator distinguish two range-ambiguity mechanisms:

- Same-sweep beat frequencies outside the analog IF passband are attenuated
  before sampling.
- Returns delayed by one or more physical chirp repetition intervals (CRIs)
  reappear at their residual delay when that residual beat is in band.

The simulator models an indefinitely repeated, homogeneous chirp and excitation
sequence. Full sample-level transmit gating, cross-profile waveform history,
and link-budget attenuation remain out of scope.

## Scene contract

Add the following pure-model configuration to `SimScene`:

```cpp
enum class IfSamplingMode { kComplexIq, kReal };
enum class AnalogIfResponseType { kRaisedCosine, kNone };

struct AnalogIfConfig {
  IfSamplingMode sampling_mode{IfSamplingMode::kComplexIq};
  AnalogIfResponseType response{AnalogIfResponseType::kRaisedCosine};
  double passband_fraction{0.9};
  double stopband_fraction{1.0};
};
```

`SimResult` records the sampling mode. `to_real_if()` rejects a result that was
not synthesized in `kReal` mode. Applications use the filtered complex-IQ
default; no application YAML surface is added in this change.

For each propagation path, use the physical on-wire CRI stored in
`SimScene::waveform`:

```text
q = floor(delay / physical_CRI)
delta = delay - q * physical_CRI
apparent_range = c * delta / 2
```

The complex-IQ positive-beat boundary is `sample_rate`; the real-IF boundary is
`sample_rate / 2`. The default response is unity through 90% of that boundary,
then a raised-cosine transition to zero at the boundary. `kNone` is explicit
unity gain for tests that intentionally exercise ADC aliasing.

## Signal behavior

- Compute the fast-time beat from the residual delay:
  `f_beat = slope * delta`.
- Apply the IF magnitude to that beat before sampled accumulation.
- Preserve the full propagation delay in the carrier/spatial phase. Under the
  repository's simplified convention:
  `phase = 2*pi*slope*delta*t - 2*pi*carrier*delay`.
- An exact integer-CRI delay therefore lands at zero beat and may subsequently
  be removed by DC, leakage, clutter, or notch processing.
- Do not reject delays greater than the chirp duration or CRI.
- Treat the configured waveform and excitation pattern as periodic across CPI
  boundaries. Cross-profile history is not inferred.

For MIMO, use excitation slot
`modulo(receive_slot - q, excitation_period)`. This leaves SISO/SIMO unit
excitation unchanged, selects the delayed TDM TX slot, and applies the delayed
DDM offset or phase code. Ambiguity orders greater than one wrap across the
complete excitation period.

Apply the IF magnitude before pre-mixer remodulation so software dechirping
reproduces the filtered dechirped result. Thermal noise remains an ADC-referred
post-solver additive and is not IF-shaped.

Validate finite positive waveform fields,
`0 < passband_fraction < stopband_fraction <= 1`, a sample window no longer
than the chirp duration, and PRI no shorter than chirp duration. Normalize
floating residuals close to zero or CRI to zero.

## Verification

- Verify passband, transition midpoint, transition samples, and stopband gains.
- Show that a same-sweep out-of-band target is suppressed by default and folds
  digitally under `kNone`.
- Show that targets separated by `q*c*CRI/2` have the same range-FFT magnitude
  peak while their carrier phases may differ.
- Verify that a target just beyond one CRI passes when its residual beat is in
  band, and an exact-CRI target lands at zero range.
- Exercise a moving target crossing a CRI boundary.
- Verify delayed-slot selection for TDM and delayed offset/code selection for
  both DDM encodings.
- Verify complex-IQ versus real-IF boundaries, pre-mixer equivalence,
  deterministic output, and CPU/GPU parity for SISO, SIMO, TDM, and DDM.
- Preserve existing application target locations. The 245 m targets sit in the
  default transition band, so their attenuation is intentional.

Update the simulator README and `TESTING.md` to distinguish analog IF rejection,
ADC aliasing under the explicit unfiltered model, CRI range folding, and
slow-time Doppler folding. Document that finite chirp/idle overlap, range law,
antenna patterns, multipath, filter phase, and cross-profile history remain out
of scope.
