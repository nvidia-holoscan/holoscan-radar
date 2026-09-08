# Shared angle-estimation open items

## EFT source-count threshold fidelity

MUSIC, Root-MUSIC, and ESPRIT share the EFT source-count helper. Its
closed-form noise-eigenvalue profile uses a first-order Gaussian threshold
rather than Monte-Carlo-calibrated η_P. Consequently, `source_count_pfa` is a
target rather than a guaranteed false-alarm rate; rank-deficient inputs fall
back to one source.

Keep this as shared algorithm research rather than resolving it during an
operator cleanup. Any replacement needs a validated statistical study and
documented accuracy expectations across supported channel, snapshot, SNR, and
precision ranges.

## MUSIC denominator floor

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

## Subspace model-order performance sweeps

MUSIC now measures fixed source counts 1, 2, and
`min(4, num_channels - 1)` independently from its live-override matrix.
Root-MUSIC and ESPRIT also have model-order-dependent work but do not yet carry
equivalent sweeps. Add estimator-specific fixed-order scenarios when their
performance harnesses are next revised; do not multiply each operator's full
live-override matrix by every model order.
