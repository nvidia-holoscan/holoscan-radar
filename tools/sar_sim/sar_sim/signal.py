# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-target phase-history synthesis.

CPHD is *compensated* phase history: the signal has already been deramped and
motion-compensated to a reference point, so what is stored for pulse ``v`` and
frequency sample ``k`` is the residual phase of each scatterer relative to
that reference. For an ideal point at ``p`` with the antenna phase centre at
``a(v)`` and the stabilization reference point at ``srp(v)``, the differential
range is

    dR(v) = |a(v) - p| - |a(v) - srp(v)|

and the two-way residual phase at frequency ``f`` is ``-4 pi f dR / c``.

**The sign is a convention, and it is the convention.** It pairs with
``Global/SGN = -1`` in the written XML: with that pair, transforming the
frequency axis to the delay (TOA) axis with a forward ``exp(+2 pi i f t)``
kernel puts a target at *positive* ``dR`` at a *positive* delay, and
backprojection conjugates the same phase to focus. Everything downstream —
the reader, `SARRangeCompressionOp`, `BackprojectionOp` — has to agree with
this or the image mirrors about the reference point. That failure is quiet:
a mirrored image of a symmetric scene looks fine.

Two things guard it. `verify_range_compression` below is a self-check that a
target lands at its predicted delay bin, which catches an internal
inconsistency between this model and the declared ``SGN``. Agreement with the
*outside world* is not something a self-consistent simulator can establish,
and is left to the phase 2 comparison against a real collection.

Synthesis runs on the GPU; the self-check deliberately does not. It is one
pulse and costs milliseconds either way, and running it on a different array
stack than the thing it checks means a fault in one is not reproduced by the
other.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from .geometry import SPEED_OF_LIGHT, CollectionGeometry
from .recipe import Recipe

# Device memory the (pulses x targets x samples) intermediate may occupy, and
# the measured cost of one of its elements: a float64 phase, its complex128
# exponential and the reflectivity-scaled copy coexist, which comes to about
# 72 bytes. Budgeting elements rather than fixing a pulse count is what bounds
# the working set in all three dimensions -- a fixed chunk of 1024 pulses is
# 720 MiB for a 5-target scene and 7.2 GiB for a 50-target one.
_MEMORY_BUDGET_BYTES = 512 * 1024**2
_BYTES_PER_ELEMENT = 72


def frequency_axis(recipe: Recipe) -> np.ndarray:
    """The per-sample transmit frequencies, in Hz.

    The band is centred on the carrier and spans the chirp bandwidth, which
    is what `Global/FxBand` and the per-vector `SC0` / `SCSS` PVPs describe.
    """
    waveform = recipe.waveform
    fx_min = waveform.carrier_frequency - 0.5 * waveform.bandwidth
    num_samples = recipe.collection.num_samples
    spacing = waveform.bandwidth / (num_samples - 1)
    return fx_min + spacing * np.arange(num_samples, dtype=float)


def target_positions_ecef(recipe: Recipe, geometry: CollectionGeometry) -> np.ndarray:
    """Scene targets as ECEF positions, shape (num_targets, 3)."""
    enu = np.array(
        [[t.east_m, t.north_m, t.up_m] for t in recipe.targets],
        dtype=float,
    )
    return geometry.enu_to_ecef(enu)


def target_reflectivity(recipe: Recipe) -> np.ndarray:
    """Complex reflectivity per target, shape (num_targets,)."""
    amplitude = np.array([t.amplitude for t in recipe.targets], dtype=float)
    phase = np.radians(np.array([t.phase_deg for t in recipe.targets], dtype=float))
    return amplitude * np.exp(1j * phase)


def differential_range(apc_ecef, srp_ecef, targets_ecef, xp=np):
    """Per-pulse, per-target differential range in metres, (pulses, targets).

    Array-module agnostic so the GPU synthesis path and the host self-check
    below can share one statement of the geometry.
    """
    to_target = targets_ecef[None, :, :] - apc_ecef[:, None, :]
    to_srp = srp_ecef - apc_ecef
    return xp.linalg.norm(to_target, axis=2) - xp.linalg.norm(to_srp, axis=1)[:, None]


def _chunk_shape(num_targets: int, num_samples: int) -> tuple[int, int]:
    """Pulses and targets per synthesis chunk, within the memory budget.

    Pulses are split first, and targets only when a single pulse row does not
    fit on its own. That order matters: splitting the target axis regroups the
    sum over it, and a regrouped float64 sum differs from the whole-axis one
    in its last bits. Fixtures are regenerated and compared byte for byte, so
    the grouping is held fixed for every collection that has a choice.
    """
    budget = max(_MEMORY_BUDGET_BYTES // _BYTES_PER_ELEMENT, 1)
    per_pulse = num_targets * num_samples
    if per_pulse <= budget:
        return max(budget // per_pulse, 1), num_targets
    return 1, max(budget // num_samples, 1)


def synthesize(
    recipe: Recipe, geometry: CollectionGeometry, *, dtype: str = "complex64"
) -> np.ndarray:
    """Synthesize the (num_pulses, num_samples) phase-history array.

    Runs on the GPU: the work is one complex exponential per
    (pulse, target, sample), which is millions of independent transcendentals
    and nothing else. Returned as a host array because the CPHD writer takes
    one.

    Phase is accumulated in float64 and cast once at the end: the argument
    reaches tens of thousands of radians for a realistic X-band geometry, and
    a float32 accumulation there loses the fractional part that carries all
    the information.
    """
    freqs = cp.asarray(frequency_axis(recipe))
    targets = cp.asarray(target_positions_ecef(recipe, geometry))
    reflectivity = cp.asarray(target_reflectivity(recipe))
    # Spotlight: the antenna stares at the IARP, so the SRP is fixed for the
    # whole collection. That is what makes `SRPFixed = true` honest.
    srp = cp.asarray(geometry.iarp_ecef)
    apc = cp.asarray(geometry.apc_ecef)

    num_pulses = geometry.num_pulses
    num_targets = int(targets.shape[0])
    signal = np.zeros((num_pulses, freqs.size), dtype=dtype)
    scale = -4.0 * np.pi / SPEED_OF_LIGHT
    pulse_chunk, target_chunk = _chunk_shape(num_targets, freqs.size)

    for start in range(0, num_pulses, pulse_chunk):
        stop = min(start + pulse_chunk, num_pulses)
        delta_r = differential_range(apc[start:stop], srp, targets, xp=cp)
        # Seeded from the first block rather than from zeros so that the
        # single-block case — every recipe whose target axis fits the budget —
        # is the same reduction it always was, down to the sign of a zero.
        accumulated = None
        for first in range(0, num_targets, target_chunk):
            last = min(first + target_chunk, num_targets)
            phase = scale * delta_r[:, first:last, None] * freqs[None, None, :]
            block = cp.exp(1j * phase) * reflectivity[None, first:last, None]
            partial = block.sum(axis=1)
            accumulated = partial if accumulated is None else accumulated + partial
        signal[start:stop] = cp.asnumpy(accumulated.astype(dtype))

    return signal


def verify_range_compression(
    recipe: Recipe, geometry: CollectionGeometry, signal: np.ndarray
) -> dict[str, float]:
    """Check that the mid-aperture pulse compresses to the predicted delays.

    This is the simulator's tautology guard against itself: the phase model
    and the declared ``SGN`` are two statements of the same convention, and
    nothing else in the pipeline notices when they disagree. Compressing one
    pulse and asking where the targets landed costs milliseconds and catches
    a flipped sign, a wrong factor of two, and a swapped frequency axis.

    Returns the worst per-target error in metres, alongside the resolution it
    should be compared against.
    """
    pulse_index = geometry.num_pulses // 2
    freqs = frequency_axis(recipe)
    num_samples = freqs.size

    # Zero-pad so the delay grid is fine enough that discretization is not
    # what the assertion measures.
    oversample = 16
    padded = np.zeros(num_samples * oversample, dtype=complex)
    padded[:num_samples] = signal[pulse_index]
    # numpy's ifft carries the exp(+2 pi i k m / N) kernel, which is the
    # positive-exponent transform the module docstring pairs with this
    # model's phase sign. np.fft.fft is the conjugate kernel and puts every
    # target at the negative of its true delay -- the mirrored image this
    # check exists to catch.
    compressed = np.fft.ifft(padded)

    sample_spacing = freqs[1] - freqs[0]
    # Range bin size: a delay bin of 1 / (N_padded * df) is a two-way delay,
    # so the one-way range step is c / 2 times that.
    range_step = SPEED_OF_LIGHT / (2.0 * padded.size * sample_spacing)

    targets = target_positions_ecef(recipe, geometry)
    expected = differential_range(
        geometry.apc_ecef[pulse_index : pulse_index + 1], geometry.iarp_ecef, targets
    )[0]

    magnitude = np.abs(compressed)
    worst_error = 0.0
    for target_range in expected:
        bin_index = int(round(target_range / range_step)) % padded.size
        # Search a small neighborhood: the peak should be at the predicted
        # bin, and allowing a couple of bins keeps the check about the
        # convention rather than about interpolation.
        window = 3
        indices = (np.arange(bin_index - window, bin_index + window + 1)) % padded.size
        peak = int(indices[int(np.argmax(magnitude[indices]))])
        # Targets nearer than the reference point have negative differential
        # range and land in the upper half of the transform.
        if peak > padded.size // 2:
            peak -= padded.size
        measured = peak * range_step
        worst_error = max(worst_error, abs(measured - target_range))

    return {
        "worst_range_error_m": float(worst_error),
        "range_resolution_m": float(recipe.waveform.range_resolution),
        "range_step_m": float(range_step),
    }
