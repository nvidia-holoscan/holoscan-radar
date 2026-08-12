# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reference time-domain backprojector.

This is the independent statement of SAR image geometry that the simulator's
sidecar is checked against, and that `BackprojectionOp` will be measured
against once it exists. Its value is entirely in being *someone else's*
implementation, so two rules govern it:

- **It shares no code with `tools/sar_sim`.** Validating a simulator with the
  simulator's own geometry would assert nothing.
- **It stays a literal transcription of the algorithm** -- compress in range,
  then sum each pulse's contribution at the delay each pixel implies. It is
  not the place to try the approximations `BackprojectionOp` will make; if it
  adopted them, agreement between the two would stop being evidence.

Running on the GPU costs none of that, since the independence that matters is
of authorship and algorithm rather than of array library.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

SPEED_OF_LIGHT = 299792458.0


def _as_complex(signal) -> np.ndarray:
    """CPHD signal samples as complex128, in whichever container they arrive.

    A CF8 collection reads back as complex floats, but CI2 and CI4 read back
    as a *structured* pair of signed integers, which numpy refuses to cast to
    complex at all. They are plain signed values -- no offset, no
    normalisation -- so the conversion is a cast, and whatever scaling the
    collection needs is what AmpSF carries.
    """
    array = np.asarray(signal)
    if array.dtype.names is None:
        return array.astype(np.complex128)
    return array["real"].astype(np.float64) + 1j * array["imag"].astype(np.float64)


def _amplitude_scale(pvps, num_pulses: int) -> np.ndarray | None:
    """The per-pulse AmpSF column, or None when the collection declares none.

    Omitting it forms a perfectly focused image at the wrong brightness --
    wrong by a smooth per-pulse factor, so nothing about the image looks
    broken.
    """
    names = getattr(pvps.dtype, "names", None)
    if names is None or "AmpSF" not in names:
        return None
    return np.asarray(pvps["AmpSF"], dtype=float).reshape(num_pulses, 1)


def backproject(signal, pvps, lattice, *, oversample: int = 8) -> np.ndarray:
    """Form a complex image on ``lattice`` from CPHD signal and PVP arrays.

    ``lattice`` is the sidecar's ``sample_geometry`` mapping: ``origin``,
    ``axis_v``, ``axis_u``, ``spacing`` and ``counts``.
    """
    samples = _as_complex(signal)
    scale = _amplitude_scale(pvps, samples.shape[0])
    if scale is not None:
        samples = samples * scale
    signal = cp.asarray(np.ascontiguousarray(samples))
    num_samples = signal.shape[1]
    padded = num_samples * oversample
    # The positive-exponent transform, which numpy and cupy both spell ifft.
    # Using fft here mirrors the image about the reference point.
    compressed = cp.fft.ifft(signal, n=padded, axis=1) * padded

    sc0 = float(pvps["SC0"][0])
    range_step = SPEED_OF_LIGHT / (2.0 * padded * float(pvps["SCSS"][0]))

    origin = cp.asarray(np.asarray(lattice["origin"], dtype=float))
    axis_v = cp.asarray(np.asarray(lattice["axis_v"], dtype=float))
    axis_u = cp.asarray(np.asarray(lattice["axis_u"], dtype=float))
    spacing_v, spacing_u = lattice["spacing"]
    num_v, num_u = lattice["counts"]

    grid_v = cp.arange(num_v)[:, None, None] * spacing_v * axis_v[None, None, :]
    grid_u = cp.arange(num_u)[None, :, None] * spacing_u * axis_u[None, None, :]
    pixels = (origin + grid_v + grid_u).reshape(-1, 3)

    tx_pos = cp.asarray(np.asarray(pvps["TxPos"], dtype=float))
    srp_pos = cp.asarray(np.asarray(pvps["SRPPos"], dtype=float))

    image = cp.zeros(pixels.shape[0], dtype=complex)
    # One kernel launch per pulse, chosen for clarity over speed: the loop
    # reads as the sum it implements. Batching the pulses would be faster, but
    # it is the kind of rewrite that lands this on the same approximations
    # `BackprojectionOp` makes, and agreement reached that way is not evidence.
    for pulse in range(signal.shape[0]):
        apc = tx_pos[pulse]
        delta_r = cp.linalg.norm(pixels - apc, axis=1) - cp.linalg.norm(srp_pos[pulse] - apc)
        index = delta_r / range_step
        low = cp.floor(index).astype(cp.int64)
        frac = index - low
        sample = (
            compressed[pulse][low % padded] * (1.0 - frac)
            + compressed[pulse][(low + 1) % padded] * frac
        )
        # Undo the carrier phase factored out by the range transform. The
        # demodulation frequency is SC0, the first sample's frequency, not
        # the band centre -- that pairing is set by the frequency axis the
        # transform ran over.
        image += sample * cp.exp(1j * 4.0 * cp.pi * sc0 * delta_r / SPEED_OF_LIGHT)

    return cp.asnumpy(image.reshape(num_v, num_u))
