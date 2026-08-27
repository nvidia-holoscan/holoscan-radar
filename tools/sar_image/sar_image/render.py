# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a complex or magnitude SAR image to PNG.

Deliberately offline and deliberately plain. `SIGPROC_PLAN.md` gates the mode
1 demo on the formed image being agreed correct *before* any Holoviz work
starts, so this exists to make that judgement possible without a display
pipeline standing between the operator and the eye.

The PNG is a 1:1 greyscale raster of the dB-windowed magnitude: one file
pixel per image sample, no axes, no pad. Row 0 is the top of the PNG, as
stored. The Atlanta demo lattice has ``axis_v`` = south so row 0 is north;
this function does not resample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Window the colormap against this image's own histogram, not the
# single brightest pixel and not a scale borrowed from another scene.
# A real collect's peak is a specular; pinning 0 dB there crushes the
# rest of the scene into a few gray levels. The floor is the 25th
# percentile of |z| (the dark end of the scene), never wider than
# 30 dB. A 2nd-percentile floor is the sidelobe tail on a filled
# lattice and leaves low-contrast SNAPSHOTs washed in mid-grey.
_HIGH_PERCENTILE = 99.8
_LOW_PERCENTILE = 25.0


def to_db(
    image: np.ndarray,
    *,
    floor_db: float = -30.0,
    low_percentile: float = _LOW_PERCENTILE,
    high_percentile: float = _HIGH_PERCENTILE,
) -> np.ndarray:
    """Magnitude in dB relative to this image's high percentile, clipped.

    ``high_percentile`` sets 0 dB. The floor is this image's
    ``low_percentile``, never wider than ``floor_db`` (negative dB).
    Speculars above the high percentile clip.
    """
    magnitude, high, floor_used = _window(image, floor_db=floor_db,
                                          low_percentile=low_percentile,
                                          high_percentile=high_percentile)
    db = 20.0 * np.log10(magnitude / high + 1e-12)
    return np.clip(db, floor_used, 0.0)


def _window(
    image: np.ndarray,
    *,
    floor_db: float,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, float, float]:
    magnitude = np.abs(np.asarray(image))
    while magnitude.ndim > 2:
        magnitude = magnitude[0]
    valid = magnitude[np.isfinite(magnitude) & (magnitude > 0.0)]
    if valid.size == 0:
        raise ValueError("image is identically zero -- nothing to render")
    high = float(np.percentile(valid, high_percentile))
    if high <= 0.0:
        raise ValueError("image is identically zero -- nothing to render")
    low_from_db = high * 10.0 ** (floor_db / 20.0)
    low = low_from_db
    if low_percentile > 0.0:
        low = max(float(np.percentile(valid, low_percentile)), low_from_db)
    if not (low > 0.0 and low < high):
        low = low_from_db
    floor_used = 20.0 * np.log10(low / high)
    return magnitude, high, floor_used


def render(
    image: np.ndarray,
    output: Path,
    *,
    floor_db: float = -30.0,
    low_percentile: float = _LOW_PERCENTILE,
) -> Path:
    """Write ``image`` to ``output`` as a native-resolution greyscale PNG."""
    if not (floor_db < 0.0):
        raise ValueError("floor_db must be negative")
    magnitude, high, floor_used = _window(
        image,
        floor_db=floor_db,
        low_percentile=low_percentile,
        high_percentile=_HIGH_PERCENTILE,
    )
    span = -floor_used
    if span <= 0.0:
        raise ValueError("display window is empty")
    db = np.clip(20.0 * np.log10(magnitude / high + 1e-12), floor_used, 0.0)
    raster = np.rint((db - floor_used) / span * 255.0).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Row 0 is the first PNG row (top), matching storage.
    Image.fromarray(raster, mode="L").save(output)
    return output
