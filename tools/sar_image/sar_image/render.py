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

# Window the colormap against a robust high percentile, not the single
# brightest pixel. A real collect's peak is a specular; pinning 0 dB
# there crushes the rest of the scene into a few gray levels.
_HIGH_PERCENTILE = 99.8


def to_db(image: np.ndarray, *, floor_db: float = -30.0) -> np.ndarray:
    """Magnitude in dB relative to a robust high percentile, clipped.

    ``floor_db`` is the window below that percentile (negative). The
    single brightest pixel is allowed to clip.
    """
    magnitude = np.abs(np.asarray(image))
    while magnitude.ndim > 2:
        magnitude = magnitude[0]
    valid = magnitude[np.isfinite(magnitude) & (magnitude > 0.0)]
    if valid.size == 0:
        raise ValueError("image is identically zero -- nothing to render")
    high = float(np.percentile(valid, _HIGH_PERCENTILE))
    if high <= 0.0:
        raise ValueError("image is identically zero -- nothing to render")
    db = 20.0 * np.log10(magnitude / high + 1e-12)
    return np.clip(db, floor_db, 0.0)


def render(
    image: np.ndarray,
    output: Path,
    *,
    floor_db: float = -30.0,
) -> Path:
    """Write ``image`` to ``output`` as a native-resolution greyscale PNG."""
    db = to_db(image, floor_db=floor_db)
    span = -floor_db
    if span <= 0.0:
        raise ValueError("floor_db must be negative")
    raster = np.rint((db - floor_db) / span * 255.0).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Row 0 is the first PNG row (top), matching storage.
    Image.fromarray(raster, mode="L").save(output)
    return output
