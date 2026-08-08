# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a complex or magnitude SAR image to PNG.

Deliberately offline and deliberately plain. `SIGPROC_PLAN.md` gates the mode
1 demo on the formed image being agreed correct *before* any Holoviz work
starts, so this exists to make that judgement possible without a display
pipeline standing between the operator and the eye.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def to_db(image: np.ndarray, *, floor_db: float = -45.0) -> np.ndarray:
    """Magnitude in dB relative to the scene peak, clipped at ``floor_db``."""
    magnitude = np.abs(image)
    peak = magnitude.max()
    if peak <= 0.0:
        raise ValueError("image is identically zero -- nothing to render")
    db = 20.0 * np.log10(magnitude / peak + 1e-12)
    return np.clip(db, floor_db, 0.0)


def render(
    image: np.ndarray,
    output: Path,
    *,
    title: str | None = None,
    floor_db: float = -45.0,
    markers: list[dict] | None = None,
) -> Path:
    """Write ``image`` to ``output`` as a labelled greyscale PNG.

    ``markers`` entries take ``index`` as a ``(v, u)`` pair and an optional
    ``label``; they are drawn as open circles, which is how a predicted pixel
    is checked against where the energy actually landed.
    """
    db = to_db(image, floor_db=floor_db)

    figure, axes = plt.subplots(figsize=(9, 7), dpi=110)
    mesh = axes.imshow(db, cmap="gray", vmin=floor_db, vmax=0.0, origin="lower", aspect="equal")
    figure.colorbar(mesh, ax=axes, label="dB below peak")

    for marker in markers or []:
        row, column = marker["index"]
        axes.plot(column, row, "o", mfc="none", mec="#ff3b30", ms=16, mew=1.4)
        if marker.get("label"):
            axes.annotate(
                f"  {marker['label']}",
                (column, row),
                color="#ff3b30",
                fontsize=9,
                va="center",
            )

    axes.set_xlabel("azimuth sample (u)")
    axes.set_ylabel("slant range sample (v)")
    if title:
        axes.set_title(title)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output
