# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-repo SAR collection simulator.

Generates CPHD fixtures from committed recipes. No CPHD binaries live in git,
so this is how both CI smoke fixtures and demo scenes come into existence —
and, because users are expected to generate their own test input, it is
public and self-contained rather than a wrapper around anything internal.

Python-only by design: `sarpy` / `sarkit` are confined to this generate path,
and no operator, test binary, or application links against them.
"""

from .cphd import build_pvps, build_xmltree, validate_xmltree, write_cphd
from .generate import generate
from .geometry import CollectionGeometry, build_geometry, ecef_to_llh, llh_to_ecef
from .recipe import Collection, Geometry, PointTarget, Recipe, Waveform, load_recipe
from .signal import synthesize, verify_range_compression
from .truth import build_sidecar, suggested_sample_geometry, write_sidecar

__all__ = [
    "Collection",
    "CollectionGeometry",
    "Geometry",
    "PointTarget",
    "Recipe",
    "Waveform",
    "build_geometry",
    "build_pvps",
    "build_sidecar",
    "build_xmltree",
    "ecef_to_llh",
    "generate",
    "llh_to_ecef",
    "load_recipe",
    "suggested_sample_geometry",
    "synthesize",
    "validate_xmltree",
    "verify_range_compression",
    "write_cphd",
    "write_sidecar",
]
