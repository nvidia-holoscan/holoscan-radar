# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# conftest.py — makes the build-tree Python package visible to pytest.
#
# Holoscan SDK is installed as a regular package (has __init__.py), so Python
# discards namespace-package directories alongside it. We extend holoscan.__path__
# directly after import to insert our build-tree holoscan/ directory, making
# holoscan.radar resolvable to our compiled modules.

import os
import sys

# Allow `from utils.harness import ...` from tests under tests/python/.
_PYTHON_TEST_ROOT = os.path.dirname(__file__)
if _PYTHON_TEST_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_TEST_ROOT)

build_dir = os.environ.get(
    "HOLOSCAN_RADAR_BUILD_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "build"),
)

try:
    import holoscan  # noqa: E402

    # CMake stages under build/python/lib/holoscan (HOLOSCAN_RADAR_PYTHON_ROOT).
    build_holoscan_path = os.path.join(build_dir, "python", "lib", "holoscan")
    if build_holoscan_path not in holoscan.__path__:
        holoscan.__path__.insert(0, build_holoscan_path)
    # 1) Load holoscan.radar .so's BEFORE CuPy — CuPy-first makes MatX eig
    #    GetCache().Clear abort on this stack.
    # 2) Import CuPy next (registers its CUDA atexit).
    # 3) Register MatX cache retirement AFTER CuPy so we run first on exit
    #    (atexit is LIFO) while CUDA is still alive.
    try:
        import holoscan.radar  # noqa: E402,F401
        try:
            import cupy  # noqa: E402,F401
        except ImportError:
            pass
        else:
            holoscan.radar.install_matx_cache_atexit()
    except ImportError:
        pass
except ImportError:
    # holoscan not importable (e.g. CUDA not available outside the dev container).
    # Individual tests use pytest.importorskip("holoscan") to skip gracefully.
    pass
