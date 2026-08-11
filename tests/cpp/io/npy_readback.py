# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read TensorDumpOp's output with NumPy itself.

The C++ test can only check our writer with our own parser, which proves the
two agree and nothing more. This asserts that the files load in the library
whose format we claim to be writing, and that the values match a formula both
sides compute independently rather than bytes handed over from one to the
other.

Usage: npy_readback.py <artifact-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def ramp(i: np.ndarray) -> np.ndarray:
    """The value the C++ test writes at flat index i, computed here afresh."""
    return np.float32(i) * np.float32(1.5) - np.float32(3.0)


def check_complex_npy(path: Path) -> None:
    array = np.load(path)
    assert array.shape == (2, 3, 4), f"{path}: shape {array.shape}"
    assert array.dtype == np.complex64, f"{path}: dtype {array.dtype}"
    expected_real = ramp(np.arange(24)).reshape(2, 3, 4)
    np.testing.assert_array_equal(array.real, expected_real)
    np.testing.assert_array_equal(array.imag, -expected_real * np.float32(0.25))


def check_real_npy(path: Path) -> None:
    array = np.load(path)
    assert array.shape == (10,), f"{path}: shape {array.shape}"
    assert array.dtype == np.float64, f"{path}: dtype {array.dtype}"
    np.testing.assert_array_equal(array, ramp(np.arange(10)).astype(np.float64))


def check_raw(path: Path) -> None:
    # raw carries neither shape nor dtype, so the reader supplies both -- the
    # trade the format makes, asserted rather than assumed.
    array = np.fromfile(path, dtype=np.complex64).reshape(2, 3, 4)
    expected_real = ramp(np.arange(24)).reshape(2, 3, 4)
    np.testing.assert_array_equal(array.real, expected_real)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    artifacts = Path(sys.argv[1])
    check_complex_npy(artifacts / "readback_c8_0.npy")
    check_real_npy(artifacts / "readback_f8.npy")
    check_raw(artifacts / "readback_c8.bin")
    print(f"npy_readback: 3 artifacts in {artifacts} load correctly in NumPy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
