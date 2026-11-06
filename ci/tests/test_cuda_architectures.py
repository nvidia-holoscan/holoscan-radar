# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the default CUDA architectures in CMake script mode, without a GPU or toolkit."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "cmake/HoloscanRadarCudaArchitectures.cmake"


def architectures(*definitions, cudaarchs=None):
    env = {key: value for key, value in os.environ.items() if key != "CUDAARCHS"}
    if cudaarchs is not None:
        env["CUDAARCHS"] = cudaarchs
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "probe.cmake"
        script.write_text(
            f'include("{MODULE.as_posix()}")\n'
            'message(NOTICE "archs=[${CMAKE_CUDA_ARCHITECTURES}]")\n'
        )
        result = subprocess.run(
            ["cmake", *(f"-D{d}" for d in definitions), "-P", str(script)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    return result.stderr.rsplit("archs=[", 1)[1].split("]", 1)[0]


class CudaArchitectureDefaults(unittest.TestCase):
    def test_development_builds_target_the_build_gpu(self):
        self.assertEqual(architectures(), "native")

    def test_package_builds_need_no_gpu(self):
        self.assertEqual(architectures("PKG_holoscan_radar=ON"), "all-major")
        self.assertEqual(architectures("SKBUILD=2"), "all-major")

    def test_explicit_architectures_override_package_default(self):
        self.assertEqual(
            architectures("PKG_holoscan_radar=ON", "CMAKE_CUDA_ARCHITECTURES=80;90"), "80;90"
        )

    def test_environment_defers_to_cmake(self):
        self.assertEqual(architectures("PKG_holoscan_radar=ON", cudaarchs="80"), "")

    def test_old_architectures_are_floored(self):
        self.assertEqual(architectures("CMAKE_CUDA_ARCHITECTURES=60"), "70")


if __name__ == "__main__":
    unittest.main()
