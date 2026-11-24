<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Debian C++ runtime graph tests

Run this project inside a clean Ubuntu 24.04 CUDA development container with
`holoscan-core`, `libholoscan-radar-cuda-13`, and
`libholoscan-radar-dev-cuda-13` installed from the matching local repository. Mount
the matching source checkout read-only. No Radar operator translation unit
is compiled by this project: CMake finds the installed `holoscan-radar` package,
and the source checkout supplies app graphs, app helpers, presets, and the
synthetic SAR fixture generator. The app graphs use only installed Radar operator
headers and libraries; no operator source include directory is added. The
package's installed public headers are also checked separately by the existing
`tests/packaging/headers` project.

From the checkout root inside the container:

```sh
cmake -S tests/packaging/runtime_consumer -B /tmp/radar-debian-runtime-build
cmake --build /tmp/radar-debian-runtime-build -j 4
ctest --test-dir /tmp/radar-debian-runtime-build --output-on-failure --output-junit /tmp/radar-debian-runtime-results.xml
dpkg-query -W holoscan-core libholoscan-radar-cuda-13 libholoscan-radar-dev-cuda-13
ldd /tmp/radar-debian-runtime-build/bin/siso
```

Run the same commands on representative x86_64 and aarch64 GPU hosts. Keep the CTest
log, JUnit XML, `dpkg-query` output, `sha256sum` of the installed local-repo
DEBs, and `ldd` output for each of the five graph executables. The Radar
libraries reported by `ldd` must resolve to package-owned files under `/lib` or
`/usr/lib`, not a source build.
The FMCW cases require two report frames and a cluster at the injected target;
the SAR cases require the expected image counts and dimensions. A GPU, link,
graph startup, or output failure is a test failure on either host.
