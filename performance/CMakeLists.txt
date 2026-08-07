# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Performance characterization suite. Built only when
# HOLOSCAN_RADAR_BUILD_PERFORMANCE=ON. These are standalone measurement
# executables; they are not part of the operator libraries or the packaged
# module, and deliberately add no third-party benchmarking dependencies.

add_subdirectory(cpp)
