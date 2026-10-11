# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# operators/ is organized per functional group:
#   common/           — shared radar_common library (descriptors, windows, utils)
#   detection/        — detection operators (CFAR)
#   fmcw/             — FMCW signal processing (+ future MIMO marshaling)
#   clustering/       — clustering operators (DBSCAN)
#   angle_estimation/ — angle estimation operators (beamformers, subspace DOA)

# The shared library is not gated: every operator group links against it.
add_subdirectory(common)

# Each operator is gated on its own OP_<name> option, defaulting to ${BUILD_ALL}.
# One add_holohub_operator() call per operator (inside the group's
# CMakeLists.txt for grouped operators).
add_subdirectory(fmcw)

# SAR image-formation operators (ported slices are registered in the group).
add_subdirectory(sar)
add_subdirectory(io)

# Detection operators (one add_holohub_operator() per operator inside).
add_subdirectory(detection)

# Clustering operators (one add_holohub_operator() per operator inside).
add_subdirectory(clustering)

# Angle-estimation operators (one add_holohub_operator() per operator inside).
add_subdirectory(angle_estimation)

# One HSDK 5 provider owns all FMCW operator registrations.
if(HOLOSCAN_RADAR_BUILD_PYTHON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/python/holoscan_radar
                     ${CMAKE_BINARY_DIR}/python/holoscan_radar)
endif()
