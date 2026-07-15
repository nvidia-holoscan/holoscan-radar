# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Python3 + pybind11 are pulled in (find_package + FetchContent) by
# pybind11_add_holohub_module, which the operator's own python/CMakeLists.txt
# includes when cpp == 'cpp'. No find_package needed
# here at the aggregate operators/ level.

# operators/ is organized per functional group:
#   common/           — shared radar_common library (descriptors, windows, utils)
#   detection/        — detection operators (CFAR)
#   fmcw/             — FMCW signal processing (+ future MIMO marshaling)
#   clustering/       — (future) clustering operators
#   angle_estimation/ — (future) angle estimation operators

# The shared library is not gated: every operator group links against it.
add_subdirectory(common)

# Each operator is gated on its own OP_<name> option, defaulting to ${BUILD_ALL}.
# One add_holohub_operator() call per operator (inside the group's
# CMakeLists.txt for grouped operators).
add_subdirectory(fmcw)

# Detection operators (one add_holohub_operator() per operator inside).
add_subdirectory(detection)

# Clustering operators (one add_holohub_operator() per operator inside).
add_subdirectory(clustering)

# Angle-estimation operators (one add_holohub_operator() per operator inside).
add_subdirectory(angle_estimation)

# Install the aggregate __init__.py alongside each operator's built module.
configure_file(
    ${PROJECT_SOURCE_DIR}/python/holoscan/radar/__init__.py
    ${HOLOSCAN_RADAR_PYTHON_PKG_DIR}/__init__.py COPYONLY)

# Relative DESTINATION — interpreted as <CMAKE_INSTALL_PREFIX>/holoscan/<mod>
# in plain `cmake --install` (set --prefix to ${Python3_SITEARCH}), and as
# <wheel-root>/holoscan/<mod> when scikit-build-core builds the wheel.
install(DIRECTORY ${HOLOSCAN_RADAR_PYTHON_PKG_DIR}
    DESTINATION holoscan)
