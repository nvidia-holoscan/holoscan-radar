# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Python3 + pybind11 are pulled in (find_package + FetchContent) by
# pybind11_add_holohub_module, which the operator's own python/CMakeLists.txt
# includes when cpp == 'cpp'. No find_package needed
# here at the aggregate operators/ level.

# operators/ is organized per functional group:
#   common/           — shared radar_common library (descriptors, windows, utils)
#   detection/        — (future) detection operators
#   clustering/       — (future) clustering operators
#   angle_estimation/ — (future) angle estimation operators

# The shared library is not gated: every operator group links against it.
add_subdirectory(common)

# Template placeholder — kept until the first real operators land with
# tests, then removed.
add_holohub_operator(radar_op)

# Install the aggregate __init__.py alongside each operator's built module.
configure_file(
    ${PROJECT_SOURCE_DIR}/python/holoscan/radar/__init__.py
    ${HOLOSCAN_RADAR_PYTHON_PKG_DIR}/__init__.py COPYONLY)

# Relative DESTINATION — interpreted as <CMAKE_INSTALL_PREFIX>/holoscan/<mod>
# in plain `cmake --install` (set --prefix to ${Python3_SITEARCH}), and as
# <wheel-root>/holoscan/<mod> when scikit-build-core builds the wheel.
install(DIRECTORY ${HOLOSCAN_RADAR_PYTHON_PKG_DIR}
    DESTINATION holoscan)
