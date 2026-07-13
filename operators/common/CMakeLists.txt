# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared library for the supporting data structures and helpers used across
# operator groups (see README.md in this directory). Host-side C++ only: MatX-facing helpers are header-only
# (radar_common/tensor/matx_utils.hpp) and are compiled into each operator's .cu
# translation units, so this target does not link MatX or CUDA.

add_library(radar_common STATIC
    radar_common/tensor/tensor_utils.cpp
    radar_common/dsp/window.cpp
)
add_library(holoscan::radar_common ALIAS radar_common)

# Headers install under ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar (see the
# install(DIRECTORY) below), and code includes them as "radar_common/...",
# so that directory is the exported include root.
include(GNUInstallDirs)
target_include_directories(radar_common
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators/common>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar>
)

target_link_libraries(radar_common
    PUBLIC
        holoscan::core
)

target_compile_features(radar_common PUBLIC cxx_std_17)
set_property(TARGET radar_common PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS radar_common
    EXPORT holoscan_radar_common_targets
)

install(EXPORT holoscan_radar_common_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

install(DIRECTORY radar_common
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar
    FILES_MATCHING PATTERN "*.hpp"
)
