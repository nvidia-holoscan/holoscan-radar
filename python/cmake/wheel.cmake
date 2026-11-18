# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Declare the holoscan-wheel-radar install component: the operators-only
# holoscan_radar package, its extension, and the native libraries it owns.
# scikit-build-core installs only this component (pyproject install.components),
# so headers, CMake package files, and app support never reach the wheel.

if(NOT COMMAND holoscan_add_python_wheel_component)
    if(SKBUILD)
        message(FATAL_ERROR "Radar wheels require an SDK exporting HoloscanPythonWheel.cmake")
    endif()
    return()
endif()
if(SKBUILD AND HOLOSCAN_RADAR_BUILD_PYTHON_APPS)
    message(FATAL_ERROR "Radar wheels contain operators only; set HOLOSCAN_RADAR_BUILD_PYTHON_APPS=OFF")
endif()

# The helper does not follow link dependencies, so list every owned runtime:
# the operators the provider registers, the radar_common libraries they link,
# and tinyxml2 for CPHDReaderOp. Core comes from holoscan-core-cu13; cuFFT,
# cuBLAS, and cuSOLVER come from their nvidia-* wheels under nvidia/cu13/lib.
set(_radar_wheel_native_targets
    ${_radar_python_operators}
    radar_common
    radar_common_mimo_compensation
    radar_common_reduce
    radar_common_source_count
    radar_common_spatial_smoothing
    radar_common_steering)
if("cphd_reader_op" IN_LIST _radar_python_operators)
    # Imported targets are directory-scoped; import tinyxml2 here as well.
    find_package(tinyxml2 REQUIRED CONFIG)
    list(APPEND _radar_wheel_native_targets tinyxml2::tinyxml2)
endif()

holoscan_add_python_wheel_component(
    MODULE radar
    PACKAGE holoscan_radar
    PYTHON_DIRECTORY "${HOLOSCAN_RADAR_PYTHON_ROOT}/holoscan_radar"
    PYTHON_FILES __init__.py __init__.pyi py.typed
    EXTENSION_TARGET holoscan_radar_python
    NATIVE_TARGETS ${_radar_wheel_native_targets}
    LIBRARY_RUNPATH "$ORIGIN/../../holoscan/lib" "$ORIGIN/../../nvidia/cu13/lib"
    EXTENSION_RUNPATH "$ORIGIN/../holoscan/lib" "$ORIGIN/../nvidia/cu13/lib")
