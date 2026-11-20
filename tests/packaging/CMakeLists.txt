# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

add_test(NAME Packaging_ci_helpers
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${PROJECT_SOURCE_DIR}/ci/tests -v)
set_tests_properties(Packaging_ci_helpers PROPERTIES
    LABELS "packaging;unit"
    TIMEOUT 120)

set(_radar_package_modes debian)
if(HOLOSCAN_RADAR_BUILD_PYTHON)
    list(APPEND _radar_package_modes wheel)
endif()
foreach(_mode IN LISTS _radar_package_modes)
    add_test(NAME Packaging_${_mode}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_packages.py
            ${_mode}
            --source ${PROJECT_SOURCE_DIR}
            --work ${CMAKE_CURRENT_BINARY_DIR}/${_mode}
            --sdk-dir ${holoscan_DIR}
            --architectures "${CMAKE_CUDA_ARCHITECTURES}")
    set_tests_properties(Packaging_${_mode} PROPERTIES
        LABELS "packaging;integration"
        RUN_SERIAL TRUE
        TIMEOUT 1800)
endforeach()
unset(_radar_package_modes)
