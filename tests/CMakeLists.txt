# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(cpp)

# The provider suite checks Python bindings and Graph execution. Numerical
# Monte Carlo CFAR coverage remains in the C++ statistical CTest tier; the
# removed Python statistical suite is not part of tests/provider. Registering
# pytest without bindings would fail on import.
if(NOT HOLOSCAN_RADAR_BUILD_PYTHON)
    message(STATUS
        "Skipping pytest provider suite "
        "(HOLOSCAN_RADAR_BUILD_PYTHON=OFF)")
    return()
endif()

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
add_test(NAME pytest_provider
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${PROJECT_SOURCE_DIR}/tests/provider/)
set_tests_properties(pytest_provider PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:${HOLOSCAN_RADAR_SDK_PYTHON_SITE}:$ENV{PYTHONPATH}"
    LABELS "python;unit")
