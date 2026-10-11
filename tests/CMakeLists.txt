# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(cpp)

# The pytest suites exercise the Python bindings, so they only
# make sense when those are built. Registering them unconditionally made a
# bare `ctest` fail when no operator bindings are available (no operator bindings to import) and forced
# every caller to remember `-E pytest`.
if(NOT HOLOSCAN_RADAR_BUILD_PYTHON)
    message(STATUS
        "Skipping pytest suites "
        "(HOLOSCAN_RADAR_BUILD_PYTHON=OFF)")
    return()
endif()

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
add_test(NAME pytest_provider
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${PROJECT_SOURCE_DIR}/tests/provider/)
set_tests_properties(pytest_provider PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:${HOLOSCAN_RADAR_SDK_PYTHON_SITE}:$ENV{PYTHONPATH}"
    LABELS "python;unit")
