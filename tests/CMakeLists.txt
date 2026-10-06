# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(cpp)

find_program(BASH_EXECUTABLE bash REQUIRED)
add_test(
    NAME holohub_wrapper
    COMMAND "${BASH_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_holohub_wrapper.sh"
        "${PROJECT_SOURCE_DIR}/holohub"
)
set_tests_properties(holohub_wrapper PROPERTIES LABELS "unit")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
    NAME pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python/ -v --tb=short
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# Path wiring: HOLOSCAN_RADAR_BUILD_DIR lets conftest.py prepend
# ${CMAKE_BINARY_DIR}/python/lib/holoscan onto holoscan.__path__ after the
# SDK package is imported. Do NOT put HOLOSCAN_RADAR_PYTHON_ROOT on
# PYTHONPATH — a partial holoscan/ tree there would shadow the SDK package.
#
# No SKIP_RETURN_CODE: pytest exits 5 when it collects zero items, which is
# what an unimportable ``holoscan`` looks like. Treating that as a skip made
# CTest report success for a run that tested nothing — a broken Python
# staging step went green. A CUDA-capable GPU and an importable SDK are hard
# dependencies for this module (see the "GPU as a hard dependency" settled
# decision), so exit 5 is a genuine failure. Build and test in the container.
#
# This is a DELIBERATE DEVIATION from the standard Holoscan module
# scaffolding, which ships SKIP_RETURN_CODE 5 so the same test invocation is
# valid on both GPU and CPU-only hosts. That is right for a template targeting
# GPU-optional modules and wrong for this one. If that guidance appears to
# contradict this file, this comment is the answer.
#
# Default pytest.ini excludes ``statistical`` (Monte Carlo); mirror C++
# ``ctest -L statistical`` with a dedicated target.
set_tests_properties(pytest PROPERTIES
    ENVIRONMENT "HOLOSCAN_RADAR_BUILD_DIR=${CMAKE_BINARY_DIR}"
    LABELS "python;unit")

add_test(
    NAME pytest_statistical
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python/
            -v --tb=short -m statistical
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(pytest_statistical PROPERTIES
    ENVIRONMENT "HOLOSCAN_RADAR_BUILD_DIR=${CMAKE_BINARY_DIR}"
    LABELS "python;statistical")
