# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
    NAME pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python/ -v --tb=short
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# Path wiring: HOLOSCAN_RADAR_BUILD_DIR lets conftest.py prepend
# ${CMAKE_BINARY_DIR}/python/lib/holoscan onto holoscan.__path__ after the
# SDK package is imported. Do NOT put HOLOSCAN_RADAR_PYTHON_ROOT on
# PYTHONPATH — a partial holoscan/ tree there would shadow the SDK package.
# SKIP_RETURN_CODE 5 is the safety net when holoscan isn't importable
# (pytest collects zero items / importorskip, exits 5).
# Default pytest.ini excludes ``statistical`` (Monte Carlo); mirror C++
# ``ctest -L statistical`` with a dedicated target.
set_tests_properties(pytest PROPERTIES
    ENVIRONMENT "HOLOSCAN_RADAR_BUILD_DIR=${CMAKE_BINARY_DIR}"
    SKIP_RETURN_CODE 5
    LABELS "python;unit")

add_test(
    NAME pytest_statistical
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python/
            -v --tb=short -m statistical
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(pytest_statistical PROPERTIES
    ENVIRONMENT "HOLOSCAN_RADAR_BUILD_DIR=${CMAKE_BINARY_DIR}"
    SKIP_RETURN_CODE 5
    LABELS "python;statistical")
