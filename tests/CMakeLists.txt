# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
add_test(NAME ApplicationBuildScope
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${PROJECT_SOURCE_DIR}/tests/scripts -p test_build_scope.py)
set_tests_properties(ApplicationBuildScope PROPERTIES LABELS "build;unit" TIMEOUT 60)
add_test(NAME BlossomCheckoutTrust
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${PROJECT_SOURCE_DIR}/tests/scripts -p test_blossom_trust.py)
set_tests_properties(BlossomCheckoutTrust PROPERTIES LABELS "build;unit" TIMEOUT 60)
add_test(NAME DevcontainerX11Authorization
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${PROJECT_SOURCE_DIR}/tests/scripts -p test_devcontainer_x11.py)
set_tests_properties(DevcontainerX11Authorization PROPERTIES LABELS "build;unit" TIMEOUT 60)

add_subdirectory(cpp)

if(HOLOSCAN_RADAR_TEST_PACKAGING)
    add_subdirectory(packaging)
endif()

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

if(HOLOSCAN_RADAR_TEST_APP_BUILDS)
    foreach(_case IN ITEMS ApplicationConfigureMatrix ApplicationCliSisoSmoke)
        add_test(NAME ${_case}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_app_builds.py
                ${_case} --source ${PROJECT_SOURCE_DIR}
                --work ${CMAKE_CURRENT_BINARY_DIR}/app-build
                --log-dir ${PROJECT_BINARY_DIR}/Testing/Temporary)
        set_tests_properties(${_case} PROPERTIES LABELS "app-build")
    endforeach()
    set_tests_properties(ApplicationConfigureMatrix PROPERTIES TIMEOUT 300)
    set_tests_properties(ApplicationCliSisoSmoke PROPERTIES RUN_SERIAL TRUE TIMEOUT 1800)
endif()

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
add_test(NAME pytest_provider
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${PROJECT_SOURCE_DIR}/tests/provider/)
set_tests_properties(pytest_provider PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:${HOLOSCAN_RADAR_SDK_PYTHON_SITE}:$ENV{PYTHONPATH};HOLOSCAN_RADAR_TEST_BUILD_DIR=${PROJECT_BINARY_DIR}"
    FIXTURES_REQUIRED sar_cphd
    LABELS "python;unit")
