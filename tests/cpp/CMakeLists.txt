# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(GTest REQUIRED)
include(GoogleTest)

# Shared test utilities: analytic point-target simulator, forward bin
# model, tensor fixtures, and the single-operator harness (tests/cpp/utils).
add_library(radar_test_utils INTERFACE)
target_include_directories(radar_test_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radar_test_utils INTERFACE
    holoscan::core
    holoscan::radar_common
    CUDA::cudart
    GTest::gtest)

# One executable per operator (plus the shared-library suite) so the
# parallel implementation phase owns disjoint test files.
function(radar_add_test NAME)
    add_executable(${NAME} ${NAME}.cpp)
    target_link_libraries(${NAME} PRIVATE
        radar_test_utils
        GTest::gtest_main
        ${ARGN})
    gtest_discover_tests(${NAME} PROPERTIES LABELS "unit")
endfunction()

radar_add_test(test_common)
radar_add_test(test_fmcw_range_processor
    holoscan::fmcw_range_processor_op)
radar_add_test(test_clutter_filter
    holoscan::clutter_filter_op)
radar_add_test(test_fmcw_doppler_processor
    holoscan::fmcw_doppler_processor_op)
radar_add_test(test_fmcw_range_doppler_processor
    holoscan::fmcw_range_doppler_processor_op
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op)

# radar_op template placeholder test — removed together with radar_op once
# the first real operator lands (docs/planning/STATUS.md).
add_executable(test_operators test_operators.cpp)
target_link_libraries(test_operators PRIVATE
    holoscan::radar_op
    holoscan::core
    GTest::gtest_main)
target_include_directories(test_operators PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)
gtest_discover_tests(test_operators PROPERTIES LABELS "unit")
