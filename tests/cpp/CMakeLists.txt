# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(GTest REQUIRED)
include(GoogleTest)

# Shared test utilities: forward bin model, tensor fixtures, and the
# single-operator harness (tests/cpp/utils). The analytic point-target
# simulator now lives in radar_common/sim (reached via holoscan::radar_common).
add_library(radar_test_utils INTERFACE)
target_include_directories(radar_test_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radar_test_utils INTERFACE
    holoscan::core
    holoscan::radar_common
    CUDA::cudart
    GTest::gtest)

# One executable per operator (plus the shared-library suite) so the
# parallel implementation phase owns disjoint test files. NAME is a
# family-relative source path without the extension (e.g. detection/test_cfar);
# the target name is the flat basename so `ctest -R <name>` stays stable
# regardless of where the source lives in the family tree.
function(radar_add_test NAME)
    get_filename_component(_tgt ${NAME} NAME_WE)
    add_executable(${_tgt} ${NAME}.cpp)
    target_link_libraries(${_tgt} PRIVATE
        radar_test_utils
        GTest::gtest_main
        ${ARGN})
    gtest_discover_tests(${_tgt} PROPERTIES LABELS "unit")
endfunction()

function(radar_add_integration_test NAME)
    get_filename_component(_tgt ${NAME} NAME_WE)
    add_executable(${_tgt} ${NAME}.cpp)
    target_link_libraries(${_tgt} PRIVATE
        radar_test_utils
        GTest::gtest_main
        ${ARGN})
    gtest_discover_tests(${_tgt} PROPERTIES LABELS "integration")
endfunction()

function(radar_add_statistical_test NAME)
    get_filename_component(_tgt ${NAME} NAME_WE)
    add_executable(${_tgt} ${NAME}.cpp)
    target_link_libraries(${_tgt} PRIVATE
        radar_test_utils
        GTest::gtest_main
        ${ARGN})
    gtest_discover_tests(${_tgt} PROPERTIES LABELS "statistical")
endfunction()

radar_add_test(common/test_window)
radar_add_test(common/test_descriptors)
radar_add_test(common/test_sdk)
radar_add_test(common/test_sim)
radar_add_test(common/test_tensor)
radar_add_test(fmcw/test_fmcw_range_processor
    holoscan::fmcw_range_processor_op)
radar_add_test(fmcw/test_clutter_filter
    holoscan::clutter_filter_op)
radar_add_test(fmcw/test_fmcw_doppler_processor
    holoscan::fmcw_doppler_processor_op)
radar_add_test(fmcw/test_fmcw_range_doppler_processor
    holoscan::fmcw_range_doppler_processor_op
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op)
# Cross-family end-to-end integration test lives at the tests/cpp root.
radar_add_integration_test(test_pipeline
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::fmcw_range_doppler_processor_op)
radar_add_test(detection/test_cfar_detector
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_go
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_so
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_os
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_alpha_lut
    holoscan::cfar_detector_op)
# DetectionReduction: host tests + CUDA TU for compile-time ScoreFn path.
add_executable(test_detection_reduction
    detection/test_detection_reduction.cpp
    detection/test_detection_reduction_custom.cu)
target_link_libraries(test_detection_reduction PRIVATE
    radar_test_utils
    GTest::gtest_main
    holoscan::detection_reduction_op)
gtest_discover_tests(test_detection_reduction PROPERTIES LABELS "unit")
set_source_files_properties(detection/test_detection_reduction_custom.cu
    PROPERTIES LANGUAGE CUDA)
radar_add_test(detection/test_feature_assembly
    holoscan::feature_assembly_op)
radar_add_test(test_cluster_dbscan
    holoscan::cluster_dbscan_op)
radar_add_statistical_test(detection/test_cfar_detector_statistical
    holoscan::cfar_detector_op)
