# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Optional EXT overrides the source extension (default .cpp); use .cu when
# the suite includes device stage kernels (e.g. FeatureAssembly).
function(radar_add_test NAME)
    cmake_parse_arguments(_RAT "" "EXT" "" ${ARGN})
    if(NOT _RAT_EXT)
        set(_RAT_EXT ".cpp")
    endif()
    get_filename_component(_tgt ${NAME} NAME_WE)
    add_executable(${_tgt} ${NAME}${_RAT_EXT})
    target_link_libraries(${_tgt} PRIVATE
        radar_test_utils
        GTest::gtest_main
        ${_RAT_UNPARSED_ARGUMENTS})
    gtest_discover_tests(${_tgt} PROPERTIES LABELS "unit")
endfunction()

# [EA HSDK5 ONLY] Only register tests for radar_common helpers that do not
# depend on the removed 4.x harness / unported operators, plus ported ops.
# Expand as each operator lands.
message(STATUS
    "[EA HSDK5 ONLY] C++ tests limited to common helpers + ported operators")

radar_add_test(common/test_window)
radar_add_test(common/test_steering_weights_device
    holoscan::radar_common_steering)
radar_add_test(common/test_spatial_smoothing)
radar_add_test(common/test_subarray_selection)
radar_add_test(common/test_source_count)
radar_add_test(common/test_angle_convention)
radar_add_test(common/test_descriptors)
# yaml_converters.hpp is header-only; HSDK 5 no longer re-exports yaml-cpp.
find_package(yaml-cpp REQUIRED)
radar_add_test(common/test_sdk yaml-cpp)
radar_add_test(common/test_sim)
radar_add_test(common/test_sim_gpu holoscan::radar_common_simulator)
radar_add_test(common/test_tensor)
radar_add_test(fmcw/test_clutter_filter
    holoscan::clutter_filter_op)
radar_add_test(angle_estimation/test_snapshot_gather
    holoscan::snapshot_gather_op)
radar_add_test(angle_estimation/test_beamscan_estimator
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering)
radar_add_test(angle_estimation/test_phase_shift_beamformer
    holoscan::phase_shift_beamformer_op
    holoscan::radar_common_steering)
radar_add_test(angle_estimation/test_subarray_selection_op
    holoscan::subarray_selection_op
    holoscan::radar_common_spatial_smoothing)
radar_add_test(angle_estimation/test_mvdr_estimator
    holoscan::mvdr_estimator_op
    holoscan::radar_common_steering
    holoscan::radar_common_spatial_smoothing)
radar_add_test(angle_estimation/test_root_music_estimator
    holoscan::root_music_estimator_op
    holoscan::radar_common_spatial_smoothing
    holoscan::radar_common_source_count)
radar_add_test(angle_estimation/test_esprit_estimator
    holoscan::esprit_estimator_op
    holoscan::radar_common_spatial_smoothing
    holoscan::radar_common_source_count)
radar_add_test(angle_estimation/test_music_estimator
    holoscan::music_estimator_op
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering
    holoscan::radar_common_spatial_smoothing
    holoscan::radar_common_source_count)
radar_add_test(clustering/test_cluster_dbscan
    holoscan::cluster_dbscan_op)
radar_add_test(detection/test_detection_reduction
    holoscan::detection_reduction_op)
radar_add_test(detection/test_feature_assembly
    EXT .cu
    holoscan::feature_assembly_op)
radar_add_test(detection/test_cfar_detector
    holoscan::cfar_detector_op)
