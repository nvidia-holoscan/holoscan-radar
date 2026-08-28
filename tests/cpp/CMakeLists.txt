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

# The SAR fixtures are generated, never committed
# (docs/planning/SAR/INPUT_DATA.md -> Policy), so generating them is a CTest
# setup fixture rather than a build step: a missing Python dependency then
# fails one visible test instead of breaking the build for everyone, and the
# tests that need the file are held back rather than silently skipped.
#
# One recipe per storage format the reader decodes. The integer formats are
# the same collection as mode01_smoke in a different container, so a test can
# difference them; the real CI4 collections are Capella's and are gigabytes.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(SAR_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/sar_fixtures)
set(SAR_FIXTURE_RECIPES mode01_smoke mode01_ci4 mode01_ci2)
foreach(_recipe IN LISTS SAR_FIXTURE_RECIPES)
    add_test(NAME sar_generate_${_recipe}
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_SOURCE_DIR}/tools/sar_sim
            ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tools/sar_sim/generate_cphd.py
            ${PROJECT_SOURCE_DIR}/tools/sar_sim/recipes/${_recipe}.yaml
            --output-dir ${SAR_FIXTURE_DIR})
    set_tests_properties(sar_generate_${_recipe} PROPERTIES
        FIXTURES_SETUP sar_fixtures
        LABELS "unit")
endforeach()

radar_add_test(common/test_window)
radar_add_test(common/test_fft_size)
radar_add_test(common/test_steering_weights_device
    holoscan::radar_common_steering)
radar_add_test(common/test_spatial_smoothing)
radar_add_test(common/test_subarray_selection)
radar_add_test(common/test_angle_convention)
radar_add_test(common/test_descriptors)
radar_add_test(common/test_sar_descriptors)
radar_add_test(common/test_sample_lattice)
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
radar_add_test(fmcw/test_mimo_deinterleave
    holoscan::mimo_deinterleave_op)
radar_add_test(fmcw/test_virtual_array_assembly
    holoscan::virtual_array_assembly_op)
radar_add_test(fmcw/test_mimo_coherent_compensation
    holoscan::mimo_coherent_compensation_op)
radar_add_test(sar/test_sar_range_compression
    holoscan::sar_range_compression_op)
radar_add_test(sar/test_backprojection
    holoscan::backprojection_op)
radar_add_test(sar/test_aperture_manager
    holoscan::aperture_manager_op)
# Declared explicitly rather than via radar_add_test: it needs the generated
# fixture's location as a compile definition and the CTest fixture dependency
# that produces it.
add_executable(test_cphd_reader io/test_cphd_reader.cpp)
target_link_libraries(test_cphd_reader PRIVATE
    radar_test_utils
    GTest::gtest_main
    holoscan::cphd_reader_op)
target_compile_definitions(test_cphd_reader PRIVATE
    SAR_FIXTURE_DIR="${SAR_FIXTURE_DIR}")
gtest_discover_tests(test_cphd_reader PROPERTIES
    LABELS "unit"
    FIXTURES_REQUIRED sar_fixtures)
# Also declared explicitly: it writes artifacts into a known directory that a
# second test loads with NumPy, which is the only independent check available
# on a format we write ourselves.
set(TENSOR_DUMP_ARTIFACT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensor_dump_artifacts)
add_executable(test_tensor_dump io/test_tensor_dump.cpp)
target_link_libraries(test_tensor_dump PRIVATE
    radar_test_utils
    GTest::gtest_main
    holoscan::tensor_dump_op)
target_compile_definitions(test_tensor_dump PRIVATE
    TENSOR_DUMP_ARTIFACT_DIR="${TENSOR_DUMP_ARTIFACT_DIR}")
gtest_discover_tests(test_tensor_dump PROPERTIES
    LABELS "unit"
    FIXTURES_SETUP tensor_dump_artifacts)
add_test(NAME test_tensor_dump_npy_readback
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/io/npy_readback.py
        ${TENSOR_DUMP_ARTIFACT_DIR})
set_tests_properties(test_tensor_dump_npy_readback PROPERTIES
    LABELS "unit"
    FIXTURES_REQUIRED tensor_dump_artifacts)
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
radar_add_test(clustering/test_cluster_dbscan
    holoscan::cluster_dbscan_op)
radar_add_test(angle_estimation/test_subarray_selection_op
    holoscan::subarray_selection_op)
radar_add_test(angle_estimation/test_snapshot_gather
    holoscan::snapshot_gather_op)
radar_add_test(angle_estimation/test_beamscan_estimator
    holoscan::beamscan_estimator_op)
radar_add_test(angle_estimation/test_mvdr_estimator
    holoscan::mvdr_estimator_op)
radar_add_test(angle_estimation/test_phase_shift_beamformer
    holoscan::phase_shift_beamformer_op)
radar_add_statistical_test(detection/test_cfar_detector_statistical
    holoscan::cfar_detector_op)
