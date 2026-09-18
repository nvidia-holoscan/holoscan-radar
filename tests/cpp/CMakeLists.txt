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
    holoscan::radar_schemas
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
    cmake_parse_arguments(_RAT "" "EXT;LABELS" "" ${ARGN})
    if(NOT _RAT_EXT)
        set(_RAT_EXT ".cpp")
    endif()
    if(NOT _RAT_LABELS)
        set(_RAT_LABELS "unit")
    endif()
    get_filename_component(_tgt ${NAME} NAME_WE)
    add_executable(${_tgt} ${NAME}${_RAT_EXT})
    target_link_libraries(${_tgt} PRIVATE
        radar_test_utils
        GTest::gtest_main
        ${_RAT_UNPARSED_ARGUMENTS})
    gtest_discover_tests(${_tgt} PROPERTIES LABELS "${_RAT_LABELS}")
endfunction()

# [EA HSDK5 ONLY] Only register tests for radar_common helpers that do not
# depend on the removed 4.x harness / unported operators, plus ported ops.
# Expand as each operator lands.
message(STATUS
    "[EA HSDK5 ONLY] C++ tests limited to common helpers + ported operators")

radar_add_test(common/test_window)
radar_add_test(common/test_uniform_grid)
radar_add_test(common/test_steering_weights_device
    holoscan::radar_common_steering)
radar_add_test(common/test_spatial_smoothing)
radar_add_test(common/test_subarray_selection)
radar_add_test(common/test_source_count)
radar_add_test(common/test_angle_convention)
radar_add_test(common/test_descriptors)
radar_add_test(common/test_ddm_code_generators
    holoscan::radar_common_mimo_compensation)
radar_add_test(common/test_live_override_payloads)
radar_add_test(common/test_wire_descriptors holoscan::radar_schemas)
# yaml_converters.hpp / max_tensor_extents_yaml.hpp are header-only; HSDK 5
# no longer re-exports yaml-cpp.
find_package(yaml-cpp REQUIRED)
radar_add_test(common/test_sdk yaml-cpp holoscan::config_io)
radar_add_test(common/test_max_tensor_extents yaml-cpp)
radar_add_test(common/test_sim)
radar_add_test(common/test_sim_gpu holoscan::radar_common_simulator)
radar_add_test(common/test_tensor)
radar_add_test(common/test_virtual_array)
# Holoscan EA reproducer for mid-stream scalar override + device Tensor path
# (HSDK5_PORT.md §1). Not in the default `unit` label — run with -L ea_repro.
radar_add_test(utils/test_live_override_repro LABELS ea_repro)
# Standalone kLocalEnum vs kUInt32 repro (also drop-in for HSDK; no GTest).
add_executable(enum_tensor_bind_repro
    utils/hsdk_enum_tensor_bind_repro/enum_tensor_bind_repro.cpp)
target_link_libraries(enum_tensor_bind_repro PRIVATE
    holoscan::core
    CUDA::cudart)
add_test(NAME EnumTensorBindRepro COMMAND enum_tensor_bind_repro)
set_tests_properties(EnumTensorBindRepro PROPERTIES LABELS ea_repro)

# Plain pageable-host Tensor config probe under OnEach + Exact + SafeDefault.
radar_add_test(utils/test_host_tensor_exact_override LABELS ea_repro)
radar_add_test(fmcw/test_clutter_filter
    holoscan::clutter_filter_op)
radar_add_test(fmcw/test_mimo_deinterleave
    holoscan::mimo_deinterleave_op)
radar_add_test(fmcw/test_tdm_dense_compensation
    holoscan::tdm_dense_compensation_op)
radar_add_test(fmcw/test_ddm_dense_demultiplex
    holoscan::ddm_dense_demultiplex_op
    holoscan::fmcw_doppler_processor_op)
radar_add_test(fmcw/test_doppler_notch
    holoscan::doppler_notch_op)
radar_add_test(fmcw/test_doppler_notch_pipeline
    holoscan::fmcw_range_processor_op
    holoscan::fmcw_doppler_processor_op
    holoscan::ddm_dense_demultiplex_op
    holoscan::doppler_notch_op)
radar_add_test(fmcw/test_ddm_power_fold
    holoscan::ddm_power_fold_op)
radar_add_test(fmcw/test_ddm_sparse_demultiplex
    holoscan::ddm_sparse_demultiplex_op)
radar_add_test(fmcw/test_tdm_sparse_compensation
    holoscan::tdm_sparse_compensation_op)
radar_add_test(fmcw/test_virtual_array_assembly
    holoscan::virtual_array_assembly_op
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering)
radar_add_test(fmcw/test_fmcw_doppler_processor
    holoscan::fmcw_doppler_processor_op)
radar_add_test(fmcw/test_fmcw_range_processor
    holoscan::fmcw_range_processor_op)
radar_add_test(fmcw/test_fmcw_range_doppler_processor
    holoscan::fmcw_range_doppler_processor_op
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op)
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
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering
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
    holoscan::subarray_selection_op
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
radar_add_test(detection/test_detection_reduction_custom
    EXT .cu
    holoscan::detection_reduction_op)
radar_add_test(detection/test_feature_assembly
    EXT .cu
    holoscan::feature_assembly_op)
radar_add_test(detection/test_cfar_detector
    holoscan::cfar_detector_op
    holoscan::feature_assembly_op)
radar_add_test(detection/test_cfar_detector_alpha_lut
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_os
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_so
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_go
    holoscan::cfar_detector_op)
radar_add_test(detection/test_cfar_detector_statistical
    LABELS statistical
    holoscan::cfar_detector_op)

radar_add_test(app/test_fmcw_simo_config
    yaml-cpp
    holoscan::params_io
    holoscan::radar_app
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op
    holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op
    holoscan::phase_shift_beamformer_op
    holoscan::cluster_dbscan_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op)
target_include_directories(test_fmcw_simo_config PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_fmcw_simo_config PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

radar_add_test(app/test_fmcw_mimo_config
    yaml-cpp
    holoscan::params_io
    holoscan::radar_app
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::mimo_deinterleave_op
    holoscan::tdm_sparse_compensation_op
    holoscan::ddm_dense_demultiplex_op
    holoscan::doppler_notch_op
    holoscan::virtual_array_assembly_op
    holoscan::cfar_detector_op
    holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op
    holoscan::cluster_dbscan_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op)
target_include_directories(test_fmcw_mimo_config PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_fmcw_mimo_config PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

radar_add_test(app/test_fmcw_simo_report
    holoscan::radar_app)
target_include_directories(test_fmcw_simo_report PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)

if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    radar_add_test(app/test_fmcw_dashboard holoscan::fmcw_dashboard)
endif()
