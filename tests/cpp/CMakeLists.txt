# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(GTest REQUIRED)
include(GoogleTest)

# Shared test utilities: forward bin model and radar-specific Tensor fixtures.
# Single-operator tests use the SDK's native holoscan::testing fixture. The
# analytic point-target simulator lives in radar_common/sim (reached via
# holoscan::radar_common).
add_library(radar_test_utils INTERFACE)
target_include_directories(radar_test_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radar_test_utils INTERFACE
    holoscan::core
    holoscan::testing
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
radar_add_test(utils/test_perf_profiles yaml-cpp)
radar_add_test(utils/test_perf_runner)
radar_add_test(utils/test_perf_yaml_standard yaml-cpp holoscan::doppler_notch_op)
target_include_directories(test_perf_yaml_standard PRIVATE
    ${PROJECT_SOURCE_DIR}/performance/harness/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_include_directories(test_perf_runner PRIVATE ${PROJECT_SOURCE_DIR}/performance/harness/cpp)
target_include_directories(test_perf_profiles PRIVATE
    ${PROJECT_SOURCE_DIR}/performance/harness/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_perf_profiles PRIVATE
    HOLOSCAN_RADAR_PERF_OPERATOR_ROOT="${PROJECT_SOURCE_DIR}/performance/operators")
target_compile_definitions(test_perf_yaml_standard PRIVATE
    HOLOSCAN_RADAR_PERF_OPERATOR_ROOT="${PROJECT_SOURCE_DIR}/performance/operators")
radar_add_test(common/test_sim)
radar_add_test(sar/test_phase_correction holoscan::phase_correction_op)
radar_add_test(sar/test_aperture_manager holoscan::aperture_manager_op)
radar_add_test(sar/test_backprojection holoscan::backprojection_op)
radar_add_test(sar/test_sar_range_compression holoscan::sar_range_compression_op)
radar_add_test(io/test_tensor_dump holoscan::tensor_dump_op)
radar_add_test(io/test_cphd_reader holoscan::cphd_reader_op)
add_executable(generate_cphd_fixture io/generate_cphd_fixture.cpp)
target_include_directories(generate_cphd_fixture PRIVATE ${PROJECT_SOURCE_DIR}/tests/cpp)
radar_add_test(common/test_sim_gpu holoscan::radar_common_simulator)
radar_add_test(common/test_tensor)
radar_add_test(common/test_virtual_array)
radar_add_test(utils/test_native_operator_testing
    holoscan::clutter_filter_op)
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
    holoscan::cfar_detector_op
    holoscan::fmcw_range_doppler_processor_op
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op)
radar_add_test(angle_estimation/test_snapshot_gather
    holoscan::snapshot_gather_op)
radar_add_test(angle_estimation/test_beamscan_estimator
    holoscan::beamscan_estimator_op
    holoscan::radar_common_steering)
radar_add_test(angle_estimation/test_rank_one_residual_screen
    EXT .cu
    holoscan::rank_one_residual_screen_op)
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
radar_add_test(detection/test_multi_source_expansion
    EXT .cu
    holoscan::multi_source_expansion_op)
radar_add_test(integration/test_selective_aoa_operators
    holoscan::rank_one_residual_screen_op
    holoscan::multi_source_expansion_op)
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

radar_add_test(app/test_fmcw_siso_config
    yaml-cpp
    holoscan::params_io
    holoscan::radar_app
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op
    holoscan::cluster_dbscan_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op)
target_include_directories(test_fmcw_siso_config PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/fmcw/siso/cpp
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_fmcw_siso_config PRIVATE
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

radar_add_test(app/test_adaptive_aoa_config
    yaml-cpp
    holoscan::params_io
    holoscan::radar_app
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::mimo_deinterleave_op
    holoscan::tdm_sparse_compensation_op
    holoscan::cfar_detector_op
    holoscan::beamscan_estimator_op
    holoscan::rank_one_residual_screen_op
    holoscan::music_estimator_op
    holoscan::root_music_estimator_op
    holoscan::esprit_estimator_op
    holoscan::mvdr_estimator_op
    holoscan::feature_assembly_op
    holoscan::multi_source_expansion_op)
target_include_directories(test_adaptive_aoa_config PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/applications/fmcw/adaptive_aoa/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_adaptive_aoa_config PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

if(TARGET adaptive_aoa)
    add_test(NAME AdaptiveAoASmoke
        COMMAND $<TARGET_FILE:adaptive_aoa>
            ${PROJECT_SOURCE_DIR}/applications/fmcw/adaptive_aoa/config/adaptive_aoa_smoke.yaml)
    set_tests_properties(AdaptiveAoASmoke
        PROPERTIES LABELS "application;smoke")
endif()

radar_add_test(app/test_multi_profile_source
    yaml-cpp
    holoscan::params_io
    holoscan::radar_app
    holoscan::radar_common_simulator
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::mimo_deinterleave_op
    holoscan::tdm_sparse_compensation_op
    holoscan::cfar_detector_op
    holoscan::beamscan_estimator_op
    holoscan::cluster_dbscan_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op)
target_include_directories(test_multi_profile_source PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/applications/fmcw/multi_waveform/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_compile_definitions(test_multi_profile_source PRIVATE
    HOLOSCAN_RADAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

if(TARGET multi_waveform)
    add_test(NAME MultiWaveformSmoke
        COMMAND $<TARGET_FILE:multi_waveform>
            ${PROJECT_SOURCE_DIR}/applications/fmcw/multi_waveform/config/multi_waveform_smoke.yaml)
    set_tests_properties(MultiWaveformSmoke
        PROPERTIES LABELS "application;smoke")
endif()

radar_add_test(app/test_fmcw_simo_report
    holoscan::radar_app)
target_include_directories(test_fmcw_simo_report PRIVATE
    ${PROJECT_SOURCE_DIR}/applications/common
    ${PROJECT_SOURCE_DIR}/operators)

if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    radar_add_test(app/test_fmcw_dashboard holoscan::fmcw_dashboard)
endif()

set(SAR_SMOKE_DIR "${PROJECT_BINARY_DIR}/tests/cpp")
set(SAR_APP_NAMES spotlight video_sar tiled_strip continuous_strip independent_images)
set(SAR_COUNTS 1 5 5 5 2)
set(SAR_DIMS "5x5" "5x5" "15x6" "15x6" "5x5")
if(TARGET spotlight OR TARGET video_sar OR TARGET tiled_strip OR
        TARGET continuous_strip OR TARGET independent_images)
    add_test(NAME SarFixture
        COMMAND $<TARGET_FILE:generate_cphd_fixture>
            "${SAR_SMOKE_DIR}/generated_sar_smoke.cphd")
    set_tests_properties(SarFixture PROPERTIES FIXTURES_SETUP sar_cphd)
endif()
foreach(index RANGE 0 4)
    list(GET SAR_APP_NAMES ${index} name)
    list(GET SAR_COUNTS ${index} count)
    list(GET SAR_DIMS ${index} dims)
    if(TARGET ${name})
        set(config_dir "${PROJECT_BINARY_DIR}/applications/sar/${name}/cpp")
        foreach(variant smoke viz)
            if(variant STREQUAL "viz")
                set(config "${config_dir}/${name}_viz_generated_smoke.yaml")
            else()
                set(config "${config_dir}/${name}_generated_smoke.yaml")
            endif()
            add_test(NAME Sar_${name}_${variant}
                COMMAND ${CMAKE_COMMAND}
                    -DSAR_EXECUTABLE=$<TARGET_FILE:${name}>
                    -DSAR_CONFIG=${config}
                    -DSAR_COUNT=${count}
                    -DSAR_DIMS=${dims}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_smoke.cmake)
            set_tests_properties(Sar_${name}_${variant} PROPERTIES
                FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
        endforeach()
        math(EXPR next_index "(${index} + 1) % 5")
        list(GET SAR_APP_NAMES ${next_index} wrong_name)
        set(wrong_config
            "${PROJECT_BINARY_DIR}/applications/sar/${wrong_name}/cpp/${wrong_name}_generated_smoke.yaml")
        if(TARGET ${wrong_name})
            add_test(NAME Sar_${name}_rejects_other_topology
                COMMAND ${CMAKE_COMMAND}
                    -DSAR_EXECUTABLE=$<TARGET_FILE:${name}>
                    -DSAR_CONFIG=${wrong_config}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_reject.cmake)
            set_tests_properties(Sar_${name}_rejects_other_topology PROPERTIES
                FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
        endif()
    endif()
endforeach()

if(TARGET spotlight)
    set(spotlight_dir "${PROJECT_BINARY_DIR}/applications/sar/spotlight/cpp")
    add_test(NAME Sar_spotlight_dump
        COMMAND ${CMAKE_COMMAND}
            -DSAR_EXECUTABLE=$<TARGET_FILE:spotlight>
            -DSAR_CONFIG=${spotlight_dir}/spotlight_dump_generated_smoke.yaml
            -DSAR_DUMP_PATH=${spotlight_dir}/spotlight_dump_0.npy
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sar_dump.cmake)
    set_tests_properties(Sar_spotlight_dump PROPERTIES
        FIXTURES_REQUIRED sar_cphd LABELS "application;smoke")
endif()
