# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reusable harness headers (utils/) plus thin per-operator consumers. Shared
# code lives in utils/ as an INTERFACE library; only the scenario matrix is
# operator-specific.
add_library(radar_perf_utils INTERFACE)
target_include_directories(radar_perf_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/operators)
target_link_libraries(radar_perf_utils INTERFACE
    holoscan::core
    holoscan::radar_common
    holoscan::radar_schemas
    CUDA::cudart)

# Public performance targets, also used to register smoke tests.
set(HOLOSCAN_RADAR_PERF_TARGETS
    clutter_filter
    mimo_deinterleave
    phase_shift_beamformer
    tdm_dense_compensation
    ddm_dense_demultiplex
    doppler_notch
    ddm_power_fold
    ddm_sparse_demultiplex
    tdm_sparse_compensation
    virtual_array_assembly
    fmcw_range_processor
    fmcw_doppler_processor
    fmcw_range_doppler_processor
    fmcw_split_vs_fused
    cluster_dbscan
    beamscan_estimator
    mvdr_estimator
    music_estimator
    root_music_estimator
    esprit_estimator
    cfar_detector
    snapshot_gather
    subarray_selection
    detection_reduction
    feature_assembly
    rank_one_residual_screen
    multi_source_expansion
)

function(radar_add_perf NAME)
    set(_op_target ${ARGN})
    if(NOT TARGET ${_op_target})
        message(STATUS
            "Skipping perf_${NAME} (missing ${_op_target})")
        return()
    endif()
    add_executable(perf_${NAME} perf_${NAME}.cpp)
    target_link_libraries(perf_${NAME} PRIVATE
        radar_perf_utils
        ${_op_target})
endfunction()

radar_add_perf(cfar_detector holoscan::cfar_detector_op)
radar_add_perf(detection_reduction holoscan::detection_reduction_op)
radar_add_perf(feature_assembly holoscan::feature_assembly_op)
radar_add_perf(rank_one_residual_screen
    holoscan::rank_one_residual_screen_op)
radar_add_perf(multi_source_expansion
    holoscan::multi_source_expansion_op)
radar_add_perf(fmcw_range_processor holoscan::fmcw_range_processor_op)
radar_add_perf(fmcw_doppler_processor holoscan::fmcw_doppler_processor_op)
radar_add_perf(clutter_filter holoscan::clutter_filter_op)
radar_add_perf(fmcw_range_doppler_processor
    holoscan::fmcw_range_doppler_processor_op)

# Multi-op fused/split harness needs every constituent operator target.
if(TARGET holoscan::fmcw_range_processor_op
   AND TARGET holoscan::clutter_filter_op
   AND TARGET holoscan::fmcw_doppler_processor_op
   AND TARGET holoscan::fmcw_range_doppler_processor_op)
    add_executable(perf_fmcw_split_vs_fused perf_fmcw_split_vs_fused.cpp)
    target_link_libraries(perf_fmcw_split_vs_fused PRIVATE
        radar_perf_utils
        holoscan::fmcw_range_processor_op
        holoscan::clutter_filter_op
        holoscan::fmcw_doppler_processor_op
        holoscan::fmcw_range_doppler_processor_op)
else()
    message(STATUS
        "Skipping perf_fmcw_split_vs_fused (missing operator targets)")
endif()

radar_add_perf(cluster_dbscan holoscan::cluster_dbscan_op)
radar_add_perf(snapshot_gather holoscan::snapshot_gather_op)
radar_add_perf(subarray_selection holoscan::subarray_selection_op)
radar_add_perf(beamscan_estimator holoscan::beamscan_estimator_op)
radar_add_perf(mvdr_estimator holoscan::mvdr_estimator_op)
radar_add_perf(music_estimator holoscan::music_estimator_op)
radar_add_perf(root_music_estimator holoscan::root_music_estimator_op)
radar_add_perf(esprit_estimator holoscan::esprit_estimator_op)
radar_add_perf(phase_shift_beamformer holoscan::phase_shift_beamformer_op)
radar_add_perf(mimo_deinterleave holoscan::mimo_deinterleave_op)
radar_add_perf(virtual_array_assembly holoscan::virtual_array_assembly_op)
radar_add_perf(tdm_dense_compensation
    holoscan::tdm_dense_compensation_op)
radar_add_perf(ddm_dense_demultiplex
    holoscan::ddm_dense_demultiplex_op)
radar_add_perf(doppler_notch holoscan::doppler_notch_op)
radar_add_perf(ddm_power_fold
    holoscan::ddm_power_fold_op)
radar_add_perf(ddm_sparse_demultiplex
    holoscan::ddm_sparse_demultiplex_op)
radar_add_perf(tdm_sparse_compensation
    holoscan::tdm_sparse_compensation_op)

# Optional smoke: runs a tiny scenario subset and verifies a report file is
# written. No timing thresholds — CI only checks the tool executes and emits.
if(HOLOSCAN_RADAR_BUILD_TESTING)
    function(radar_add_perf_smoke NAME)
        if(NOT TARGET perf_${NAME})
            return()
        endif()
        add_test(NAME perf_${NAME}_smoke
            COMMAND perf_${NAME} --sizes xs --warmup 1 --frames 2 --quiet
                --output ${CMAKE_CURRENT_BINARY_DIR}/perf_${NAME}_smoke.json)
        set_tests_properties(perf_${NAME}_smoke PROPERTIES
            LABELS "performance")
    endfunction()

    foreach(_perf_name IN LISTS HOLOSCAN_RADAR_PERF_TARGETS)
        radar_add_perf_smoke(${_perf_name})
    endforeach()
endif()
