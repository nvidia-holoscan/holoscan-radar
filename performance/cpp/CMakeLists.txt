# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared typed sources, YAML selection, Graph timing, and reporting live in
# utils/. Per-operator adapters construct valid tensors, parameters, and ports.
add_library(radar_perf_utils INTERFACE)
target_include_directories(radar_perf_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/operators)
target_link_libraries(radar_perf_utils INTERFACE
    holoscan::core
    holoscan::radar_common
    holoscan::radar_schemas
    CUDA::cudart
    yaml-cpp)
target_compile_definitions(radar_perf_utils INTERFACE
    HOLOSCAN_RADAR_PERF_PROFILE_DIR="${PROJECT_SOURCE_DIR}/performance/profiles")

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

foreach(_perf_name IN LISTS HOLOSCAN_RADAR_PERF_TARGETS)
    radar_add_perf(${_perf_name} holoscan::${_perf_name}_op)
endforeach()

# Minimal smoke for every operator. Timing values remain informational.
if(HOLOSCAN_RADAR_BUILD_TESTING)
    foreach(_perf_name IN LISTS HOLOSCAN_RADAR_PERF_TARGETS)
        if(TARGET perf_${_perf_name})
            add_test(NAME perf_${_perf_name}_smoke
                COMMAND perf_${_perf_name} --scope minimal --sizes small
                    --warmup 1 --frames 3 --repeats 1 --quiet
                    --output ${CMAKE_CURRENT_BINARY_DIR}/perf_${_perf_name}_smoke.json)
            set_tests_properties(perf_${_perf_name}_smoke PROPERTIES LABELS "performance")
        endif()
    endforeach()
    # Representative live paths exercise POD, host Tensor, descriptor, and
    # three-input sources. Reports are checked for completeness by perf_main.
    foreach(_perf_name IN ITEMS
            clutter_filter cfar_detector beamscan_estimator detection_reduction)
        if(TARGET perf_${_perf_name})
            add_test(NAME perf_${_perf_name}_overrides_smoke
                COMMAND perf_${_perf_name} --scope overrides --sizes small
                    --warmup 1 --frames 3 --repeats 2 --quiet
                    --output ${CMAKE_CURRENT_BINARY_DIR}/perf_${_perf_name}_overrides_smoke.json)
            set_tests_properties(perf_${_perf_name}_overrides_smoke PROPERTIES LABELS "performance")
        endif()
    endforeach()
    # Exercise recipe dispatch across scalar, host-vector, descriptor, and
    # aligned multi-input payloads beyond the initial focused smoke group.
    foreach(_perf_name IN ITEMS
            phase_shift_beamformer mvdr_estimator music_estimator
            root_music_estimator esprit_estimator cluster_dbscan
            subarray_selection snapshot_gather tdm_sparse_compensation
            ddm_sparse_demultiplex virtual_array_assembly doppler_notch
            tdm_dense_compensation fmcw_range_doppler_processor)
        if(TARGET perf_${_perf_name})
            add_test(NAME perf_${_perf_name}_recipe_overrides_smoke
                COMMAND perf_${_perf_name} --scope overrides --sizes small
                    --warmup 1 --frames 3 --repeats 2 --quiet
                    --output ${CMAKE_CURRENT_BINARY_DIR}/perf_${_perf_name}_recipe_overrides_smoke.json)
            set_tests_properties(perf_${_perf_name}_recipe_overrides_smoke
                PROPERTIES LABELS "performance")
        endif()
    endforeach()
    foreach(_perf_name IN ITEMS fmcw_range_processor fmcw_doppler_processor)
        if(TARGET perf_${_perf_name})
            add_test(NAME perf_${_perf_name}_cached_plan_smoke
                COMMAND perf_${_perf_name} --scope thorough --sizes medium
                    --warmup 1 --frames 3 --repeats 2 --quiet
                    --output ${CMAKE_CURRENT_BINARY_DIR}/perf_${_perf_name}_cached_plan_smoke.json)
            set_tests_properties(perf_${_perf_name}_cached_plan_smoke PROPERTIES LABELS "performance")
        endif()
    endforeach()
endif()
