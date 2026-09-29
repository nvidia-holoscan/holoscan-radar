# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Performance is an optional, standalone measurement suite. A recipe and its
# optional typed adapter live together under operators/<group>/<operator>/.
# The shared harness remains separate from production operator sources.
find_package(yaml-cpp REQUIRED)
add_library(radar_perf_utils INTERFACE)
target_include_directories(radar_perf_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/harness/cpp
    ${PROJECT_SOURCE_DIR}/operators)
target_link_libraries(radar_perf_utils INTERFACE
    holoscan::core
    holoscan::radar_common
    holoscan::radar_schemas
    CUDA::cudart
    yaml-cpp)
target_compile_definitions(radar_perf_utils INTERFACE
    HOLOSCAN_RADAR_PERF_OPERATOR_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/operators")

# One explicit entry per operator; keep this in the same group as the
# production operator. The executable name and build/performance/cpp output
# location stay stable regardless of source organization.
set(HOLOSCAN_RADAR_PERF_OPERATORS
    angle_estimation/beamscan_estimator
    angle_estimation/esprit_estimator
    angle_estimation/music_estimator
    angle_estimation/mvdr_estimator
    angle_estimation/phase_shift_beamformer
    angle_estimation/rank_one_residual_screen
    angle_estimation/root_music_estimator
    angle_estimation/snapshot_gather
    angle_estimation/subarray_selection
    clustering/cluster_dbscan
    detection/cfar_detector
    detection/detection_reduction
    detection/feature_assembly
    detection/multi_source_expansion
    fmcw/clutter_filter
    fmcw/ddm_dense_demultiplex
    fmcw/ddm_power_fold
    fmcw/ddm_sparse_demultiplex
    fmcw/doppler_notch
    fmcw/fmcw_doppler_processor
    fmcw/fmcw_range_doppler_processor
    fmcw/fmcw_range_processor
    fmcw/mimo_deinterleave
    fmcw/tdm_dense_compensation
    fmcw/tdm_sparse_compensation
    fmcw/virtual_array_assembly
)

function(radar_configure_perf_target NAME OP_TARGET)
    target_link_libraries(perf_${NAME} PRIVATE radar_perf_utils ${OP_TARGET})
    set_target_properties(perf_${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cpp")
endfunction()

set(HOLOSCAN_RADAR_PERF_TARGETS)
foreach(_perf_entry IN LISTS HOLOSCAN_RADAR_PERF_OPERATORS)
    get_filename_component(_perf_name "${_perf_entry}" NAME)
    set(_operator_dir "${CMAKE_CURRENT_SOURCE_DIR}/operators/${_perf_entry}")
    if(NOT EXISTS "${_operator_dir}/recipe.yaml")
        message(FATAL_ERROR "Missing performance recipe: ${_operator_dir}/recipe.yaml")
    endif()
    if(NOT EXISTS "${_operator_dir}/adapter.cpp")
        message(FATAL_ERROR "Missing performance adapter: ${_operator_dir}/adapter.cpp")
    endif()
    set(_op_target holoscan::${_perf_name}_op)
    if(NOT TARGET ${_op_target})
        message(STATUS "Skipping perf_${_perf_name} (missing ${_op_target})")
        continue()
    endif()
    add_executable(perf_${_perf_name} "${_operator_dir}/adapter.cpp")
    radar_configure_perf_target(${_perf_name} ${_op_target})
    list(APPEND HOLOSCAN_RADAR_PERF_TARGETS ${_perf_name})
endforeach()

# A standard recipe can omit adapter.cpp. Register its HSDK operator type once;
# the generated entry point uses the same named recipe and shared harness.
function(radar_add_yaml_perf ENTRY OP_TARGET OP_HEADER OP_TYPE)
    get_filename_component(_perf_name "${ENTRY}" NAME)
    set(_operator_dir "${CMAKE_CURRENT_SOURCE_DIR}/operators/${ENTRY}")
    if(NOT EXISTS "${_operator_dir}/recipe.yaml")
        message(FATAL_ERROR "Missing performance recipe: ${_operator_dir}/recipe.yaml")
    endif()
    if(NOT TARGET ${OP_TARGET})
        message(STATUS "Skipping perf_${_perf_name} (missing ${OP_TARGET})")
        return()
    endif()
    set(PERF_OPERATOR_NAME ${_perf_name})
    set(PERF_OPERATOR_HEADER ${OP_HEADER})
    set(PERF_OPERATOR_TYPE ${OP_TYPE})
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/generated/perf_${_perf_name}.cpp")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/harness/cpp/utils/perf_yaml_main.cpp.in
                   ${_generated} @ONLY)
    add_executable(perf_${_perf_name} ${_generated})
    radar_configure_perf_target(${_perf_name} ${OP_TARGET})
    list(APPEND HOLOSCAN_RADAR_PERF_TARGETS ${_perf_name})
    set(HOLOSCAN_RADAR_PERF_TARGETS "${HOLOSCAN_RADAR_PERF_TARGETS}" PARENT_SCOPE)
endfunction()

# Minimal smoke for every operator. Timing values remain informational.
if(HOLOSCAN_RADAR_BUILD_TESTING)
    foreach(_perf_name IN LISTS HOLOSCAN_RADAR_PERF_TARGETS)
        if(TARGET perf_${_perf_name})
            add_test(NAME perf_${_perf_name}_smoke
                COMMAND perf_${_perf_name} --scope minimal --sizes small
                    --warmup 1 --frames 3 --repeats 1 --quiet
                    --output ${CMAKE_CURRENT_BINARY_DIR}/cpp/perf_${_perf_name}_smoke.json)
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
                    --output ${CMAKE_CURRENT_BINARY_DIR}/cpp/perf_${_perf_name}_overrides_smoke.json)
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
                    --output ${CMAKE_CURRENT_BINARY_DIR}/cpp/perf_${_perf_name}_recipe_overrides_smoke.json)
            set_tests_properties(perf_${_perf_name}_recipe_overrides_smoke
                PROPERTIES LABELS "performance")
        endif()
    endforeach()
    foreach(_perf_name IN ITEMS fmcw_range_processor fmcw_doppler_processor)
        if(TARGET perf_${_perf_name})
            add_test(NAME perf_${_perf_name}_cached_plan_smoke
                COMMAND perf_${_perf_name} --scope thorough --sizes medium
                    --warmup 1 --frames 3 --repeats 2 --quiet
                    --output ${CMAKE_CURRENT_BINARY_DIR}/cpp/perf_${_perf_name}_cached_plan_smoke.json)
            set_tests_properties(perf_${_perf_name}_cached_plan_smoke PROPERTIES LABELS "performance")
        endif()
    endforeach()
endif()
