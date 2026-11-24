# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(RadarDebianRuntime LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)
find_package(holoscan 5 CONFIG REQUIRED COMPONENTS core params_io)
find_package(holoscan-radar 1.0 CONFIG REQUIRED)

if(NOT DEFINED RADAR_SOURCE_ROOT)
    get_filename_component(RADAR_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()
if(NOT EXISTS "${RADAR_SOURCE_ROOT}/applications/fmcw/siso/cpp/siso.cpp")
    message(FATAL_ERROR "Matching Radar source checkout is required")
endif()
set(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ OFF)
set(HOLOSCAN_RADAR_HAS_X11 OFF)
if(NOT DEFINED RADAR_APP_CMAKE_DIR)
    set(RADAR_APP_CMAKE_DIR "${RADAR_SOURCE_ROOT}/applications/common/radar_app")
endif()
add_subdirectory("${RADAR_APP_CMAKE_DIR}" radar_app)

function(add_radar_graph name source)
    add_executable(${name} "${RADAR_SOURCE_ROOT}/${source}")
    target_compile_features(${name} PRIVATE cxx_std_20)
    target_link_libraries(${name} PRIVATE holoscan::radar_app holoscan::params_io)
endfunction()

add_radar_graph(siso applications/fmcw/siso/cpp/siso.cpp)
target_link_libraries(siso PRIVATE holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op holoscan::feature_assembly_op
    holoscan::cluster_dbscan_op holoscan::detection_reduction_op
    holoscan::radar_common_simulator)

add_radar_graph(simo applications/fmcw/simo/cpp/simo.cpp)
target_link_libraries(simo PRIVATE holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op holoscan::feature_assembly_op
    holoscan::cluster_dbscan_op holoscan::detection_reduction_op
    holoscan::radar_common_simulator)

add_radar_graph(mimo_ddm applications/fmcw/mimo_ddm/cpp/mimo_ddm.cpp)
target_link_libraries(mimo_ddm PRIVATE holoscan::fmcw_range_processor_op
    holoscan::fmcw_doppler_processor_op holoscan::ddm_dense_demultiplex_op
    holoscan::doppler_notch_op holoscan::virtual_array_assembly_op
    holoscan::cfar_detector_op holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op holoscan::feature_assembly_op
    holoscan::cluster_dbscan_op holoscan::detection_reduction_op
    holoscan::radar_common_simulator)

set(_fmcw_root "${RADAR_SOURCE_ROOT}/applications/fmcw")
foreach(_app IN ITEMS siso simo mimo_ddm)
    file(GLOB _configs "${_fmcw_root}/${_app}/config/*.yaml")
    foreach(_config IN LISTS _configs)
        get_filename_component(_name "${_config}" NAME)
        configure_file("${_config}" "${CMAKE_CURRENT_BINARY_DIR}/${_app}/${_name}" COPYONLY)
    endforeach()
endforeach()

add_library(sar_mosaic STATIC
    "${RADAR_SOURCE_ROOT}/applications/common/radar_app/sar/mosaic_sample.cu")
target_include_directories(sar_mosaic PRIVATE "${RADAR_SOURCE_ROOT}/applications/common")
target_link_libraries(sar_mosaic PUBLIC CUDA::cudart)
target_compile_definitions(sar_mosaic PRIVATE
    HOLOSCAN_RADAR_DATA_DIR="${RADAR_SOURCE_ROOT}/data")

foreach(_app IN ITEMS spotlight tiled_strip)
    add_radar_graph(${_app} "applications/sar/${_app}/cpp/${_app}.cpp")
    target_compile_definitions(${_app} PRIVATE
        HOLOSCAN_RADAR_DATA_DIR="${RADAR_SOURCE_ROOT}/data")
    target_link_libraries(${_app} PRIVATE holoscan::cphd_reader_op
        holoscan::sar_range_compression_op holoscan::aperture_manager_op
        holoscan::backprojection_op holoscan::tensor_dump_op)
endforeach()
target_link_libraries(tiled_strip PRIVATE sar_mosaic)

add_executable(generate_cphd_fixture
    "${RADAR_SOURCE_ROOT}/tests/cpp/io/generate_cphd_fixture.cpp")
target_include_directories(generate_cphd_fixture PRIVATE "${RADAR_SOURCE_ROOT}/tests/cpp")
target_compile_features(generate_cphd_fixture PRIVATE cxx_std_20)

set(SAR_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/generated_sar_smoke.cphd")
foreach(_app IN ITEMS spotlight tiled_strip)
    configure_file("${RADAR_SOURCE_ROOT}/applications/sar/${_app}/cpp/${_app}_smoke.yaml.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${_app}_generated_smoke.yaml" @ONLY)
endforeach()

add_test(NAME sar_fixture COMMAND $<TARGET_FILE:generate_cphd_fixture> "${SAR_FIXTURE}")
set_tests_properties(sar_fixture PROPERTIES FIXTURES_SETUP sar_cphd)
foreach(_app IN ITEMS spotlight tiled_strip)
    if(_app STREQUAL "spotlight")
        set(_count 1)
        set(_dims 5x5)
    else()
        set(_count 5)
        set(_dims 15x6)
    endif()
    add_test(NAME sar_${_app} COMMAND "${CMAKE_COMMAND}"
        "-DSAR_EXECUTABLE=$<TARGET_FILE:${_app}>"
        "-DSAR_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/${_app}_generated_smoke.yaml"
        "-DSAR_COUNT=${_count}" "-DSAR_DIMS=${_dims}"
        -P "${RADAR_SOURCE_ROOT}/tests/cpp/check_sar_smoke.cmake")
    set_tests_properties(sar_${_app} PROPERTIES FIXTURES_REQUIRED sar_cphd)
endforeach()

foreach(_app IN ITEMS siso simo mimo_ddm)
    if(_app STREQUAL "mimo_ddm")
        set(_preset mimo_ddm_nonlinear_smoke.yaml)
    else()
        set(_preset ${_app}_smoke.yaml)
    endif()
    add_test(NAME fmcw_${_app} COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/check_fmcw.py"
        "$<TARGET_FILE:${_app}>"
        "${CMAKE_CURRENT_BINARY_DIR}/${_app}/${_preset}" ${_app})
endforeach()
