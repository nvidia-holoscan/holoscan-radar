# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(holoscan REQUIRED COMPONENTS core params_io)

if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    add_library(simo_viz STATIC viz/simo_viz_ops.cu)
    add_library(holoscan::simo_viz ALIAS simo_viz)
    target_include_directories(simo_viz PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_link_libraries(simo_viz PUBLIC
        holoscan::core
        holoscan::radar_app
        holoscan::radar_common
        holoscan::operators::holoviz
        CUDA::cudart)
    target_compile_definitions(simo_viz PUBLIC
        HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ=1)
    set_property(TARGET simo_viz PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

set(_simo_common_libraries
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op
    holoscan::feature_assembly_op
    holoscan::radar_app
    holoscan::radar_common_simulator
    holoscan::radar_common
    holoscan::params_io
    holoscan::core
    CUDA::cudart)

if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    list(APPEND _simo_common_libraries holoscan::simo_viz)
endif()

add_executable(fmcw_simo_detect_then_beamform
    detect_then_beamform/fmcw_simo_detect_then_beamform.cpp)
target_link_libraries(fmcw_simo_detect_then_beamform PRIVATE
    ${_simo_common_libraries}
    holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op
    holoscan::cluster_dbscan_op
    holoscan::detection_reduction_op)

add_executable(fmcw_simo_beamform_then_detect
    beamform_then_detect/fmcw_simo_beamform_then_detect.cpp)
target_link_libraries(fmcw_simo_beamform_then_detect PRIVATE
    ${_simo_common_libraries}
    holoscan::phase_shift_beamformer_op)

foreach(_target IN ITEMS
        fmcw_simo_detect_then_beamform
        fmcw_simo_beamform_then_detect)
    target_include_directories(${_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/operators)
endforeach()

configure_file(../config/default.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/default.yaml COPYONLY)
configure_file(../config/smoke.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/smoke.yaml COPYONLY)
