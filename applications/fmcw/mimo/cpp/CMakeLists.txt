# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(holoscan REQUIRED COMPONENTS core params_io)

set(_mimo_common_libraries
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::virtual_array_assembly_op
    holoscan::cfar_detector_op
    holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op
    holoscan::feature_assembly_op
    holoscan::cluster_dbscan_op
    holoscan::detection_reduction_op
    holoscan::radar_app
    holoscan::radar_common_simulator
    holoscan::radar_common
    holoscan::params_io
    holoscan::core
    CUDA::cudart)

if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    list(APPEND _mimo_common_libraries holoscan::fmcw_dashboard)
endif()

add_executable(fmcw_mimo_tdm tdm/fmcw_mimo_tdm.cpp)
target_link_libraries(fmcw_mimo_tdm PRIVATE
    ${_mimo_common_libraries}
    holoscan::mimo_deinterleave_op
    holoscan::tdm_sparse_compensation_op)

add_executable(fmcw_mimo_ddm ddm/fmcw_mimo_ddm.cpp)
target_link_libraries(fmcw_mimo_ddm PRIVATE
    ${_mimo_common_libraries}
    holoscan::ddm_dense_demultiplex_op
    holoscan::doppler_notch_op)

foreach(_target IN ITEMS fmcw_mimo_tdm fmcw_mimo_ddm)
    target_include_directories(${_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/operators)
endforeach()

foreach(_config IN ITEMS
        tdm.yaml
        tdm_smoke.yaml
        ddm.yaml
        ddm_smoke.yaml
        ddm_phase_codes.yaml)
    configure_file(../config/${_config}
        ${CMAKE_CURRENT_BINARY_DIR}/${_config} COPYONLY)
endforeach()
