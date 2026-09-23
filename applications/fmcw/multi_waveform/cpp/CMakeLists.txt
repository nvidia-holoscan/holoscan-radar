# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(holoscan REQUIRED COMPONENTS core params_io)

add_executable(multi_waveform multi_waveform.cpp)
target_include_directories(multi_waveform PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)
target_link_libraries(multi_waveform PRIVATE
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::mimo_deinterleave_op
    holoscan::tdm_sparse_compensation_op
    holoscan::cfar_detector_op
    holoscan::beamscan_estimator_op
    holoscan::feature_assembly_op
    holoscan::cluster_dbscan_op
    holoscan::detection_reduction_op
    holoscan::radar_app
    holoscan::radar_common_simulator
    holoscan::radar_common
    holoscan::params_io
    holoscan::core
    yaml-cpp
    CUDA::cudart)
if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    target_link_libraries(multi_waveform PRIVATE
        holoscan::fmcw_dashboard)
endif()

foreach(_config IN ITEMS
        multi_waveform.yaml
        multi_waveform_smoke.yaml)
    configure_file(../config/${_config}
        ${CMAKE_CURRENT_BINARY_DIR}/${_config} COPYONLY)
endforeach()
