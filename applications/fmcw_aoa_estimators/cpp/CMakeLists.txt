# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(fmcw_aoa_estimators fmcw_aoa_estimators.cpp)

find_package(yaml-cpp REQUIRED)

target_link_libraries(fmcw_aoa_estimators PRIVATE
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op
    holoscan::snapshot_gather_op
    holoscan::beamscan_estimator_op
    holoscan::mvdr_estimator_op
    holoscan::music_estimator_op
    holoscan::root_music_estimator_op
    holoscan::esprit_estimator_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op
    holoscan::cluster_dbscan_op
    holoscan::radar_app
    holoscan::radar_common_simulator
    holoscan::radar_common
    holoscan::core
    holoscan::ops::holoviz
    holoscan::viz
    yaml-cpp::yaml-cpp
    CUDA::cudart)

target_include_directories(fmcw_aoa_estimators PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)

configure_file(fmcw_aoa_estimators.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/fmcw_aoa_estimators.yaml COPYONLY)
