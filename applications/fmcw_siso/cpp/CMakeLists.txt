# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(fmcw_siso fmcw_siso.cpp)

# yaml-cpp ships with the Holoscan SDK (found transitively via
# find_package(holoscan)); the app parses its runtime config with it.
find_package(yaml-cpp REQUIRED)

target_link_libraries(fmcw_siso PRIVATE
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::cfar_detector_op
    holoscan::feature_assembly_op
    holoscan::detection_reduction_op
    holoscan::cluster_dbscan_op
    holoscan::radar_app
    holoscan::radar_common_sim_gpu
    holoscan::radar_common
    holoscan::core
    holoscan::ops::holoviz
    holoscan::viz
    yaml-cpp::yaml-cpp
    CUDA::cudart)

target_include_directories(fmcw_siso PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)

# Stage the default config next to the executable so it is found without a CLI
# argument (the run command's workdir is the app bin directory).
configure_file(fmcw_siso.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/fmcw_siso.yaml COPYONLY)
