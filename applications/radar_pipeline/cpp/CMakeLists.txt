# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(radar_pipeline radar_pipeline.cpp)

target_link_libraries(radar_pipeline PRIVATE
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::radar_common
    holoscan::core
    CUDA::cudart)

target_include_directories(radar_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)
