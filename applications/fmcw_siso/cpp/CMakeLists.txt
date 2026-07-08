# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(fmcw_siso fmcw_siso.cpp)

target_link_libraries(fmcw_siso PRIVATE
    holoscan::fmcw_range_processor_op
    holoscan::clutter_filter_op
    holoscan::fmcw_doppler_processor_op
    holoscan::radar_common
    holoscan::core
    CUDA::cudart)

target_include_directories(fmcw_siso PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)
