# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(radar_pipeline radar_pipeline.cpp)

target_link_libraries(radar_pipeline PRIVATE
    holoscan::radar_op
    holoscan::core)

target_include_directories(radar_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)

