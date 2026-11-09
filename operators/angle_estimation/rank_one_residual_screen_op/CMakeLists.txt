# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(rank_one_residual_screen_op rank_one_residual_screen_op.cu)
add_library(holoscan::rank_one_residual_screen_op ALIAS rank_one_residual_screen_op)
target_include_directories(rank_one_residual_screen_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(rank_one_residual_screen_op PUBLIC
    holoscan::core holoscan::radar_common holoscan::radar_schemas)
holoscan_radar_set_library_properties(rank_one_residual_screen_op)
install(TARGETS rank_one_residual_screen_op
    EXPORT holoscan_radar_targets)
include(GNUInstallDirs)
install(FILES rank_one_residual_screen_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/angle_estimation)
