# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(radar_op radar_op.cpp)
add_library(holoscan::radar_op ALIAS radar_op)

target_include_directories(radar_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(radar_op PRIVATE holoscan::core)

install(TARGETS radar_op
    EXPORT holoscan_radar_op_targets
)

install(EXPORT holoscan_radar_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

add_subdirectory(python)

