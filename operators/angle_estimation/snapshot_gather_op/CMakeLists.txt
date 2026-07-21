# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(snapshot_gather_op snapshot_gather_op.cu)
add_library(holoscan::snapshot_gather_op ALIAS snapshot_gather_op)

target_include_directories(snapshot_gather_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(snapshot_gather_op
    PUBLIC
        holoscan::core
        holoscan::radar_common)

set_property(TARGET snapshot_gather_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS snapshot_gather_op
    EXPORT holoscan_snapshot_gather_op_targets
)

install(EXPORT holoscan_snapshot_gather_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES snapshot_gather_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/angle_estimation
)

add_subdirectory(python)
