# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(radar_op radar_op.cpp)
add_library(holoscan::radar_op ALIAS radar_op)

target_include_directories(radar_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(radar_op PUBLIC holoscan::core)

install(TARGETS radar_op
    EXPORT holoscan_radar_op_targets
)

install(EXPORT holoscan_radar_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES radar_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar
)

add_subdirectory(python)

