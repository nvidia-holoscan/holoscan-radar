# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(clutter_filter_op clutter_filter_op.cu)
add_library(holoscan::clutter_filter_op ALIAS clutter_filter_op)

target_include_directories(clutter_filter_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(clutter_filter_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx)

set_property(TARGET clutter_filter_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS clutter_filter_op
    EXPORT holoscan_clutter_filter_op_targets
)

install(EXPORT holoscan_clutter_filter_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES clutter_filter_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw
)

add_subdirectory(python)
