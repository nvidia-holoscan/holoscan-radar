# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(cfar_detector_op cfar_detector_op.cu cfar_alpha.cpp nci_plan.cpp)
add_library(holoscan::cfar_detector_op ALIAS cfar_detector_op)

target_include_directories(cfar_detector_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(cfar_detector_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx)

set_property(TARGET cfar_detector_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS cfar_detector_op
    EXPORT holoscan_cfar_detector_op_targets
)

install(EXPORT holoscan_cfar_detector_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES cfar_detector_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/detection
)
