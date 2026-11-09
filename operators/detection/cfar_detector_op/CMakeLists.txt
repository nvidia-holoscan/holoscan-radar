# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(cfar_detector_op cfar_detector_op.cu cfar_alpha.cpp cfar_alpha_mc.cu
    nci_plan.cpp)
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

holoscan_radar_set_library_properties(cfar_detector_op)

install(TARGETS cfar_detector_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES cfar_detector_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/detection
)
