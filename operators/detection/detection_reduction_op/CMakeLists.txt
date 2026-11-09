# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Thin operator over holoscan::radar_common_reduce (group-by-label kernels).
add_library(detection_reduction_op detection_reduction_op.cu)
add_library(holoscan::detection_reduction_op ALIAS detection_reduction_op)

target_include_directories(detection_reduction_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(detection_reduction_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_common_reduce)

holoscan_radar_set_library_properties(detection_reduction_op)

install(TARGETS detection_reduction_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES
    detection_reduction_op.hpp
    detection_reduction_op_impl.cuh
    detection_reduction_stage.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/detection
)
