# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(beamscan_estimator_op beamscan_estimator_op.cu)
add_library(holoscan::beamscan_estimator_op ALIAS beamscan_estimator_op)

target_include_directories(beamscan_estimator_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(beamscan_estimator_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_common_steering
        holoscan::radar_schemas
    PRIVATE
        matx::matx)

holoscan_radar_set_library_properties(beamscan_estimator_op)

install(TARGETS beamscan_estimator_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES beamscan_estimator_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/angle_estimation
)
