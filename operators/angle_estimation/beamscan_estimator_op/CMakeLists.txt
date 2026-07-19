# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(beamscan_estimator_op
    beamscan_estimator_op.cu
    is_valid_mask.cpp)
add_library(holoscan::beamscan_estimator_op ALIAS beamscan_estimator_op)

target_include_directories(beamscan_estimator_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(beamscan_estimator_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_common_steering
    PRIVATE
        matx::matx)

set_property(TARGET beamscan_estimator_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS beamscan_estimator_op
    EXPORT holoscan_beamscan_estimator_op_targets
)

install(EXPORT holoscan_beamscan_estimator_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES beamscan_estimator_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/angle_estimation
)

add_subdirectory(python)
