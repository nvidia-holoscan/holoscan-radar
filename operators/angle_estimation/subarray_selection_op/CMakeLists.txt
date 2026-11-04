# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(subarray_selection_op
    subarray_selection_op.cu)
add_library(holoscan::subarray_selection_op ALIAS subarray_selection_op)

target_include_directories(subarray_selection_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(subarray_selection_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_schemas
        holoscan::radar_common_spatial_smoothing)

set_property(TARGET subarray_selection_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS subarray_selection_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES subarray_selection_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/angle_estimation
)
