# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(sar_range_compression_op sar_range_compression_op.cu)
add_library(holoscan::sar_range_compression_op ALIAS sar_range_compression_op)

target_include_directories(sar_range_compression_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(sar_range_compression_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_schemas
    PRIVATE
        matx::matx)

set_property(TARGET sar_range_compression_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS sar_range_compression_op
    EXPORT holoscan_sar_range_compression_op_targets
)

install(EXPORT holoscan_sar_range_compression_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES sar_range_compression_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/sar
)
