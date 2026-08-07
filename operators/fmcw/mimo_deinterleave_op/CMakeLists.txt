# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(mimo_deinterleave_op
    mimo_deinterleave_op.cu
    mimo_deinterleave_stage.cu)
add_library(holoscan::mimo_deinterleave_op ALIAS mimo_deinterleave_op)

target_include_directories(mimo_deinterleave_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(mimo_deinterleave_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx)

set_property(TARGET mimo_deinterleave_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS mimo_deinterleave_op
    EXPORT holoscan_mimo_deinterleave_op_targets
)

install(EXPORT holoscan_mimo_deinterleave_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES mimo_deinterleave_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw
)

add_subdirectory(python)
