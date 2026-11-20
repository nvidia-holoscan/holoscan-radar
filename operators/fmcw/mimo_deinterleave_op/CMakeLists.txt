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

holoscan_radar_set_library_properties(mimo_deinterleave_op)

install(TARGETS mimo_deinterleave_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES mimo_deinterleave_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw
)
