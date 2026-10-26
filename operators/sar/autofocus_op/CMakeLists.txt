# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(autofocus_op autofocus_op.cu)
add_library(holoscan::autofocus_op ALIAS autofocus_op)

target_include_directories(autofocus_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(autofocus_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx
        CUDA::cudart)

set_property(TARGET autofocus_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS autofocus_op
    EXPORT holoscan_autofocus_op_targets
)

install(EXPORT holoscan_autofocus_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES autofocus_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/sar
)
