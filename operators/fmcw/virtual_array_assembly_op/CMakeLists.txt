# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(virtual_array_assembly_op
    virtual_array_assembly_op.cu
    virtual_array_assembly_stage.cu)
add_library(holoscan::virtual_array_assembly_op ALIAS virtual_array_assembly_op)

target_include_directories(virtual_array_assembly_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(virtual_array_assembly_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx)

set_property(TARGET virtual_array_assembly_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS virtual_array_assembly_op
    EXPORT holoscan_virtual_array_assembly_op_targets
)

install(EXPORT holoscan_virtual_array_assembly_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES virtual_array_assembly_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw
)

add_subdirectory(python)
