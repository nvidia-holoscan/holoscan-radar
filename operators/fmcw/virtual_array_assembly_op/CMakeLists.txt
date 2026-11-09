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
        holoscan::radar_schemas
    PRIVATE
        matx::matx)

holoscan_radar_set_library_properties(virtual_array_assembly_op)

install(TARGETS virtual_array_assembly_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES virtual_array_assembly_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw
)
install(FILES virtual_array_assembly_stage.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw/virtual_array_assembly_op
)
