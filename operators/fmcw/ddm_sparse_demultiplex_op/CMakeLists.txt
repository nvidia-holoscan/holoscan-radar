# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(ddm_sparse_demultiplex_op
    ddm_sparse_demultiplex_op.cu
    ddm_sparse_demultiplex_stage.cu)
add_library(holoscan::ddm_sparse_demultiplex_op ALIAS
    ddm_sparse_demultiplex_op)
target_include_directories(ddm_sparse_demultiplex_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(ddm_sparse_demultiplex_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_common_mimo_compensation
        holoscan::radar_schemas
)
set_property(TARGET ddm_sparse_demultiplex_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)
install(TARGETS ddm_sparse_demultiplex_op
    EXPORT holoscan_ddm_sparse_demultiplex_op_targets)
install(EXPORT holoscan_ddm_sparse_demultiplex_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar)
include(GNUInstallDirs)
install(FILES ddm_sparse_demultiplex_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw)
