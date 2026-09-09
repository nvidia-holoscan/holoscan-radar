# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(tdm_sparse_compensation_op
    tdm_sparse_compensation_op.cu
    tdm_sparse_compensation_stage.cu)
add_library(holoscan::tdm_sparse_compensation_op ALIAS
    tdm_sparse_compensation_op)

target_include_directories(tdm_sparse_compensation_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(tdm_sparse_compensation_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_schemas
    PRIVATE
        holoscan::mimo_coherent_compensation_op
        holoscan::virtual_array_assembly_op)

set_property(TARGET tdm_sparse_compensation_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS tdm_sparse_compensation_op
    EXPORT holoscan_tdm_sparse_compensation_op_targets)

install(EXPORT holoscan_tdm_sparse_compensation_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar)

include(GNUInstallDirs)
install(FILES tdm_sparse_compensation_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw)

if(HOLOSCAN_RADAR_BUILD_PYTHON)
    add_subdirectory(python)
endif()
