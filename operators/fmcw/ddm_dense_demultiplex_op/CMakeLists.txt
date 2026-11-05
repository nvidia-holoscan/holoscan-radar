# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(ddm_dense_demultiplex_op ddm_dense_demultiplex_op.cu)
add_library(holoscan::ddm_dense_demultiplex_op ALIAS ddm_dense_demultiplex_op)
target_include_directories(ddm_dense_demultiplex_op PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators> $<INSTALL_INTERFACE:include>)
target_link_libraries(ddm_dense_demultiplex_op PUBLIC holoscan::core holoscan::radar_common holoscan::radar_common_mimo_compensation)
set_property(TARGET ddm_dense_demultiplex_op PROPERTY POSITION_INDEPENDENT_CODE ON)
install(TARGETS ddm_dense_demultiplex_op EXPORT holoscan_radar_targets)
include(GNUInstallDirs)
install(FILES ddm_dense_demultiplex_op.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw)
