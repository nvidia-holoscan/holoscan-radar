# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(tdm_dense_compensation_op tdm_dense_compensation_op.cu)
add_library(holoscan::tdm_dense_compensation_op ALIAS tdm_dense_compensation_op)
target_include_directories(tdm_dense_compensation_op PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators> $<INSTALL_INTERFACE:include>)
target_link_libraries(tdm_dense_compensation_op PUBLIC holoscan::core holoscan::radar_common holoscan::radar_common_mimo_compensation holoscan::radar_schemas)
holoscan_radar_set_library_properties(tdm_dense_compensation_op)
install(TARGETS tdm_dense_compensation_op EXPORT holoscan_radar_targets)
include(GNUInstallDirs)
install(FILES tdm_dense_compensation_op.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw)
