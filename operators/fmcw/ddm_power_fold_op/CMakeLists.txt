# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(ddm_power_fold_op ddm_power_fold_op.cu)
add_library(holoscan::ddm_power_fold_op ALIAS ddm_power_fold_op)

target_include_directories(ddm_power_fold_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(ddm_power_fold_op PUBLIC
    holoscan::core
    holoscan::radar_common
    holoscan::radar_common_mimo_compensation)
set_property(TARGET ddm_power_fold_op PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS ddm_power_fold_op
    EXPORT holoscan_ddm_power_fold_op_targets)
install(EXPORT holoscan_ddm_power_fold_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar)
include(GNUInstallDirs)
install(FILES ddm_power_fold_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw)

if(HOLOSCAN_RADAR_BUILD_PYTHON)
    add_subdirectory(python)
endif()
