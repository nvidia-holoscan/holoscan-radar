# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# group_by_label.cu owns the CUDA kernels; host launch wrappers are the public
# surface (no RDC required). Link this target rather than recompiling kernels.
add_library(detection_reduction_op
    detection_reduction_op.cu
    ${PROJECT_SOURCE_DIR}/operators/common/radar_common/reduce/group_by_label.cu)
add_library(holoscan::detection_reduction_op ALIAS detection_reduction_op)

target_include_directories(detection_reduction_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(detection_reduction_op
    PUBLIC
        holoscan::core
        holoscan::radar_common)

set_property(TARGET detection_reduction_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS detection_reduction_op
    EXPORT holoscan_detection_reduction_op_targets
)

install(EXPORT holoscan_detection_reduction_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES detection_reduction_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/detection
)
