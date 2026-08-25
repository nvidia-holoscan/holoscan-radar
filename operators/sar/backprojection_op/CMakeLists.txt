# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(backprojection_op backprojection_op.cu)
add_library(holoscan::backprojection_op ALIAS backprojection_op)

target_include_directories(backprojection_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(backprojection_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx)

set_property(TARGET backprojection_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

# I1: bundled matx::experimental::sar_bp needs C++20 (concepts). Local
# to this translation unit; not a MatX pin bump.
set_target_properties(backprojection_op PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)

install(TARGETS backprojection_op
    EXPORT holoscan_backprojection_op_targets
)

install(EXPORT holoscan_backprojection_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES backprojection_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/sar
)
