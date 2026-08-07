# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(mimo_coherent_compensation_op
    mimo_coherent_compensation_op.cu
    mimo_coherent_compensation_stage.cu)
add_library(holoscan::mimo_coherent_compensation_op ALIAS
            mimo_coherent_compensation_op)

target_include_directories(mimo_coherent_compensation_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(mimo_coherent_compensation_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx
        CUDA::cufft)

set_property(TARGET mimo_coherent_compensation_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS mimo_coherent_compensation_op
    EXPORT holoscan_mimo_coherent_compensation_op_targets
)

install(EXPORT holoscan_mimo_coherent_compensation_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES mimo_coherent_compensation_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw
)

add_subdirectory(python)
