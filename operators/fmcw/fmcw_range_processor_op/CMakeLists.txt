# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(fmcw_range_processor_op fmcw_range_processor_op.cu)
add_library(holoscan::fmcw_range_processor_op ALIAS fmcw_range_processor_op)

target_include_directories(fmcw_range_processor_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(fmcw_range_processor_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        matx::matx)

set_property(TARGET fmcw_range_processor_op
    PROPERTY POSITION_INDEPENDENT_CODE ON)

# Optional NVTX compute()-phase instrumentation (no-op unless
# -DHOLOSCAN_RADAR_PERF_NVTX=ON). See cmake/RadarPerfNvtx.cmake.
radar_enable_perf_nvtx(fmcw_range_processor_op)

install(TARGETS fmcw_range_processor_op
    EXPORT holoscan_fmcw_range_processor_op_targets
)

install(EXPORT holoscan_fmcw_range_processor_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES fmcw_range_processor_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/fmcw
)

add_subdirectory(python)
