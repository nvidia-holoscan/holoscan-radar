# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The one io operator with no format-library dependency: it writes .npy and
# raw bytes itself (npy_writer), so it costs a build nothing beyond the SDK.
#
add_library(tensor_dump_op tensor_dump_op.cu npy_writer.cpp)
add_library(holoscan::tensor_dump_op ALIAS tensor_dump_op)

target_include_directories(tensor_dump_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(tensor_dump_op
    PUBLIC
        holoscan::core
        holoscan::radar_common)

set_property(TARGET tensor_dump_op PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS tensor_dump_op
    EXPORT holoscan_tensor_dump_op_targets
)

install(EXPORT holoscan_tensor_dump_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
install(FILES tensor_dump_op.hpp npy_writer.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/io
)
