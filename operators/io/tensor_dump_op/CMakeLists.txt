# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

holoscan_radar_set_library_properties(tensor_dump_op)

install(TARGETS tensor_dump_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES tensor_dump_op.hpp npy_writer.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/io
)
