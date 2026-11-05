# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The XML dependency the io group exists to contain: it is linked PRIVATE and
# named by no public header, so a build that leaves OP_cphd_reader_op off
# never looks for it.
#
# Search quietly so a missing config reports the package to install and the
# option for disabling this operator. Use the imported target so installed
# static exports resolve tinyxml2 at the consumer instead of storing a build path.
find_package(tinyxml2 QUIET CONFIG)
if(NOT TARGET tinyxml2::tinyxml2)
    message(FATAL_ERROR
        "cphd_reader_op needs tinyxml2. Install libtinyxml2-dev, or "
        "configure with -DOP_cphd_reader_op=OFF to build without the reader.")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(cphd_reader_op
    cphd_cf8_decode.cu
    cphd_file.cpp
    cphd_reader_op.cu)
add_library(holoscan::cphd_reader_op ALIAS cphd_reader_op)

target_include_directories(cphd_reader_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(cphd_reader_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
        holoscan::radar_schemas
    PRIVATE
        tinyxml2::tinyxml2
        Threads::Threads)

set_property(TARGET cphd_reader_op PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS cphd_reader_op
    EXPORT holoscan_radar_targets
)

include(GNUInstallDirs)
install(FILES cphd_reader_op.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/io
)
# The operator header includes these as "io/cphd_reader_op/<file>"; installing
# them elsewhere leaves an installed header that cannot compile.
install(FILES collection_paths.hpp host_read_ahead.hpp playback_scheduler.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/io/cphd_reader_op
)
