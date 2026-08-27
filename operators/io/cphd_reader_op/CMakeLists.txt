# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The XML dependency the io group exists to contain: it is linked PRIVATE and
# named by no public header, so a build that leaves OP_cphd_reader_op off
# never looks for it.
#
# Searched without REQUIRED so the failure can name the fix: REQUIRED aborts
# with "Could NOT find TINYXML2_INCLUDE_DIR", which says nothing about what to
# install or that the operator can simply be switched off.
find_path(TINYXML2_INCLUDE_DIR NAMES tinyxml2.h)
find_library(TINYXML2_LIBRARY NAMES tinyxml2)
if(NOT TINYXML2_INCLUDE_DIR OR NOT TINYXML2_LIBRARY)
    message(FATAL_ERROR
        "cphd_reader_op needs tinyxml2 to parse the CPHD XML metadata block. "
        "Install it (Debian/Ubuntu: apt-get install libtinyxml2-dev), or "
        "configure with -DOP_cphd_reader_op=OFF to build without the reader.")
endif()

add_library(cphd_reader_op
    cphd_cf8_decode.cu
    cphd_file.cpp
    cphd_reader_op.cu)
add_library(holoscan::cphd_reader_op ALIAS cphd_reader_op)

target_include_directories(cphd_reader_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)
target_include_directories(cphd_reader_op PRIVATE ${TINYXML2_INCLUDE_DIR})

target_link_libraries(cphd_reader_op
    PUBLIC
        holoscan::core
        holoscan::radar_common
    PRIVATE
        ${TINYXML2_LIBRARY})

set_property(TARGET cphd_reader_op PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS cphd_reader_op
    EXPORT holoscan_cphd_reader_op_targets
)

install(EXPORT holoscan_cphd_reader_op_targets
    NAMESPACE holoscan::
    DESTINATION lib/cmake/holoscan_radar
)

include(GNUInstallDirs)
# collection_paths.hpp travels with the operator header, which includes it --
# installing one without the other leaves an installed header that cannot
# compile.
install(FILES cphd_reader_op.hpp collection_paths.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/io
)
