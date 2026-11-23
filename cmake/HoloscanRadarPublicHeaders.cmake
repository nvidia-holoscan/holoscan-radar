# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

# Give source builds the same operator include spelling as installed consumers.
# Links keep edits to an operator header visible without reconfiguring CMake.
function(holoscan_radar_stage_public_headers)
    set(HOLOSCAN_RADAR_PUBLIC_HEADER_ROOT
        "${PROJECT_BINARY_DIR}/radar_public_headers" PARENT_SCOPE)
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/operators/*/*/*_op.hpp"
        "${PROJECT_SOURCE_DIR}/operators/*/*/params_config.hpp")
    list(APPEND _headers
        "${PROJECT_SOURCE_DIR}/operators/io/cphd_reader_op/cphd_file.hpp"
        "${PROJECT_SOURCE_DIR}/operators/io/tensor_dump_op/npy_writer.hpp")
    foreach(_header IN LISTS _headers)
        get_filename_component(_name "${_header}" NAME)
        get_filename_component(_op_dir "${_header}" DIRECTORY)
        get_filename_component(_op "${_op_dir}" NAME)
        get_filename_component(_group_dir "${_op_dir}" DIRECTORY)
        get_filename_component(_group "${_group_dir}" NAME)
        if(_name STREQUAL "params_config.hpp")
            set(_relative "holoscan/radar/${_group}/${_op}/${_name}")
        else()
            set(_relative "holoscan/radar/${_group}/${_name}")
        endif()
        set(_link "${PROJECT_BINARY_DIR}/radar_public_headers/${_relative}")
        get_filename_component(_parent "${_link}" DIRECTORY)
        file(MAKE_DIRECTORY "${_parent}")
        file(CREATE_LINK "${_header}" "${_link}" SYMBOLIC COPY_ON_ERROR)
    endforeach()
endfunction()

# Parameter codecs are used by app YAML readers. Install only codecs whose
# operator is part of this build, preserving intentional package subsets.
function(holoscan_radar_install_app_codecs)
    file(GLOB_RECURSE _codecs CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/operators/*/*/params_config.hpp")
    foreach(_codec IN LISTS _codecs)
        get_filename_component(_op_dir "${_codec}" DIRECTORY)
        get_filename_component(_op "${_op_dir}" NAME)
        get_filename_component(_group_dir "${_op_dir}" DIRECTORY)
        get_filename_component(_group "${_group_dir}" NAME)
        if(TARGET ${_op})
            install(FILES "${_codec}"
                DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/holoscan/radar/${_group}/${_op}")
        endif()
    endforeach()
endfunction()
