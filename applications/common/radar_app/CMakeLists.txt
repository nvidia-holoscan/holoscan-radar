# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# radar_app — shared application support for the FMCW examples. It holds the
# reusable, app-agnostic source, reporting, config, and visualization pieces
# that would otherwise be copied per application.
#
# The base radar_app target remains INTERFACE. CUDA-backed visualization
# components use separate compiled targets and expose headers under
# "radar_app/...".
#
# The HSDK 5 dashboard stays optional; report sinks always build.
add_library(radar_app INTERFACE)
add_library(holoscan::radar_app ALIAS radar_app)
if(DEFINED RADAR_SOURCE_ROOT)
    set(_radar_source_root "${RADAR_SOURCE_ROOT}")
else()
    get_filename_component(_radar_source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()

# "radar_app/..." resolves from applications/common; the radar_common include
# root arrives transitively through holoscan::radar_common below.
target_include_directories(radar_app INTERFACE
    $<BUILD_INTERFACE:${_radar_source_root}/applications/common>)

target_link_libraries(radar_app INTERFACE
    holoscan::core
    holoscan::radar_common
    CUDA::cudart
    ${CMAKE_DL_LIBS})

# Cold application device selection is shared by C++, Python and SAR entry points.
add_library(radar_app_gpu STATIC app/gpu_placement.cpp)
target_include_directories(radar_app_gpu PUBLIC
    $<BUILD_INTERFACE:${_radar_source_root}/applications/common>)
target_link_libraries(radar_app_gpu PUBLIC holoscan::core holoscan::params_io CUDA::cudart)
set_target_properties(radar_app_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    target_sources(radar_app_gpu PRIVATE viz/image_fit.cu)
    target_compile_definitions(radar_app_gpu PRIVATE HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ=1)
    target_link_libraries(radar_app_gpu PRIVATE holoscan::visualization::holoviz)
endif()
target_link_libraries(radar_app INTERFACE radar_app_gpu)

# Generate GUI copy from the canonical C++ manifests; changes reconfigure the build.
set(_radar_descriptions "#pragma once\n#include <stdexcept>\n#include <string>\n#include <string_view>\nnamespace holoscan::radar::app {\ninline std::string application_description(std::string_view application, std::string_view mode = \"default\") {\n")
foreach(_radar_app IN ITEMS siso simo dense_beamforming mimo_tdm mimo_ddm adaptive_aoa multi_waveform)
    set(_radar_manifest "${_radar_source_root}/applications/fmcw/${_radar_app}/cpp/metadata.json")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_radar_manifest}")
    file(READ "${_radar_manifest}" _radar_metadata)
    string(JSON _radar_mode_count LENGTH "${_radar_metadata}" application modes)
    math(EXPR _radar_last_mode "${_radar_mode_count} - 1")
    foreach(_radar_index RANGE ${_radar_last_mode})
        string(JSON _radar_mode MEMBER "${_radar_metadata}" application modes ${_radar_index})
        if(_radar_mode MATCHES "smoke")
            continue()
        endif()
        string(JSON _radar_description GET "${_radar_metadata}"
            application modes "${_radar_mode}" description)
        string(REPLACE "\\" "\\\\" _radar_description "${_radar_description}")
        string(REPLACE "\"" "\\\"" _radar_description "${_radar_description}")
        string(REPLACE "\n" "\\n" _radar_description "${_radar_description}")
        string(APPEND _radar_descriptions
            "if (application == \"${_radar_app}\" && mode == \"${_radar_mode}\") return \"${_radar_description}\";\n")
    endforeach()
endforeach()
string(APPEND _radar_descriptions
    "throw std::invalid_argument(\"Unknown FMCW application description: \" + std::string(application) + \"/\" + std::string(mode));\n}\n}\n")
file(GENERATE OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/generated/radar_app/app/application_descriptions.hpp"
    CONTENT "${_radar_descriptions}")
target_include_directories(radar_app INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)

# The HSDK 5 dashboard compositor is compiled because its raster path contains
# CUDA kernels. Graph wiring remains header-only for reuse by FMCW applications.
if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    add_library(fmcw_dashboard STATIC viz/fmcw_dashboard_op.cu)
    add_library(holoscan::fmcw_dashboard ALIAS fmcw_dashboard)

    target_include_directories(fmcw_dashboard PUBLIC
        $<BUILD_INTERFACE:${_radar_source_root}/applications/common>)

    target_link_libraries(fmcw_dashboard PUBLIC
        holoscan::core
        holoscan::radar_app
        holoscan::radar_common
        holoscan::visualization::op_holoviz
        holoscan::radar_schemas
        CUDA::cudart)

    target_compile_definitions(fmcw_dashboard PUBLIC
        HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ=1)

    set_property(TARGET fmcw_dashboard
        PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(HOLOSCAN_RADAR_HAS_X11)
    target_link_libraries(radar_app INTERFACE X11 Xrender)
    target_compile_definitions(radar_app INTERFACE HOLOSCAN_RADAR_HAS_X11=1)
    target_link_libraries(radar_app_gpu PRIVATE X11)
    target_compile_definitions(radar_app_gpu PRIVATE HOLOSCAN_RADAR_HAS_X11=1)
endif()

# Python bindings live in python/ but are added from operators/CMakeLists.txt
# after radar_common exists (applications/ is configured before operators/).
