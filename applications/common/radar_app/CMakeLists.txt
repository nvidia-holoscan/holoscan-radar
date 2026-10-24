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

# "radar_app/..." resolves from applications/common; the radar_common include
# root arrives transitively through holoscan::radar_common below.
target_include_directories(radar_app INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/applications/common>)

target_link_libraries(radar_app INTERFACE
    holoscan::core
    holoscan::radar_common
    CUDA::cudart
    ${CMAKE_DL_LIBS})

# The HSDK 5 dashboard compositor is compiled because its raster path contains
# CUDA kernels. Graph wiring remains header-only for reuse by FMCW applications.
if(HOLOSCAN_RADAR_HAS_HSDK5_HOLOVIZ)
    add_library(fmcw_dashboard STATIC viz/fmcw_dashboard_op.cu)
    add_library(holoscan::fmcw_dashboard ALIAS fmcw_dashboard)

    target_include_directories(fmcw_dashboard PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/applications/common>)

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
    target_link_libraries(radar_app INTERFACE X11)
    target_compile_definitions(radar_app INTERFACE HOLOSCAN_RADAR_HAS_X11=1)
endif()

# Python bindings live in python/ but are added from operators/CMakeLists.txt
# after radar_common exists (applications/ is configured before operators/).
