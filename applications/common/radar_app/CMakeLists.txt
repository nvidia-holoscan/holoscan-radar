# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# radar_app — the shared application-support library for the FMCW example apps
# (siso, and the future simo / mimo). It holds the reusable, app-agnostic
# pieces that would otherwise be copy-pasted per app: the reporting / drain
# sinks (report/) and the range-Doppler visualization operators + colormap
# helpers (viz/). App-specific graph wiring and the synthetic source stay in
# each application's own translation unit.
#
# Header-only (operators' methods are in-class inline; free helpers are marked
# inline), so this is an INTERFACE target: it carries include paths and link
# dependencies but compiles no objects of its own. Each app that links it picks
# up the headers under "radar_app/..." plus the transitive radar_common /
# Holoviz / CUDA dependencies the operators need.
#
# Holoviz-backed pieces stay optional. When Holoviz is absent but X11 is
# present ([EA HSDK5 ONLY]), the Holoviz-free RD raster + X11RdViewerOp still
# build so Graph smokes / future Graph-ported apps can show a live RD window.
# Report sinks always build.
add_library(radar_app INTERFACE)
add_library(holoscan::radar_app ALIAS radar_app)

# "radar_app/..." resolves from applications/common; the radar_common include
# root arrives transitively through holoscan::radar_common below.
target_include_directories(radar_app INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/applications/common>)

target_link_libraries(radar_app INTERFACE
    holoscan::core
    holoscan::radar_common
    CUDA::cudart)

# RdMapToImageOp's dB mapping needs nvcc (CUB sort + column medians). Build
# the kernels whenever any live RD path is available (Holoviz and/or X11).
if(HOLOSCAN_RADAR_HAS_RD_LIVE)
    add_library(radar_app_viz_kernels STATIC viz/viz_kernels.cu)
    add_library(holoscan::radar_app_viz_kernels ALIAS radar_app_viz_kernels)

    target_include_directories(radar_app_viz_kernels
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/applications/common>)

    target_link_libraries(radar_app_viz_kernels PUBLIC CUDA::cudart)

    set_property(TARGET radar_app_viz_kernels
        PROPERTY POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(radar_app INTERFACE
        holoscan::radar_app_viz_kernels)
endif()

if(HOLOSCAN_RADAR_HAS_HOLOVIZ)
    target_link_libraries(radar_app INTERFACE holoscan::viz)
    target_compile_definitions(radar_app INTERFACE HOLOSCAN_RADAR_HAS_HOLOVIZ=1)
endif()

if(HOLOSCAN_RADAR_HAS_X11)
    target_link_libraries(radar_app INTERFACE X11)
    target_compile_definitions(radar_app INTERFACE HOLOSCAN_RADAR_HAS_X11=1)
endif()

# Python bindings live in python/ but are added from operators/CMakeLists.txt
# after radar_common exists (applications/ is configured before operators/).
