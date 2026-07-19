# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
add_library(radar_app INTERFACE)
add_library(holoscan::radar_app ALIAS radar_app)

# "radar_app/..." resolves from applications/common; the radar_common include
# root arrives transitively through holoscan::radar_common below.
target_include_directories(radar_app INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/applications/common>)

target_link_libraries(radar_app INTERFACE
    holoscan::core
    holoscan::radar_common
    holoscan::viz
    CUDA::cudart)

# Python bindings live in python/ but are added from operators/CMakeLists.txt
# after radar_common exists (applications/ is configured before operators/).
