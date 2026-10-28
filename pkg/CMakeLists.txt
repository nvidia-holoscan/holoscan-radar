# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# add_holohub_package() handles the operator cascade for both standalone
# packaging (`holoscan package holoscan-radar --pkg-generator DEB`, which
# sets -DPKG_holoscan_radar=ON via the Holoscan CLI) and in-tree
# HoloHub-monorepo packaging once this module is fetched as an external
# dependency:
#
#   1. Defines option(PKG_holoscan_radar ... ${BUILD_ALL}) — ON
#      when BUILD_ALL=ON (default standalone) or when the CLI passes the
#      option explicitly.
#   2. add_subdirectory(holoscan-radar) — enters
#      pkg/holoscan-radar/ where holohub_configure_deb() emits
#      CPackConfig-holoscan-radar.cmake.
#   3. set("OP_<x>" ON CACHE BOOL ... FORCE) cascades to the operator
#      subprojects so the .deb gets its libraries. pkg/ is entered before
#      operators/, so they see the forced ON value at option().
#
# The package contains public operators, not the example applications.
# `holoscan package` configures with -DBUILD_ALL=OFF. The root CMake file
# checks that each selected operator is registered.
#
add_holohub_package(holoscan-radar
                    OPERATORS ${HOLOSCAN_RADAR_PACKAGE_OPERATORS})
