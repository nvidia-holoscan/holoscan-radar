# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# add_holohub_package() handles the full cascade for both standalone
# packaging (`./holohub package holoscan-radar --pkg-generator DEB`, which
# sets -DPKG_holoscan_radar=ON via the HoloHub CLI) and in-tree
# HoloHub-monorepo packaging once this module is fetched as an external
# dependency:
#
#   1. Defines option(PKG_holoscan_radar ... ${BUILD_ALL}) — ON
#      when BUILD_ALL=ON (default standalone) or when the CLI passes the
#      option explicitly.
#   2. add_subdirectory(holoscan-radar) — enters
#      pkg/holoscan-radar/ where holohub_configure_deb() emits
#      CPackConfig-holoscan-radar.cmake.
#   3. set("OP_<x>" ON CACHE BOOL ... FORCE) and the equivalent for
#      APPLICATIONS — cascades to the operator and application subprojects
#      so the .deb gets actual content. pkg/ is add_subdirectory()'d
#      *before* operators/ and applications/, so those see the forced ON
#      value when their own helpers run option().
#
# These lists must name EVERY operator and application the package should
# contain. `holoscan package` configures with -DBUILD_ALL=OFF, so anything
# absent here is never enabled, never built, and never installed — it simply
# does not appear in the .deb. Nothing warns about the omission.
#
# Demo applications require Holoviz; omit them when it is gated off
# ([EA HSDK5 ONLY]).
set(_HOLOSCAN_RADAR_PKG_APPLICATIONS)
if(HOLOSCAN_RADAR_HAS_HOLOVIZ)
    list(APPEND _HOLOSCAN_RADAR_PKG_APPLICATIONS
        fmcw_aoa_estimators
        fmcw_dense_simo
        fmcw_mimo
        fmcw_simo
        fmcw_siso)
endif()

# [EA HSDK5 ONLY] Package only HSDK-5-ported operators so FORCE OP_* does
# not require unported targets. Expand with HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS.
add_holohub_package(holoscan-radar
                    OPERATORS    clutter_filter_op
                    APPLICATIONS ${_HOLOSCAN_RADAR_PKG_APPLICATIONS})

