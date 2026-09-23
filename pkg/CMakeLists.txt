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
# Package the complete public operator and application surface.
set(_HOLOSCAN_RADAR_PKG_APPLICATIONS
    siso
    simo
    dense_beamforming
    mimo_tdm
    adaptive_aoa
    multi_waveform
    mimo_ddm)

add_holohub_package(holoscan-radar
                    OPERATORS    ${HOLOSCAN_RADAR_OPERATORS}
                    APPLICATIONS ${_HOLOSCAN_RADAR_PKG_APPLICATIONS})
