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
# [EA HSDK5 ONLY] Demo applications require their visualization path and
# operator dependencies on the port allow-list — listing an app here FORCEs
# APP_<name> and its OP_<dep> options ON, and an unported dep is never created
# as a target. Every app needs the FMCW range/Doppler chain; mimo also needs
# the MIMO marshaling group. Keep these gates in step with applications/
# CMakeLists.txt.
holoscan_radar_operators_ported(_HOLOSCAN_RADAR_PKG_FMCW_CHAIN
    fmcw_range_processor_op
    fmcw_doppler_processor_op)
holoscan_radar_operators_ported(_HOLOSCAN_RADAR_PKG_MIMO
    mimo_deinterleave_op
    tdm_sparse_compensation_op
    ddm_dense_demultiplex_op
    doppler_notch_op
    virtual_array_assembly_op)

set(_HOLOSCAN_RADAR_PKG_APPLICATIONS)
if(_HOLOSCAN_RADAR_PKG_FMCW_CHAIN)
    list(APPEND _HOLOSCAN_RADAR_PKG_APPLICATIONS
        siso
        simo
        simo_dense_beamforming)
    if(_HOLOSCAN_RADAR_PKG_MIMO)
        list(APPEND _HOLOSCAN_RADAR_PKG_APPLICATIONS
            mimo_tdm
            mimo_ddm)
    endif()
endif()

# [EA HSDK5 ONLY] Package exactly the ported operators so FORCE OP_* never
# names an unported target; the allow-list is the single source of truth.
add_holohub_package(holoscan-radar
                    OPERATORS    ${HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS}
                    APPLICATIONS ${_HOLOSCAN_RADAR_PKG_APPLICATIONS})
