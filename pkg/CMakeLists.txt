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
# Keep this selection independent of the Python provider's FMCW operator list.
# Entries are operator subdirectory selection keys (OP_<key>), not CMake targets.
# Existing operators use the same key for their library target and output basename;
# packaging CI expects an installed lib<key>.* for each selected entry. If an
# operator changes that naming or emits multiple libraries, update the artifact
# check in tests/packaging/check_packages.py to reflect its intended outputs.
# Override HOLOSCAN_RADAR_PKG_OPERATORS at configure time to select a subset.
set(HOLOSCAN_RADAR_PKG_OPERATORS
    clutter_filter_op
    snapshot_gather_op
    phase_shift_beamformer_op
    beamscan_estimator_op
    rank_one_residual_screen_op
    subarray_selection_op
    cluster_dbscan_op
    detection_reduction_op
    mvdr_estimator_op
    music_estimator_op
    feature_assembly_op
    multi_source_expansion_op
    cfar_detector_op
    root_music_estimator_op
    esprit_estimator_op
    mimo_deinterleave_op
    tdm_dense_compensation_op
    ddm_dense_demultiplex_op
    doppler_notch_op
    ddm_power_fold_op
    ddm_sparse_demultiplex_op
    tdm_sparse_compensation_op
    virtual_array_assembly_op
    fmcw_doppler_processor_op
    fmcw_range_processor_op
    fmcw_range_doppler_processor_op
    phase_correction_op
    sar_range_compression_op
    aperture_manager_op
    backprojection_op
    tensor_dump_op
    cphd_reader_op
    CACHE STRING "Operators included in the holoscan-radar Debian package")

add_holohub_package(holoscan-radar OPERATORS ${HOLOSCAN_RADAR_PKG_OPERATORS})
