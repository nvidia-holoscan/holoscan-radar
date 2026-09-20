# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# FMCW signal-processing operators (see README.md in this directory).
# Future MIMO marshaling operators also live in this group.
#
#  holoscan_radar_add_operator skips operators not yet ported
# to the HSDK 5 Operator<> model (see HoloscanRadarOptionalFeatures.cmake).
# The fused operator is declared first so its DEPENDS FORCE runs before
# dependency option() binds — same ordering rule as pkg/.
holoscan_radar_add_operator(fmcw_range_doppler_processor_op
    DEPENDS OPERATORS
        fmcw_range_processor_op
        clutter_filter_op
        fmcw_doppler_processor_op
)
holoscan_radar_add_operator(tdm_sparse_compensation_op)
holoscan_radar_add_operator(ddm_sparse_demultiplex_op)
holoscan_radar_add_operator(fmcw_range_processor_op)
holoscan_radar_add_operator(clutter_filter_op)
holoscan_radar_add_operator(fmcw_doppler_processor_op)
holoscan_radar_add_operator(doppler_notch_op)
holoscan_radar_add_operator(mimo_deinterleave_op)
holoscan_radar_add_operator(virtual_array_assembly_op)
holoscan_radar_add_operator(tdm_dense_compensation_op)
holoscan_radar_add_operator(ddm_dense_demultiplex_op)
holoscan_radar_add_operator(ddm_power_fold_op)
