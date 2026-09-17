# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared application-support library (radar_app). Report sinks are always
# available; Holoviz-backed viz helpers and the demo apps that need them are
# gated on HOLOSCAN_RADAR_HAS_HOLOVIZ. [EA HSDK5 ONLY] Graph-ported demos
# (fmcw_siso and the standalone SIMO/MIMO apps) build under HOLOSCAN_RADAR_HAS_RD_LIVE
# (X11 and/or Holoviz).
add_subdirectory(common)

# Each workflow is a standalone application with its own APP_<name>
# gate, executable, metadata, and configuration set.
if(HOLOSCAN_RADAR_HAS_RD_LIVE)
    holoscan_radar_add_application(simo_detect_then_beamform
        DIRECTORY fmcw/simo_detect_then_beamform
        DEPENDS OPERATORS
            fmcw_range_processor_op
            clutter_filter_op
            fmcw_doppler_processor_op
            cfar_detector_op
            snapshot_gather_op
            beamscan_estimator_op
            feature_assembly_op
            detection_reduction_op
            cluster_dbscan_op
    )
    holoscan_radar_add_application(simo_beamform_then_detect
        DIRECTORY fmcw/simo_beamform_then_detect
        DEPENDS OPERATORS
            fmcw_range_processor_op
            clutter_filter_op
            fmcw_doppler_processor_op
            phase_shift_beamformer_op
            cfar_detector_op
            feature_assembly_op
    )
    holoscan_radar_add_application(mimo_tdm
        DIRECTORY fmcw/mimo_tdm
        DEPENDS OPERATORS
            fmcw_range_processor_op
            clutter_filter_op
            fmcw_doppler_processor_op
            mimo_deinterleave_op
            tdm_sparse_compensation_op
            cfar_detector_op
            beamscan_estimator_op
            feature_assembly_op
            detection_reduction_op
            cluster_dbscan_op
    )
    holoscan_radar_add_application(mimo_ddm
        DIRECTORY fmcw/mimo_ddm
        DEPENDS OPERATORS
            fmcw_range_processor_op
            fmcw_doppler_processor_op
            ddm_dense_demultiplex_op
            doppler_notch_op
            virtual_array_assembly_op
            cfar_detector_op
            snapshot_gather_op
            beamscan_estimator_op
            feature_assembly_op
            detection_reduction_op
            cluster_dbscan_op
    )
else()
    message(STATUS
        "Skipping standalone SIMO/MIMO applications; no Holoviz/X11 live RD")
endif()
