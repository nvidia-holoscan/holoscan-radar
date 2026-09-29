# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared application-support library used by the standalone FMCW apps.
add_subdirectory(common)

# Each workflow is a standalone application with its own APP_<name>
# gate, executable, metadata, and configuration set.
holoscan_radar_add_application(siso
    DIRECTORY fmcw/siso
    DEPENDS OPERATORS
        fmcw_range_processor_op
        clutter_filter_op
        fmcw_doppler_processor_op
        cfar_detector_op
        feature_assembly_op
        detection_reduction_op
        cluster_dbscan_op
)
holoscan_radar_add_application(simo
    DIRECTORY fmcw/simo
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
holoscan_radar_add_application(dense_beamforming
    DIRECTORY fmcw/dense_beamforming
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
holoscan_radar_add_application(adaptive_aoa
    DIRECTORY fmcw/adaptive_aoa
    DEPENDS OPERATORS
        fmcw_range_processor_op
        clutter_filter_op
        fmcw_doppler_processor_op
        mimo_deinterleave_op
        tdm_sparse_compensation_op
        cfar_detector_op
        beamscan_estimator_op
        rank_one_residual_screen_op
        music_estimator_op
        root_music_estimator_op
        esprit_estimator_op
        mvdr_estimator_op
        multi_source_expansion_op
        feature_assembly_op
)
holoscan_radar_add_application(multi_waveform
    DIRECTORY fmcw/multi_waveform
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

add_subdirectory(sar)
