# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared application-support library (radar_app), linked by the apps below.
# Added before the applications so its target exists when they link it.
add_subdirectory(common)

# Each application is gated on its own APP_<name> option, defaulting to ${BUILD_ALL}.
# Add one add_holohub_application() call per application; declare operator
# dependencies via DEPENDS OPERATORS so they auto-enable when the app is built.
add_holohub_application(fmcw_siso
    DEPENDS OPERATORS
        fmcw_range_processor_op
        clutter_filter_op
        fmcw_doppler_processor_op
        cfar_detector_op
)

add_holohub_application(fmcw_simo
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

add_holohub_application(fmcw_dense_simo
    DEPENDS OPERATORS
        fmcw_range_processor_op
        clutter_filter_op
        fmcw_doppler_processor_op
        phase_shift_beamformer_op
        cfar_detector_op
        feature_assembly_op
        detection_reduction_op
        cluster_dbscan_op
)

add_holohub_application(fmcw_mimo
    DEPENDS OPERATORS
        mimo_deinterleave_op
        mimo_coherent_compensation_op
        virtual_array_assembly_op
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
