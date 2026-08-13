# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared application-support library (radar_app). Report sinks are always
# available; Holoviz-backed viz helpers and the demo apps that need them are
# gated on HOLOSCAN_RADAR_HAS_HOLOVIZ ([EA HSDK5 ONLY]).
add_subdirectory(common)

# Each application is gated on its own APP_<name> option, defaulting to
# ${BUILD_ALL}, and additionally requires Holoviz.
#
# [EA HSDK5 ONLY] holoscan_radar_add_application() additionally skips any app
# whose operator dependencies are not on the HSDK 5 port allow-list; Holoviz
# alone is not a sufficient gate because DEPENDS OPERATORS would FORCE OP_*=ON
# for targets holoscan_radar_add_operator() never creates.
if(NOT HOLOSCAN_RADAR_HAS_HOLOVIZ)
    message(STATUS
        "Skipping demo applications (fmcw_*); Holoviz is not available")
    return()
endif()

# Add one add_holohub_application() call per application; declare operator
# dependencies via DEPENDS OPERATORS so they auto-enable when the app is built.
holoscan_radar_add_application(fmcw_siso
    DEPENDS OPERATORS
        fmcw_range_processor_op
        clutter_filter_op
        fmcw_doppler_processor_op
        cfar_detector_op
)

holoscan_radar_add_application(fmcw_simo
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

holoscan_radar_add_application(fmcw_dense_simo
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

holoscan_radar_add_application(fmcw_aoa_estimators
    DEPENDS OPERATORS
        fmcw_range_processor_op
        clutter_filter_op
        fmcw_doppler_processor_op
        cfar_detector_op
        snapshot_gather_op
        beamscan_estimator_op
        mvdr_estimator_op
        music_estimator_op
        root_music_estimator_op
        esprit_estimator_op
        feature_assembly_op
        detection_reduction_op
        cluster_dbscan_op
)

holoscan_radar_add_application(fmcw_mimo
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
