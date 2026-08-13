# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Angle-estimation operators (see README.md in this directory).
# [EA HSDK5 ONLY] Only HSDK-5-ported operators are added (see
# HoloscanRadarOptionalFeatures.cmake).
holoscan_radar_add_operator(snapshot_gather_op)
holoscan_radar_add_operator(subarray_selection_op)
holoscan_radar_add_operator(beamscan_estimator_op)
holoscan_radar_add_operator(mvdr_estimator_op)
holoscan_radar_add_operator(music_estimator_op)
holoscan_radar_add_operator(root_music_estimator_op)
holoscan_radar_add_operator(esprit_estimator_op)
holoscan_radar_add_operator(phase_shift_beamformer_op)
