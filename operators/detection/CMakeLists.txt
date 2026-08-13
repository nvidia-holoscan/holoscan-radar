# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Detection operators (see README.md in this directory).
# [EA HSDK5 ONLY] Only HSDK-5-ported operators are added (see
# HoloscanRadarOptionalFeatures.cmake).
holoscan_radar_add_operator(cfar_detector_op)
holoscan_radar_add_operator(detection_reduction_op)
holoscan_radar_add_operator(feature_assembly_op)
