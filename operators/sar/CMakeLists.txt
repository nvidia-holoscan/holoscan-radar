# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Register each operator after its HSDK 5 port is complete.
holoscan_radar_add_operator(phase_correction_op)
holoscan_radar_add_operator(sar_range_compression_op)
holoscan_radar_add_operator(aperture_manager_op)
holoscan_radar_add_operator(backprojection_op)
holoscan_radar_add_operator(autofocus_op)
