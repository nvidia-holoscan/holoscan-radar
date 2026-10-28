# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Register each operator after its HSDK 5 port is complete.
add_holohub_operator(phase_correction_op)
add_holohub_operator(sar_range_compression_op)
add_holohub_operator(aperture_manager_op)
add_holohub_operator(backprojection_op)
