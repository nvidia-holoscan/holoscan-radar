# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Detection operators (see README.md in this directory). Each operator is
# gated on its own OP_<name> option (defaulting to ${BUILD_ALL}).
add_holohub_operator(cfar_detector_op)
add_holohub_operator(detection_reduction_op)
