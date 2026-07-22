# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Angle-estimation operators (see README.md in this directory). Each operator
# is gated on its own OP_<name> option (defaulting to ${BUILD_ALL}).
add_holohub_operator(snapshot_gather_op)
add_holohub_operator(subarray_selection_op)
add_holohub_operator(beamscan_estimator_op)
add_holohub_operator(mvdr_estimator_op)
add_holohub_operator(music_estimator_op)
add_holohub_operator(root_music_estimator_op)
add_holohub_operator(phase_shift_beamformer_op)
