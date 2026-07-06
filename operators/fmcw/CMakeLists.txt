# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# FMCW signal-processing operators (see README.md in this directory).
# Future MIMO marshaling operators also live in this group.
#
# The fused operator is declared FIRST: add_holohub_operator() binds each
# OP_<name> option the first time it runs, and the fused entry's DEPENDS
# force-sets its three dependencies to ON — that FORCE must happen before
# the dependencies' own option() calls bind (same ordering rule as pkg/,
# see the top-level CMakeLists.txt).
add_holohub_operator(fmcw_range_doppler_processor_op
    DEPENDS OPERATORS
        fmcw_range_processor_op
        clutter_filter_op
        fmcw_doppler_processor_op
)
add_holohub_operator(fmcw_range_processor_op)
add_holohub_operator(clutter_filter_op)
add_holohub_operator(fmcw_doppler_processor_op)
