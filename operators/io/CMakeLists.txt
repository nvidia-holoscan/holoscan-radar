# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standard-format I/O operators (see README.md in this directory). Each
# operator is gated on its own OP_<name> option (defaulting to ${BUILD_ALL}).
#
add_holohub_operator(cphd_reader_op)
add_holohub_operator(tensor_dump_op)
