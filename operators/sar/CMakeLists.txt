# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SAR formation operators (see README.md in this directory). Each operator is
# gated on its own OP_<name> option (defaulting to ${BUILD_ALL}).
#
# The standard-format readers and writers are deliberately NOT here — they
# live in operators/io/ behind their own gates, so a build that wants only the
# formation chain pays nothing for the XML and NITF dependencies.
add_holohub_operator(sar_range_compression_op)
add_holohub_operator(backprojection_op)
