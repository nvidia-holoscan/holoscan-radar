// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {
template <>
struct ConfigCodec<radar::TensorDumpParams>
: params::DefaultedParamSchemaCodec<radar::TensorDumpParams> {};
}  // namespace holoscan::config
