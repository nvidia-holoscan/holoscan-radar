// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include "sar/backprojection_op/backprojection_op.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {
template <>
struct ConfigCodec<radar::BackprojectionParams>
: params::DefaultedParamSchemaCodec<radar::BackprojectionParams> {};
}  // namespace holoscan::config
