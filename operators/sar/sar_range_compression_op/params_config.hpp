// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include <holoscan/radar/sar/sar_range_compression_op.hpp>
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {
template <>
struct ConfigCodec<radar::SARRangeCompressionParams>
: params::DefaultedParamSchemaCodec<radar::SARRangeCompressionParams> {};
}  // namespace holoscan::config
