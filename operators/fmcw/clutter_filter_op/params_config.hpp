// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include <holoscan/radar/fmcw/clutter_filter_op.hpp>
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using ClutterParamsCodec = params::DefaultedParamSchemaCodec<radar::ClutterFilterParams>;  // NOLINT

template <>
struct ConfigCodec<radar::ClutterFilterParams> : ClutterParamsCodec {};  // NOLINT

}  // namespace holoscan::config
