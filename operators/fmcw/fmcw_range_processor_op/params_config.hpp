// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using RangeParamsCodec =
    params::DefaultedParamSchemaCodec<radar::FMCWRangeProcessorParams>;  // NOLINT

template <>
struct ConfigCodec<radar::FMCWRangeProcessorParams> : RangeParamsCodec {};  // NOLINT

}  // namespace holoscan::config
