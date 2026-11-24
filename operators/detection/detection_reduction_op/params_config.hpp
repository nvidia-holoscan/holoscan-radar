// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include <holoscan/radar/detection/detection_reduction_op.hpp>
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using ReductionParamsCodec =
    params::DefaultedParamSchemaCodec<radar::DetectionReductionParams>;  // NOLINT

template <>
struct ConfigCodec<radar::DetectionReductionParams> : ReductionParamsCodec {};  // NOLINT

}  // namespace holoscan::config
