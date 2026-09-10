// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using FeatureParamsCodec =
    params::DefaultedParamSchemaCodec<radar::FeatureAssemblyParams>;  // NOLINT

template <>
struct ConfigCodec<radar::FeatureAssemblyParams>
    : FeatureParamsCodec {};  // NOLINT

}  // namespace holoscan::config
