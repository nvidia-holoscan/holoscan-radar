// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using BeamscanParamsCodec = params::DefaultedParamSchemaCodec<radar::BeamscanParams>;  // NOLINT

template <>
struct ConfigCodec<radar::BeamscanParams> : BeamscanParamsCodec {};  // NOLINT

}  // namespace holoscan::config
