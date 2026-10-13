// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include "detection/cfar_detector_op/cfar_detector_op.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using CfarParamsCodec = params::DefaultedParamSchemaCodec<radar::CFARParams>;  // NOLINT

template <>
struct ConfigCodec<radar::CFARParams> : CfarParamsCodec {};  // NOLINT

}  // namespace holoscan::config
