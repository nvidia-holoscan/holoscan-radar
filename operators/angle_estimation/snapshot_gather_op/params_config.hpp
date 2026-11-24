// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include <holoscan/radar/angle_estimation/snapshot_gather_op.hpp>
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using SnapshotParamsCodec =
    params::DefaultedParamSchemaCodec<radar::SnapshotGatherParams>;  // NOLINT

template <>
struct ConfigCodec<radar::SnapshotGatherParams> : SnapshotParamsCodec {};  // NOLINT

}  // namespace holoscan::config
