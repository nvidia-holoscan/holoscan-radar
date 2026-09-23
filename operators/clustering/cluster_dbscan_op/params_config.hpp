// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/params/config.hpp>

#include "clustering/cluster_dbscan_op/cluster_dbscan_op.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::config {

using ClusterParamsCodec = params::DefaultedParamSchemaCodec<radar::ClusterDBSCANParams>;  // NOLINT

template <>
struct ConfigCodec<radar::ClusterDBSCANParams> : ClusterParamsCodec {};  // NOLINT

}  // namespace holoscan::config
