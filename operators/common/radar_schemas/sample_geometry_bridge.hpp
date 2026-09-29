// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <exception>
#include <span>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_schemas/bridge_utils.hpp"
#include "radar_schemas/sample_geometry_bfbs_generated.h"
#include "radar_schemas/sample_geometry_schema_traits.hpp"

namespace holoscan::radar {
using WireSampleGeometry = wire::SampleGeometryT;

inline std::array<double, 11> pack_sample_geometry(const SampleGeometry& source) {
  return {source.origin()[0], source.origin()[1],  source.origin()[2], source.axis_u()[0],
          source.axis_u()[1], source.axis_u()[2],  source.axis_v()[0], source.axis_v()[1],
          source.axis_v()[2], source.spacing()[0], source.spacing()[1]};
}

inline WireSampleGeometry to_wire(const SampleGeometry& source) {
  const auto packed = pack_sample_geometry(source);
  WireSampleGeometry result;
  result.packed_axes = wire_bridge_detail::make_host_tensor<double>(std::span<const double>{packed},
                                                                    DLDataType{kDLFloat, 64U, 1U});
  result.num_v = source.num_v();
  result.num_u = source.num_u();
  return result;
}

inline expected<SampleGeometry, Error> from_wire(const WireSampleGeometry& source) {
  auto packed = wire_bridge_detail::read_host_vector<double>(
      source.packed_axes, DLDataType{kDLFloat, 64U, 1U}, "packed_axes");
  if (!packed || packed->size() != 11U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "SampleGeometry wire packed_axes needs 11 doubles"});
  }
  try {
    const auto& p = *packed;
    return SampleGeometry(SampleGeometrySpec{.origin = {p[0], p[1], p[2]},
                                             .axis_u = {p[3], p[4], p[5]},
                                             .axis_v = {p[6], p[7], p[8]},
                                             .spacing = {p[9], p[10]},
                                             .counts = {source.num_v, source.num_u}});
  } catch (const std::exception& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string("invalid SampleGeometry wire metadata: ") + error.what()});
  }
}
}  // namespace holoscan::radar
