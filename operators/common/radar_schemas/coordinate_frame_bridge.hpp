// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <exception>
#include <span>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_schemas/bridge_utils.hpp"
#include "radar_schemas/coordinate_frame_bfbs_generated.h"
#include "radar_schemas/coordinate_frame_schema_traits.hpp"

namespace holoscan::radar {

using WireCoordinateFrame = wire::CoordinateFrameT;

inline WireCoordinateFrame to_wire(const CoordinateFrame& source) {
  WireCoordinateFrame result;
  result.frame = static_cast<uint8_t>(source.frame());
  result.ellipsoid = static_cast<uint8_t>(source.ellipsoid());
  result.has_origin_llh = source.origin_llh().has_value();
  const std::array<double, 3> origin = source.origin_llh().value_or(std::array<double, 3>{});
  result.origin_llh = wire_bridge_detail::make_host_tensor<double>(std::span<const double>{origin},
                                                                   DLDataType{kDLFloat, 64U, 1U});
  result.has_epoch = source.epoch().has_value();
  result.epoch = source.epoch().value_or(0.0);
  return result;
}

inline expected<CoordinateFrame, Error> from_wire(const WireCoordinateFrame& source) {
  auto origin = wire_bridge_detail::read_host_vector<double>(
      source.origin_llh, DLDataType{kDLFloat, 64U, 1U}, "origin_llh");
  if (!origin || origin->size() != 3U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CoordinateFrame wire origin_llh needs three doubles"});
  }
  try {
    CoordinateFrameSpec spec;
    spec.frame = static_cast<CoordinateFrameType>(source.frame);
    spec.ellipsoid = static_cast<Ellipsoid>(source.ellipsoid);
    if (source.has_origin_llh)
      spec.origin_llh = {*origin->begin(), (*origin)[1], (*origin)[2]};
    if (source.has_epoch)
      spec.epoch = source.epoch;
    return CoordinateFrame(spec);
  } catch (const std::exception& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string("invalid CoordinateFrame wire metadata: ") + error.what()});
  }
}

}  // namespace holoscan::radar
