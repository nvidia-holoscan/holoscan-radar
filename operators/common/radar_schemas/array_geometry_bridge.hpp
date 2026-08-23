// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <exception>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_schemas/bridge_utils.hpp"
#include "radar_schemas/array_geometry_schema_traits.hpp"

namespace holoscan::radar {

using WireArrayGeometry = wire::ArrayGeometryT;
using WireArrayGeometryRoot = wire::ArrayGeometry;
using WireArrayGeometryBinarySchema = wire::ArrayGeometryBinarySchema;

static_assert(
    wire_bridge_detail::aggregate_field_count<ArrayGeometry>() == 3,
    "ArrayGeometry fields changed; update its wire synchronization surfaces");

inline WireArrayGeometry to_wire(const ArrayGeometry& source) {
  const std::vector<double> packed = pack_array_geometry(source);
  WireArrayGeometry result;
  result.packed_geometry = wire_bridge_detail::make_host_matrix<double>(
      std::span<const double>{packed}, source.num_elements() + 1, 3,
      DLDataType{kDLFloat, 64U, 1U});
  result.has_geometry_type = source.geometry_type.has_value();
  if (source.geometry_type.has_value()) {
    result.geometry_type = static_cast<wire::ArrayGeometryType>(
        static_cast<std::uint8_t>(*source.geometry_type));
  }
  return result;
}

inline expected<ArrayGeometry, Error> from_wire(
    const WireArrayGeometry& source) {
  auto packed = wire_bridge_detail::read_host_matrix<double>(
      source.packed_geometry, DLDataType{kDLFloat, 64U, 1U}, 3,
      "packed_geometry");
  if (!packed) {
    return make_unexpected(std::move(packed).error());
  }
  const std::int64_t rows =
      static_cast<std::int64_t>(source.packed_geometry->shape()[0]);
  try {
    ArrayGeometry result =
        unpack_array_geometry(packed->data(), rows, 3);
    const auto type_value =
        static_cast<std::uint8_t>(source.geometry_type);
    if (source.has_geometry_type) {
      if (type_value >
          static_cast<std::uint8_t>(
              wire::ArrayGeometryType_Conformal)) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument,
                  "ArrayGeometry wire field 'geometry_type' is out of range"});
      }
      const auto metadata_type =
          static_cast<ArrayGeometryType>(type_value);
      if (!result.geometry_type.has_value() ||
          *result.geometry_type != metadata_type) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument,
                  "ArrayGeometry wire geometry_type metadata is inconsistent "
                  "with packed_geometry"});
      }
    } else if (result.geometry_type.has_value()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                "ArrayGeometry wire geometry_type presence is inconsistent "
                "with packed_geometry"});
    }
    return result;
  } catch (const std::exception& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string("invalid ArrayGeometry wire metadata: ") +
                  error.what()});
  }
}

}  // namespace holoscan::radar
