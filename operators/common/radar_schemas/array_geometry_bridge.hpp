// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_schemas/array_geometry_bfbs_generated.h"
#include "radar_schemas/array_geometry_schema_traits.hpp"
#include "radar_schemas/bridge_utils.hpp"

namespace holoscan::radar {

using WireArrayGeometry = wire::ArrayGeometryT;
using WireArrayGeometryRoot = wire::ArrayGeometry;
using WireArrayGeometryBinarySchema = wire::ArrayGeometryBinarySchema;

static_assert(wire_bridge_detail::aggregate_field_count<ArrayGeometry>() == 3,
              "ArrayGeometry fields changed; update its wire synchronization surfaces");

inline WireArrayGeometry to_wire(const ArrayGeometry& source) {
  const std::vector<double> packed = pack_array_geometry(source);
  WireArrayGeometry result;
  result.packed_geometry = wire_bridge_detail::make_host_matrix<double>(
      std::span<const double>{packed}, source.num_elements() + 1, 3, DLDataType{kDLFloat, 64U, 1U});
  result.has_geometry_type = source.geometry_type.has_value();
  if (source.geometry_type.has_value()) {
    result.geometry_type =
        static_cast<wire::ArrayGeometryType>(static_cast<std::uint8_t>(*source.geometry_type));
  }
  return result;
}

struct PreparedWireArrayGeometry {
  TensorOutputLoan loan;
  WireArrayGeometry payload;
};

inline expected<PreparedWireArrayGeometry, Error> prepare_wire_array_geometry(
    Output<WireArrayGeometry>& output, const ArrayGeometry& geometry) {
  const std::vector<double> packed = pack_array_geometry(geometry);
  const std::vector<int64_t> shape = array_geometry_tensor_shape(geometry);
  auto loan = output.allocate_tensor(
      TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLFloat, 64U, 1U}});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit([&packed](const TensorOutputWriteGuard& writer) {
    std::copy(packed.begin(), packed.end(), writer.data_as<double>());
  });
  if (!written) {
    return make_unexpected(std::move(written).error());
  }
  return PreparedWireArrayGeometry{
      .loan = std::move(*loan),
      .payload = to_wire(geometry),
  };
}

inline expected<void, Error> emit_prepared_wire_array_geometry(Output<WireArrayGeometry>& output,
                                                               ExecutionContext& context,
                                                               PreparedWireArrayGeometry prepared) {
  return emit_prepared_schema_tensor(output, context, std::move(prepared.loan),
                                     std::move(prepared.payload));
}

inline expected<ArrayGeometry, Error> from_wire(const WireArrayGeometry& source) {
  auto packed = wire_bridge_detail::read_host_matrix<double>(
      source.packed_geometry, DLDataType{kDLFloat, 64U, 1U}, 3, "packed_geometry");
  if (!packed) {
    return make_unexpected(std::move(packed).error());
  }
  const std::int64_t rows = static_cast<std::int64_t>(source.packed_geometry->shape()[0]);
  try {
    ArrayGeometry result = unpack_array_geometry(packed->data(), rows, 3);
    const auto type_value = static_cast<std::uint8_t>(source.geometry_type);
    if (source.has_geometry_type) {
      if (type_value > static_cast<std::uint8_t>(wire::ArrayGeometryType_Conformal)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                     "ArrayGeometry wire field 'geometry_type' is out of range"});
      }
      const auto metadata_type = static_cast<ArrayGeometryType>(type_value);
      if (!result.geometry_type.has_value() || *result.geometry_type != metadata_type) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                     "ArrayGeometry wire geometry_type metadata is inconsistent "
                                     "with packed_geometry"});
      }
    } else if (result.geometry_type.has_value()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "ArrayGeometry wire geometry_type presence is inconsistent "
                                   "with packed_geometry"});
    }
    return result;
  } catch (const std::exception& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string("invalid ArrayGeometry wire metadata: ") + error.what()});
  }
}

}  // namespace holoscan::radar
