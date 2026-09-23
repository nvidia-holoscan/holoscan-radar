// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "radar_common/descriptors/window_spec.hpp"
#include "radar_schemas/bridge_utils.hpp"
#include "radar_schemas/window_spec_bfbs_generated.h"
#include "radar_schemas/window_spec_schema_traits.hpp"

namespace holoscan::radar {

using WireWindowSpec = wire::WindowSpecT;
using WireWindowSpecRoot = wire::WindowSpec;
using WireWindowSpecBinarySchema = wire::WindowSpecBinarySchema;

// Keep the source descriptor, FlatBuffers schema, bridge, and mutator tests in
// sync when fields are added or removed.
static_assert(wire_bridge_detail::aggregate_field_count<WindowSpec>() == 7,
              "WindowSpec field count changed; update its wire synchronization surfaces");

inline WireWindowSpec to_wire(const WindowSpec& source) {
  WireWindowSpec result;
  result.type = static_cast<wire::WindowType>(static_cast<std::uint8_t>(source.type));
  result.sidelobe_attenuation_db = source.sidelobe_attenuation_db;
  result.has_kaiser_beta = source.kaiser_beta.has_value();
  result.kaiser_beta = source.kaiser_beta.value_or(0.0);
  result.taylor_nbar = source.taylor_nbar;
  result.tukey_alpha = source.tukey_alpha;
  result.coefficients = wire_bridge_detail::make_host_tensor<double>(
      std::span<const double>{source.coefficients}, DLDataType{kDLFloat, 64U, 1U});
  result.coherent_gain_normalization = source.coherent_gain_normalization;
  return result;
}

inline expected<WindowSpec, Error> from_wire(const WireWindowSpec& source) {
  const auto type_value = static_cast<std::uint8_t>(source.type);
  if (type_value > static_cast<std::uint8_t>(WindowType::kCustom)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "WindowSpec wire field 'type' is out of range"});
  }

  auto coefficients = wire_bridge_detail::read_host_vector<double>(
      source.coefficients, DLDataType{kDLFloat, 64U, 1U}, "coefficients");
  if (!coefficients) {
    return make_unexpected(std::move(coefficients).error());
  }

  WindowSpec result;
  result.type = static_cast<WindowType>(type_value);
  result.sidelobe_attenuation_db = source.sidelobe_attenuation_db;
  if (source.has_kaiser_beta) {
    result.kaiser_beta = source.kaiser_beta;
  }
  result.taylor_nbar = source.taylor_nbar;
  result.tukey_alpha = source.tukey_alpha;
  result.coefficients = std::move(*coefficients);
  result.coherent_gain_normalization = source.coherent_gain_normalization;
  return result;
}

}  // namespace holoscan::radar
