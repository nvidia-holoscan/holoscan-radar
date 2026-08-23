// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_schemas/bridge_utils.hpp"
#include "radar_schemas/fmcw_waveform_schema_traits.hpp"

namespace holoscan::radar {

using WireFMCWWaveform = wire::FMCWWaveformT;
using WireFMCWWaveformRoot = wire::FMCWWaveform;
using WireFMCWWaveformBinarySchema = wire::FMCWWaveformBinarySchema;

// Keep the source spec, FlatBuffers schema, bridge, and mutator tests in sync
// when fields are added or removed.
static_assert(
    wire_bridge_detail::aggregate_field_count<FMCWWaveformSpec>() == 7,
    "FMCWWaveformSpec field count changed; update its wire synchronization "
    "surfaces");

inline WireFMCWWaveform to_wire(const FMCWWaveformSpec& source) {
  WireFMCWWaveform result;
  result.carrier_frequency = source.carrier_frequency;
  result.has_bandwidth = source.bandwidth.has_value();
  result.bandwidth = source.bandwidth.value_or(0.0);
  result.has_chirp_slope = source.chirp_slope.has_value();
  result.chirp_slope = source.chirp_slope.value_or(0.0);
  result.chirp_duration = source.chirp_duration;
  result.sample_rate = source.sample_rate;
  result.chirp_repetition_interval =
      source.chirp_repetition_interval;
  result.reference_samples =
      wire_bridge_detail::make_host_tensor<std::complex<double>>(
          std::span<const std::complex<double>>{
              source.reference_samples},
          DLDataType{kDLComplex, 128U, 1U});
  return result;
}

inline WireFMCWWaveform to_wire(
    const FMCWWaveformDescriptor& source) {
  return to_wire(FMCWWaveformSpec{
      .carrier_frequency = source.carrier_frequency(),
      .bandwidth = source.bandwidth(),
      .chirp_slope = source.slope(),
      .chirp_duration = source.chirp_duration(),
      .sample_rate = source.sample_rate(),
      .chirp_repetition_interval =
          source.chirp_repetition_interval(),
      .reference_samples = source.reference_samples(),
  });
}

inline expected<FMCWWaveformDescriptor, Error> from_wire(
    const WireFMCWWaveform& source) {
  auto reference_samples =
      wire_bridge_detail::read_host_vector<std::complex<double>>(
          source.reference_samples,
          DLDataType{kDLComplex, 128U, 1U}, "reference_samples");
  if (!reference_samples) {
    return make_unexpected(std::move(reference_samples).error());
  }

  FMCWWaveformSpec spec;
  spec.carrier_frequency = source.carrier_frequency;
  if (source.has_bandwidth) {
    spec.bandwidth = source.bandwidth;
  }
  if (source.has_chirp_slope) {
    spec.chirp_slope = source.chirp_slope;
  }
  spec.chirp_duration = source.chirp_duration;
  spec.sample_rate = source.sample_rate;
  spec.chirp_repetition_interval =
      source.chirp_repetition_interval;
  spec.reference_samples = std::move(*reference_samples);

  try {
    return FMCWWaveformDescriptor(std::move(spec));
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string("invalid FMCWWaveform wire metadata: ") +
                  error.what()});
  }
}

}  // namespace holoscan::radar
