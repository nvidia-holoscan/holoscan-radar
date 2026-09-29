// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"

namespace holoscan::radar {

// The LFM and FMCW waveform specs have the same seven fields (D24). Reuse
// the existing HSDK 5 wire schema while keeping their public C++ descriptors
// distinct. A future field addition must review both specs and this bridge.
using WireLFMWaveform = wire::FMCWWaveformT;
static_assert(wire_bridge_detail::aggregate_field_count<LFMWaveformSpec>() == 7,
              "LFMWaveformSpec field count changed; update the waveform wire bridge");

inline WireLFMWaveform to_wire(const LFMWaveformDescriptor& source) {
  return to_wire(FMCWWaveformSpec{
      .carrier_frequency = source.carrier_frequency(),
      .bandwidth = source.bandwidth(),
      .chirp_slope = source.slope(),
      .chirp_duration = source.chirp_duration(),
      .sample_rate = source.sample_rate(),
      .chirp_repetition_interval = source.chirp_repetition_interval(),
      .reference_samples = source.reference_samples(),
  });
}

inline expected<LFMWaveformDescriptor, Error> from_wire_lfm(const WireLFMWaveform& source) {
  auto samples = wire_bridge_detail::read_host_vector<std::complex<double>>(
      source.reference_samples, DLDataType{kDLComplex, 128U, 1U}, "reference_samples");
  if (!samples) {
    return make_unexpected(std::move(samples).error());
  }
  LFMWaveformSpec spec;
  spec.carrier_frequency = source.carrier_frequency;
  if (source.has_bandwidth) {
    spec.bandwidth = source.bandwidth;
  }
  if (source.has_chirp_slope) {
    spec.chirp_slope = source.chirp_slope;
  }
  spec.chirp_duration = source.chirp_duration;
  spec.sample_rate = source.sample_rate;
  spec.chirp_repetition_interval = source.chirp_repetition_interval;
  spec.reference_samples = std::move(*samples);
  try {
    return LFMWaveformDescriptor(std::move(spec));
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string("invalid LFM waveform wire metadata: ") + error.what()});
  }
}

}  // namespace holoscan::radar
