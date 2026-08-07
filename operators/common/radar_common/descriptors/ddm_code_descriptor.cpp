// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/descriptors/ddm_code_descriptor.hpp"

#include <stdexcept>
#include <string>

namespace holoscan::radar {

void DDMCodeDescriptor::validate() const {
  if (num_tx <= 0) {
    throw std::invalid_argument("DDMCodeDescriptor: num_tx must be positive");
  }
  if (guard_bands < 0) {
    throw std::invalid_argument(
        "DDMCodeDescriptor: guard_bands must be >= 0");
  }
  const bool has_codes = !phase_codes.empty();
  const bool has_offsets = !doppler_offsets.empty();
  if (has_codes && has_offsets) {
    throw std::invalid_argument(
        "DDMCodeDescriptor: provide phase_codes or doppler_offsets, not both");
  }
  if (has_codes &&
      static_cast<int64_t>(phase_codes.size()) % num_tx != 0) {
    throw std::invalid_argument(
        "DDMCodeDescriptor: phase_codes length must be a multiple of num_tx");
  }
  if (has_offsets &&
      static_cast<int64_t>(doppler_offsets.size()) != num_tx) {
    throw std::invalid_argument(
        "DDMCodeDescriptor: doppler_offsets length must equal num_tx");
  }
}

}  // namespace holoscan::radar
