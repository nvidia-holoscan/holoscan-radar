// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/window_spec.hpp"

namespace holoscan::radar {

// Host-side steering table for conventional (Bartlett) beamforming.
struct SteeringTable {
  std::vector<std::complex<double>> weights;  // (num_beams, num_elements)
  std::vector<uint8_t> beam_valid;            // 1 = physical grid point
  int64_t num_az{0};
  int64_t num_el{0};
};

SteeringTable build_steering_table(const ArrayGeometry& geometry,
                                   const AngleConvention& convention,
                                   const std::vector<double>& az_angles,
                                   const std::vector<double>& el_angles,
                                   const WindowSpec& array_taper);

std::string steering_cache_key(const ArrayGeometry& geometry,
                               const AngleConvention& convention,
                               const std::vector<double>& az_angles,
                               const std::vector<double>& el_angles,
                               const WindowSpec& array_taper);

}  // namespace holoscan::radar
