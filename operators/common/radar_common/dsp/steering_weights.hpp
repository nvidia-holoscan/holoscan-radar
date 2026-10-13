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

// Build one amplitude coefficient per geometry element. Non-custom windows
// are separable across recognized filled URA axes; custom coefficients remain
// explicit per-element values in geometry/channel order.
std::vector<double> generate_steering_taper(const ArrayGeometry& geometry,
                                            const WindowSpec& array_taper);

SteeringTable build_steering_table(const ArrayGeometry& geometry, const AngleConvention& convention,
                                   const std::vector<double>& az_coordinates,
                                   const std::vector<double>& el_coordinates,
                                   const WindowSpec& array_taper);

std::string steering_cache_key(const ArrayGeometry& geometry, const AngleConvention& convention,
                               const std::vector<double>& az_coordinates,
                               const std::vector<double>& el_coordinates,
                               const WindowSpec& array_taper);

}  // namespace holoscan::radar
