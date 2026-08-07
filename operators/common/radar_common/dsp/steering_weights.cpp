// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/steering_weights.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/dsp/window.hpp"
#include "radar_common/math/constants.hpp"

namespace holoscan::radar {

std::string steering_cache_key(const ArrayGeometry& geometry,
                               const AngleConvention& convention,
                               const std::vector<double>& az_angles,
                               const std::vector<double>& el_angles,
                               const WindowSpec& array_taper) {
  std::ostringstream os;
  os << static_cast<int>(convention.preset) << "|" << geometry.wavelength << "|"
     << geometry.element_positions.size() << "|";
  for (double v : geometry.element_positions) {
    os << v << ",";
  }
  os << "|";
  for (double v : az_angles) {
    os << v << ",";
  }
  os << "|";
  for (double v : el_angles) {
    os << v << ",";
  }
  os << "|" << window_cache_key(array_taper, geometry.num_elements());
  return os.str();
}

SteeringTable build_steering_table(const ArrayGeometry& geometry,
                                   const AngleConvention& convention,
                                   const std::vector<double>& az_angles,
                                   const std::vector<double>& el_angles,
                                   const WindowSpec& array_taper) {
  geometry.validate();
  if (az_angles.empty() || el_angles.empty()) {
    throw std::invalid_argument(
        "steering table: az_angles and el_angles must be non-empty");
  }

  const int64_t num_elements = geometry.num_elements();
  const auto taper =
      generate_window(array_taper, static_cast<int64_t>(num_elements));
  const double k = kTwoPi / geometry.wavelength;

  SteeringTable table;
  table.num_az = static_cast<int64_t>(az_angles.size());
  table.num_el = static_cast<int64_t>(el_angles.size());
  const int64_t num_beams = table.num_az * table.num_el;
  table.weights.resize(static_cast<size_t>(num_beams * num_elements));
  table.beam_valid.assign(static_cast<size_t>(num_beams), 1);

  size_t beam = 0;
  for (double el_b : el_angles) {
    for (double az_b : az_angles) {
      const CanonicalDirection dir =
          boundary_to_canonical(convention, az_b, el_b);
      const bool valid =
          convention.uses_direction_cosine()
              ? is_physical_direction_cosine(az_b, el_b)
              : (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z) > 0.5;
      if (!valid) {
        table.beam_valid[beam] = 0;
      }
      for (int64_t m = 0; m < num_elements; ++m) {
        const size_t base = static_cast<size_t>(m * 3);
        const double px = geometry.element_positions[base];
        const double py = geometry.element_positions[base + 1];
        const double pz = geometry.element_positions[base + 2];
        const double phase = k * (px * dir.x + py * dir.y + pz * dir.z);
        const double scale = valid ? taper[static_cast<size_t>(m)] : 0.0;
        table.weights[beam * static_cast<size_t>(num_elements) +
                      static_cast<size_t>(m)] =
            std::complex<double>(scale * std::cos(phase),
                                 scale * std::sin(phase));
      }
      ++beam;
    }
  }
  return table;
}

}  // namespace holoscan::radar
