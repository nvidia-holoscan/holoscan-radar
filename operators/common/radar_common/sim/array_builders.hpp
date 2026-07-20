// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Antenna-array geometry builders shared by the FMCW example applications.
// These bridge the config-plane ArrayGeometry descriptor with the simulator's
// Vec3 element positions, so they live in sim/ (which may depend on
// descriptors/, not the other way around). Pure host math — no Holoscan, CUDA,
// or MatX dependency.
//
// Placement convention: element pitch is expressed as a fraction of the carrier
// wavelength (spacing_wl, e.g. 0.5 = half-wavelength) and multiplied by the
// real carrier wavelength, so the array positions and the signal model share
// one metric frame (a free-standing wavelength knob would desynchronize the
// array from the simulator and alias the DOA).

#pragma once

#include <cstdint>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/sim/point_target_sim.hpp"

namespace holoscan::radar::sim {

// Uniform rectangular array (URA) in the y-z plane, normal to boresight (+x).
// The y aperture resolves azimuth and the z aperture resolves elevation, so a
// URA measures both (a single line array has only 1D aperture and cannot
// separate the two). Element (iy, iz) sits at (0, iy * spacing, iz * spacing);
// rows (iz) vary elevation, columns (iy) vary azimuth.
inline ArrayGeometry make_ura_geometry(int64_t az_elements, int64_t el_elements,
                                       double spacing_wl, double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = ArrayGeometryType::kUra;
  geom.element_positions.reserve(
      static_cast<size_t>(az_elements * el_elements * 3));
  for (int64_t iz = 0; iz < el_elements; ++iz) {
    for (int64_t iy = 0; iy < az_elements; ++iy) {
      geom.element_positions.push_back(0.0);
      geom.element_positions.push_back(static_cast<double>(iy) * spacing_wl *
                                       wavelength);
      geom.element_positions.push_back(static_cast<double>(iz) * spacing_wl *
                                       wavelength);
    }
  }
  geom.validate();
  return geom;
}

// The same URA element positions as Vec3 receiver locations for the simulator,
// in the identical (iz outer, iy inner) order as make_ura_geometry so the
// scene's rx_positions stay row-aligned with the ArrayGeometry.
inline std::vector<Vec3> make_ura_rx_positions(int64_t az_elements,
                                               int64_t el_elements,
                                               double spacing_wl,
                                               double wavelength) {
  std::vector<Vec3> positions;
  positions.reserve(static_cast<size_t>(az_elements * el_elements));
  for (int64_t iz = 0; iz < el_elements; ++iz) {
    for (int64_t iy = 0; iy < az_elements; ++iy) {
      positions.push_back(
          Vec3{0.0, static_cast<double>(iy) * spacing_wl * wavelength,
               static_cast<double>(iz) * spacing_wl * wavelength});
    }
  }
  return positions;
}

// Virtual array for a MIMO aperture: every (TX, RX) pair contributes one
// virtual element at the sum of the physical TX and RX positions. Row order is
// (tx outer, rx inner).
inline ArrayGeometry make_virtual_geometry(
    const std::vector<Vec3>& tx_positions,
    const std::vector<Vec3>& rx_positions, double wavelength) {
  ArrayGeometry geom;
  geom.wavelength = wavelength;
  geom.geometry_type = ArrayGeometryType::kConformal;
  const int64_t num_tx = static_cast<int64_t>(tx_positions.size());
  const int64_t num_rx = static_cast<int64_t>(rx_positions.size());
  geom.element_positions.reserve(static_cast<size_t>(num_tx * num_rx * 3));
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      const Vec3& tp = tx_positions[static_cast<size_t>(tx)];
      const Vec3& rp = rx_positions[static_cast<size_t>(rx)];
      geom.element_positions.push_back(tp[0] + rp[0]);
      geom.element_positions.push_back(tp[1] + rp[1]);
      geom.element_positions.push_back(tp[2] + rp[2]);
    }
  }
  geom.validate();
  return geom;
}

// Linear array of `count` elements along the y (azimuth) axis: element i at
// (0, i * spacing_wl * wavelength, 0). Used for a MIMO TX row.
inline std::vector<Vec3> make_linear_positions(int64_t count, double spacing_wl,
                                               double wavelength) {
  std::vector<Vec3> positions;
  positions.reserve(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    positions.push_back(
        Vec3{0.0, static_cast<double>(i) * spacing_wl * wavelength, 0.0});
  }
  return positions;
}

// Flatten a list of element positions to a row-major (N, 3) double array, e.g.
// for VirtualArrayAssemblyOp's tx_positions / rx_positions parameters.
inline std::vector<double> flatten_positions(
    const std::vector<Vec3>& positions) {
  std::vector<double> flat;
  flat.reserve(positions.size() * 3);
  for (const Vec3& p : positions) {
    flat.push_back(p[0]);
    flat.push_back(p[1]);
    flat.push_back(p[2]);
  }
  return flat;
}

}  // namespace holoscan::radar::sim
