// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Serialize ArrayGeometry to/from a plain float64 Tensor for descriptor output
// ports. ArrayGeometry owns variable-length state and is therefore not a
// ChannelPod; the Tensor encoding gives the graph a bounded payload type.
//
// Layout — float64, shape (num_elements + 1, 3):
//   rows [0, num_elements): element_positions (x, y, z) meters
//   row  num_elements:      [wavelength, geometry_type_code, 0]
//     geometry_type_code: 0=ULA, 1=URA, 2=UCA, 3=CONFORMAL; NaN if unset

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

inline double array_geometry_type_to_code(std::optional<ArrayGeometryType> type) {
  if (!type.has_value()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  switch (*type) {
    case ArrayGeometryType::kUla:
      return 0.0;
    case ArrayGeometryType::kUra:
      return 1.0;
    case ArrayGeometryType::kUca:
      return 2.0;
    case ArrayGeometryType::kConformal:
      return 3.0;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

inline std::optional<ArrayGeometryType> array_geometry_type_from_code(double code) {
  if (!std::isfinite(code)) {
    return std::nullopt;
  }
  const int v = static_cast<int>(std::lround(code));
  switch (v) {
    case 0:
      return ArrayGeometryType::kUla;
    case 1:
      return ArrayGeometryType::kUra;
    case 2:
      return ArrayGeometryType::kUca;
    case 3:
      return ArrayGeometryType::kConformal;
    default:
      throw std::invalid_argument("ArrayGeometry tensor: unknown geometry_type code");
  }
}

// Pack into a contiguous host buffer of length 3 * (num_elements + 1).
inline std::vector<double> pack_array_geometry(const ArrayGeometry& geom) {
  geom.validate();
  const int64_t n = geom.num_elements();
  std::vector<double> out(static_cast<size_t>(3 * (n + 1)), 0.0);
  for (int64_t i = 0; i < 3 * n; ++i) {
    out[static_cast<size_t>(i)] = geom.element_positions[static_cast<size_t>(i)];
  }
  const size_t meta = static_cast<size_t>(3 * n);
  out[meta] = geom.wavelength;
  out[meta + 1] = array_geometry_type_to_code(geom.geometry_type);
  out[meta + 2] = 0.0;
  return out;
}

inline std::vector<int64_t> array_geometry_tensor_shape(const ArrayGeometry& geom) {
  return {geom.num_elements() + 1, 3};
}

// Unpack from a contiguous host buffer with shape (num_elements + 1, 3).
inline ArrayGeometry unpack_array_geometry(const double* data, int64_t rows, int64_t cols) {
  if (data == nullptr) {
    throw std::invalid_argument("ArrayGeometry tensor: null data");
  }
  if (cols != 3 || rows < 2) {
    throw std::invalid_argument(
        "ArrayGeometry tensor: expected shape (num_elements + 1, 3) with "
        "num_elements >= 1");
  }
  ArrayGeometry geom;
  const int64_t n = rows - 1;
  geom.element_positions.resize(static_cast<size_t>(3 * n));
  for (int64_t i = 0; i < 3 * n; ++i) {
    geom.element_positions[static_cast<size_t>(i)] = data[i];
  }
  const size_t meta = static_cast<size_t>(3 * n);
  geom.wavelength = data[meta];
  geom.geometry_type = array_geometry_type_from_code(data[meta + 1]);
  geom.validate();
  return geom;
}

}  // namespace holoscan::radar
