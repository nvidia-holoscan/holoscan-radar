// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/spatial_smoothing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

namespace {

// A detected uniform aperture: one entry per Cartesian axis (x,y,z) that spans
// more than one distinct element coordinate, in ascending axis order (so a URA
// in the y-z plane detects y=azimuth then z=elevation).
struct Lattice {
  std::vector<int> axis;        // Cartesian axis index (0=x,1=y,2=z), size D
  std::vector<int64_t> extent;  // grid points along each aperture axis
  // coord[e * D + d] = integer lattice index of element e on aperture axis d.
  std::vector<int64_t> coord;
  int64_t num_elements{0};
};

// Cluster the sorted coordinate values along one axis into unique lattice
// values within `tol`, returning the representative (mean) of each cluster.
std::vector<double> unique_axis_values(std::vector<double> vals, double tol) {
  std::sort(vals.begin(), vals.end());
  std::vector<double> uniq;
  for (double v : vals) {
    if (uniq.empty() || std::abs(v - uniq.back()) > tol) {
      uniq.push_back(v);
    }
  }
  return uniq;
}

// Nearest cluster index for `v` in the ascending `uniq` list (within tol);
// returns -1 if `v` matches no cluster (a coordinate off the detected lattice).
int64_t axis_index(const std::vector<double>& uniq, double v, double tol) {
  for (size_t i = 0; i < uniq.size(); ++i) {
    if (std::abs(v - uniq[i]) <= tol) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

// Detect a filled, uniformly spaced 1-D/2-D lattice from element_positions, or
// throw std::invalid_argument describing why the geometry is unsupported for
// spatial smoothing (non-uniform, holey, or a >2-D aperture).
Lattice detect_lattice(const ArrayGeometry& geometry) {
  geometry.validate();
  const int64_t n = geometry.num_elements();
  const std::vector<double>& p = geometry.element_positions;

  // Tolerance relative to the largest coordinate span (fall back to wavelength
  // for a degenerate single-point spread).
  double lo[3] = {p[0], p[1], p[2]};
  double hi[3] = {p[0], p[1], p[2]};
  for (int64_t e = 0; e < n; ++e) {
    for (int c = 0; c < 3; ++c) {
      const double v = p[static_cast<size_t>(e * 3 + c)];
      lo[c] = std::min(lo[c], v);
      hi[c] = std::max(hi[c], v);
    }
  }
  double span = 0.0;
  for (int c = 0; c < 3; ++c) {
    span = std::max(span, hi[c] - lo[c]);
  }
  if (!(span > 0.0)) {
    span = geometry.wavelength;
  }
  const double tol = 1e-6 * span;

  // Per-axis unique coordinate values → identify aperture axes (extent > 1).
  Lattice lat;
  lat.num_elements = n;
  std::vector<std::vector<double>> uniq(3);
  for (int c = 0; c < 3; ++c) {
    std::vector<double> vals(static_cast<size_t>(n));
    for (int64_t e = 0; e < n; ++e) {
      vals[static_cast<size_t>(e)] = p[static_cast<size_t>(e * 3 + c)];
    }
    uniq[static_cast<size_t>(c)] = unique_axis_values(std::move(vals), tol);
    if (uniq[static_cast<size_t>(c)].size() > 1) {
      lat.axis.push_back(c);
      lat.extent.push_back(
          static_cast<int64_t>(uniq[static_cast<size_t>(c)].size()));
    }
  }

  const int64_t d = static_cast<int64_t>(lat.axis.size());
  if (d == 0) {
    throw std::invalid_argument(
        "spatial_smoothing: array_geometry has no aperture (all elements "
        "colocated); smoothing needs a ULA/URA");
  }
  if (d > 2) {
    throw std::invalid_argument(
        "spatial_smoothing: 3-D element apertures are not supported; provide "
        "a 1-D (ULA) or 2-D (URA) uniform geometry");
  }

  // Uniform spacing along each aperture axis.
  for (int64_t di = 0; di < d; ++di) {
    const size_t c = static_cast<size_t>(lat.axis[static_cast<size_t>(di)]);
    const std::vector<double>& u = uniq[c];
    const double step = u[1] - u[0];
    for (size_t i = 2; i < u.size(); ++i) {
      if (std::abs((u[i] - u[i - 1]) - step) > tol) {
        throw std::invalid_argument(
            "spatial_smoothing: array_geometry is non-uniform along an "
            "aperture axis; reduce it to a uniform sub-array upstream (S3) or "
            "run with spatial_smoothing disabled");
      }
    }
  }

  // Per-element integer lattice coordinates + filled-grid (bijection) check.
  int64_t filled = 1;
  for (int64_t di = 0; di < d; ++di) {
    filled *= lat.extent[static_cast<size_t>(di)];
  }
  if (filled != n) {
    throw std::invalid_argument(
        "spatial_smoothing: array_geometry is holey (element count != filled "
        "lattice size); reduce it to a filled uniform sub-array upstream (S3) "
        "or run with spatial_smoothing disabled");
  }

  lat.coord.assign(static_cast<size_t>(n * d), 0);
  std::vector<uint8_t> occupied(static_cast<size_t>(n), 0);
  for (int64_t e = 0; e < n; ++e) {
    int64_t linear = 0;
    for (int64_t di = 0; di < d; ++di) {
      const int c = lat.axis[static_cast<size_t>(di)];
      const double v = p[static_cast<size_t>(e * 3 + c)];
      const int64_t idx = axis_index(uniq[static_cast<size_t>(c)], v, tol);
      if (idx < 0) {
        throw std::invalid_argument(
            "spatial_smoothing: element off the detected lattice");
      }
      lat.coord[static_cast<size_t>(e * d + di)] = idx;
      linear = linear * lat.extent[static_cast<size_t>(di)] + idx;
    }
    if (occupied[static_cast<size_t>(linear)]) {
      throw std::invalid_argument(
          "spatial_smoothing: duplicate element positions (not a filled "
          "lattice)");
    }
    occupied[static_cast<size_t>(linear)] = 1;
  }
  return lat;
}

}  // namespace

SpatialSmoothingPlan build_spatial_smoothing_plan(
    const ArrayGeometry& geometry, const std::vector<int64_t>& counts,
    bool forward_backward) {
  SpatialSmoothingPlan plan;
  plan.num_channels = geometry.num_elements();
  plan.forward_backward = forward_backward;

  if (counts.empty()) {
    throw std::invalid_argument(
        "spatial_smoothing: counts must be non-empty (use an empty "
        "spatial_smoothing parameter to disable smoothing)");
  }

  const Lattice lat = detect_lattice(geometry);
  const int64_t d = static_cast<int64_t>(lat.axis.size());
  if (static_cast<int64_t>(counts.size()) != d) {
    std::ostringstream os;
    os << "spatial_smoothing: expected " << d << " sub-array count(s) for a "
       << d << "-D aperture, got " << counts.size();
    throw std::invalid_argument(os.str());
  }

  // Per-aperture-axis sub-array lengths L_d = M_d - counts_d + 1.
  std::vector<int64_t> sub_len(static_cast<size_t>(d));
  bool all_ones = true;
  for (int64_t di = 0; di < d; ++di) {
    const int64_t p_d = counts[static_cast<size_t>(di)];
    if (p_d < 1 || p_d > lat.extent[static_cast<size_t>(di)]) {
      std::ostringstream os;
      os << "spatial_smoothing: sub-array count " << p_d << " on axis " << di
         << " must be in [1, " << lat.extent[static_cast<size_t>(di)] << "]";
      throw std::invalid_argument(os.str());
    }
    sub_len[static_cast<size_t>(di)] =
        lat.extent[static_cast<size_t>(di)] - p_d + 1;
    if (p_d != 1) {
      all_ones = false;
    }
  }

  // All-ones counts with no forward-backward averaging is a pure pass-through.
  if (all_ones && !forward_backward) {
    plan.enabled = false;
    return plan;
  }

  int64_t sub_size = 1;
  int64_t num_sub = 1;
  for (int64_t di = 0; di < d; ++di) {
    sub_size *= sub_len[static_cast<size_t>(di)];
    num_sub *= counts[static_cast<size_t>(di)];
  }
  plan.enabled = true;
  plan.subarray_size = sub_size;
  plan.num_subarrays = num_sub;

  // Linear lattice index (mixed radix over extents) → flat channel index.
  std::vector<int32_t> grid_to_flat(static_cast<size_t>(lat.num_elements), -1);
  for (int64_t e = 0; e < lat.num_elements; ++e) {
    int64_t linear = 0;
    for (int64_t di = 0; di < d; ++di) {
      const int64_t c = lat.coord[static_cast<size_t>(e * d + di)];
      linear = linear * lat.extent[static_cast<size_t>(di)] + c;
    }
    grid_to_flat[static_cast<size_t>(linear)] = static_cast<int32_t>(e);
  }

  // Enumerate sub-arrays: shift s (mixed radix over counts), local offset o
  // (mixed radix over sub_len). Local element order = o row-major over the
  // aperture axes; sub-array order p = s row-major over the axes.
  plan.subarray_index.assign(static_cast<size_t>(num_sub * sub_size), 0);
  std::vector<int64_t> s(static_cast<size_t>(d), 0);
  for (int64_t pidx = 0; pidx < num_sub; ++pidx) {
    std::vector<int64_t> o(static_cast<size_t>(d), 0);
    for (int64_t l = 0; l < sub_size; ++l) {
      int64_t linear = 0;
      for (int64_t di = 0; di < d; ++di) {
        const int64_t g =
            s[static_cast<size_t>(di)] + o[static_cast<size_t>(di)];
        linear = linear * lat.extent[static_cast<size_t>(di)] + g;
      }
      plan.subarray_index[static_cast<size_t>(pidx * sub_size + l)] =
          grid_to_flat[static_cast<size_t>(linear)];
      // increment local offset o (last axis fastest)
      for (int64_t di = d - 1; di >= 0; --di) {
        if (++o[static_cast<size_t>(di)] < sub_len[static_cast<size_t>(di)]) {
          break;
        }
        o[static_cast<size_t>(di)] = 0;
      }
    }
    // increment shift s (last axis fastest)
    for (int64_t di = d - 1; di >= 0; --di) {
      if (++s[static_cast<size_t>(di)] < counts[static_cast<size_t>(di)]) {
        break;
      }
      s[static_cast<size_t>(di)] = 0;
    }
  }

  // Reduced geometry: the reference sub-array (shift 0) = the first sub_size
  // entries of subarray_index, in local order.
  plan.reduced_geometry.wavelength = geometry.wavelength;
  plan.reduced_geometry.geometry_type =
      d == 1 ? ArrayGeometryType::kUla : ArrayGeometryType::kUra;
  plan.reduced_geometry.element_positions.resize(
      static_cast<size_t>(sub_size * 3));
  for (int64_t l = 0; l < sub_size; ++l) {
    const int32_t e = plan.subarray_index[static_cast<size_t>(l)];
    for (int c = 0; c < 3; ++c) {
      plan.reduced_geometry.element_positions[static_cast<size_t>(l * 3 + c)] =
          geometry.element_positions[static_cast<size_t>(e * 3 + c)];
    }
  }
  plan.reduced_geometry.validate();

  // Forward-backward flip J: reverse each aperture axis within the sub-array.
  if (forward_backward) {
    plan.fb_flip.assign(static_cast<size_t>(sub_size), 0);
    std::vector<int64_t> o(static_cast<size_t>(d), 0);
    for (int64_t l = 0; l < sub_size; ++l) {
      int64_t flipped = 0;
      for (int64_t di = 0; di < d; ++di) {
        const int64_t rev = sub_len[static_cast<size_t>(di)] - 1 -
                            o[static_cast<size_t>(di)];
        flipped = flipped * sub_len[static_cast<size_t>(di)] + rev;
      }
      plan.fb_flip[static_cast<size_t>(l)] = static_cast<int32_t>(flipped);
      for (int64_t di = d - 1; di >= 0; --di) {
        if (++o[static_cast<size_t>(di)] < sub_len[static_cast<size_t>(di)]) {
          break;
        }
        o[static_cast<size_t>(di)] = 0;
      }
    }
  }

  return plan;
}

}  // namespace holoscan::radar
