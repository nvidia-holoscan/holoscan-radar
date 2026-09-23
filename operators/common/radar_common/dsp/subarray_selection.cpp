// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/subarray_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar {

namespace {

// Greatest common divisor of two lengths within `tol` (Euclid on reals) — used
// to infer the underlying lattice unit spacing of a holey array from its
// coordinate gaps (e.g. gaps {1, 1, 2} → unit 1).
double fgcd(double a, double b, double tol) {
  a = std::abs(a);
  b = std::abs(b);
  while (b > tol) {
    const double t = std::fmod(a, b);
    a = b;
    b = t;
  }
  return a;
}

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

// The (possibly holey) integer lattice a geometry samples: aperture axes, their
// extents, and a row-major occupancy grid holding each filled cell's flat input
// channel index (-1 = hole).
struct Occupancy {
  std::vector<int> axis;        // aperture Cartesian axes (ascending), size D
  std::vector<int64_t> extent;  // lattice extent per aperture axis
  std::vector<int32_t> cell;    // prod(extent) row-major; channel index or -1
  int64_t d{0};
};

Occupancy build_occupancy(const ArrayGeometry& geometry) {
  geometry.validate();
  const int64_t n = geometry.num_elements();
  const std::vector<double>& p = geometry.element_positions;

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

  Occupancy occ;
  std::vector<std::vector<double>> uniq(3);
  for (int c = 0; c < 3; ++c) {
    std::vector<double> vals(static_cast<size_t>(n));
    for (int64_t e = 0; e < n; ++e) {
      vals[static_cast<size_t>(e)] = p[static_cast<size_t>(e * 3 + c)];
    }
    uniq[static_cast<size_t>(c)] = unique_axis_values(std::move(vals), tol);
    if (uniq[static_cast<size_t>(c)].size() > 1) {
      occ.axis.push_back(c);
    }
  }
  const int64_t d = static_cast<int64_t>(occ.axis.size());
  occ.d = d;
  if (d == 0) {
    throw std::invalid_argument(
        "subarray_selection: array_geometry has no aperture (all elements "
        "colocated)");
  }
  if (d > 2) {
    throw std::invalid_argument(
        "subarray_selection: 3-D element apertures are not supported");
  }

  // Infer unit spacing (gcd of coordinate gaps) + extent per aperture axis.
  std::vector<double> unit(static_cast<size_t>(d));
  std::vector<double> base(static_cast<size_t>(d));
  for (int64_t di = 0; di < d; ++di) {
    const std::vector<double>& u =
        uniq[static_cast<size_t>(occ.axis[static_cast<size_t>(di)])];
    double g = u[1] - u[0];
    for (size_t i = 2; i < u.size(); ++i) {
      g = fgcd(g, u[i] - u[i - 1], tol);
    }
    if (!(g > tol)) {
      throw std::invalid_argument(
          "subarray_selection: degenerate lattice spacing");
    }
    unit[static_cast<size_t>(di)] = g;
    base[static_cast<size_t>(di)] = u.front();
    const int64_t ext =
        static_cast<int64_t>(std::llround((u.back() - u.front()) / g)) + 1;
    occ.extent.push_back(ext);
  }

  int64_t total = 1;
  for (int64_t di = 0; di < d; ++di) {
    total *= occ.extent[static_cast<size_t>(di)];
  }
  occ.cell.assign(static_cast<size_t>(total), -1);
  for (int64_t e = 0; e < n; ++e) {
    int64_t linear = 0;
    for (int64_t di = 0; di < d; ++di) {
      const int c = occ.axis[static_cast<size_t>(di)];
      const double v = p[static_cast<size_t>(e * 3 + c)];
      const double r = (v - base[static_cast<size_t>(di)]) /
                       unit[static_cast<size_t>(di)];
      const int64_t idx = static_cast<int64_t>(std::llround(r));
      if (std::abs(static_cast<double>(idx) * unit[static_cast<size_t>(di)] -
                   (v - base[static_cast<size_t>(di)])) > tol ||
          idx < 0 || idx >= occ.extent[static_cast<size_t>(di)]) {
        throw std::invalid_argument(
            "subarray_selection: element off the inferred lattice "
            "(array_geometry is not on a consistent uniform grid)");
      }
      linear = linear * occ.extent[static_cast<size_t>(di)] + idx;
    }
    if (occ.cell[static_cast<size_t>(linear)] != -1) {
      throw std::invalid_argument(
          "subarray_selection: duplicate element positions");
    }
    occ.cell[static_cast<size_t>(linear)] = static_cast<int32_t>(e);
  }
  return occ;
}

// Largest filled sub-block of the occupancy grid, returned as the row-major
// list of input channel indices it contains (lattice order). 1-D: longest run
// of filled cells. 2-D: maximal all-filled rectangle (max element count, ties
// broken by larger azimuth = first-axis extent).
std::vector<int32_t> largest_filled_block(const Occupancy& occ) {
  if (occ.d == 1) {
    const int64_t m = occ.extent[0];
    int64_t best_start = 0, best_len = 0, start = 0, len = 0;
    for (int64_t i = 0; i < m; ++i) {
      if (occ.cell[static_cast<size_t>(i)] != -1) {
        if (len == 0) {
          start = i;
        }
        ++len;
        if (len > best_len) {
          best_len = len;
          best_start = start;
        }
      } else {
        len = 0;
      }
    }
    std::vector<int32_t> out;
    for (int64_t i = 0; i < best_len; ++i) {
      out.push_back(occ.cell[static_cast<size_t>(best_start + i)]);
    }
    return out;
  }

  // 2-D: maximal all-ones rectangle via the histogram (largest-rect) sweep.
  const int64_t ma = occ.extent[0];  // azimuth (first aperture axis)
  const int64_t mb = occ.extent[1];  // elevation
  std::vector<int64_t> h(static_cast<size_t>(mb), 0);
  int64_t best_area = 0, best_h = 0;
  int64_t ba0 = 0, ba1 = -1, bb0 = 0, bb1 = -1;  // inclusive rectangle bounds
  for (int64_t a = 0; a < ma; ++a) {
    for (int64_t b = 0; b < mb; ++b) {
      const bool filled = occ.cell[static_cast<size_t>(a * mb + b)] != -1;
      h[static_cast<size_t>(b)] = filled ? h[static_cast<size_t>(b)] + 1 : 0;
    }
    std::vector<int64_t> st;  // indices of increasing bar heights
    for (int64_t bb = 0; bb <= mb; ++bb) {
      const int64_t cur = (bb == mb) ? 0 : h[static_cast<size_t>(bb)];
      while (!st.empty() && cur < h[static_cast<size_t>(st.back())]) {
        const int64_t height = h[static_cast<size_t>(st.back())];
        st.pop_back();
        const int64_t left = st.empty() ? 0 : st.back() + 1;
        const int64_t width = bb - left;  // columns [left, bb-1]
        const int64_t area = height * width;
        if (area > best_area || (area == best_area && height > best_h)) {
          best_area = area;
          best_h = height;
          ba0 = a - height + 1;
          ba1 = a;
          bb0 = left;
          bb1 = bb - 1;
        }
      }
      st.push_back(bb);
    }
  }
  std::vector<int32_t> out;
  for (int64_t a = ba0; a <= ba1; ++a) {
    for (int64_t b = bb0; b <= bb1; ++b) {
      out.push_back(occ.cell[static_cast<size_t>(a * mb + b)]);
    }
  }
  return out;
}

ArrayGeometry reduced_from_channels(const ArrayGeometry& geometry,
                                    const std::vector<int32_t>& channels,
                                    int64_t d) {
  ArrayGeometry reduced;
  reduced.wavelength = geometry.wavelength;
  reduced.geometry_type =
      d == 1 ? ArrayGeometryType::kUla : ArrayGeometryType::kUra;
  reduced.element_positions.resize(channels.size() * 3);
  for (size_t l = 0; l < channels.size(); ++l) {
    const int32_t e = channels[l];
    for (int c = 0; c < 3; ++c) {
      reduced.element_positions[l * 3 + static_cast<size_t>(c)] =
          geometry.element_positions[static_cast<size_t>(e * 3 + c)];
    }
  }
  reduced.validate();
  return reduced;
}

}  // namespace

SubarraySelection build_subarray_selection(
    const ArrayGeometry& geometry,
    const std::vector<int64_t>& explicit_indices, int64_t min_subarray_size) {
  const int64_t n = geometry.num_elements();
  SubarraySelection sel;
  sel.num_channels = n;

  std::vector<int32_t> channels;  // input channel indices, lattice row-major
  int64_t d = 0;

  if (explicit_indices.empty()) {
    // Auto: largest filled ULA/URA sub-block.
    const Occupancy occ = build_occupancy(geometry);
    d = occ.d;
    channels = largest_filled_block(occ);
  } else {
    // Explicit: validate range + uniqueness, then that the chosen channels form
    // a filled uniform ULA/URA (re-derive their lattice, require no holes), and
    // re-emit them in lattice order.
    std::set<int64_t> seen;
    ArrayGeometry chosen;
    chosen.wavelength = geometry.wavelength;
    for (int64_t idx : explicit_indices) {
      if (idx < 0 || idx >= n) {
        throw std::invalid_argument(
            "subarray_selection: explicit channel index out of range");
      }
      if (!seen.insert(idx).second) {
        throw std::invalid_argument(
            "subarray_selection: duplicate explicit channel index");
      }
      for (int c = 0; c < 3; ++c) {
        chosen.element_positions.push_back(
            geometry.element_positions[static_cast<size_t>(idx * 3 + c)]);
      }
    }
    const Occupancy occ = build_occupancy(chosen);  // throws if non-uniform
    d = occ.d;
    int64_t total = 1;
    for (int64_t di = 0; di < d; ++di) {
      total *= occ.extent[static_cast<size_t>(di)];
    }
    if (total != static_cast<int64_t>(explicit_indices.size())) {
      throw std::invalid_argument(
          "subarray_selection: explicit channel indices do not form a filled "
          "ULA/URA (holey)");
    }
    // occ.cell holds indices into `explicit_indices`; map to input channels in
    // lattice row-major order.
    for (int32_t local : occ.cell) {
      channels.push_back(
          static_cast<int32_t>(explicit_indices[static_cast<size_t>(local)]));
    }
  }

  const int64_t l = static_cast<int64_t>(channels.size());
  if (l < min_subarray_size) {
    throw std::invalid_argument(
        "subarray_selection: no filled uniform sub-array of the requested "
        "minimum size exists in this geometry");
  }

  bool identity = l == n;
  for (int64_t channel = 0; identity && channel < n; ++channel) {
    identity =
        channels[static_cast<size_t>(channel)] == channel;
  }
  sel.subarray_size = l;
  sel.channel_index = std::move(channels);
  sel.reduced = !identity;
  sel.reduced_geometry =
      sel.reduced ? reduced_from_channels(geometry, sel.channel_index, d)
                  : geometry;
  return sel;
}

}  // namespace holoscan::radar
