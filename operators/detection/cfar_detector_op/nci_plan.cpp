// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/cfar_detector_op/nci_plan.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace holoscan::radar::detail {

namespace {

int64_t product(const std::vector<int64_t>& dims) {
  int64_t p = 1;
  for (int64_t d : dims) {
    p *= d;
  }
  return p;
}

int64_t compose_row_major(const std::vector<int64_t>& dims, const std::vector<int64_t>& coords) {
  int64_t idx = 0;
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(dims.size()) - 1; i >= 0; --i) {
    idx += coords[static_cast<size_t>(i)] * stride;
    stride *= dims[i];
  }
  return idx;
}

}  // namespace

void decompose_row_major(int64_t linear, const std::vector<int64_t>& dims,
                         std::vector<int64_t>& coords) {
  coords.resize(dims.size());
  int64_t rem = linear;
  for (int64_t i = static_cast<int64_t>(dims.size()) - 1; i >= 0; --i) {
    coords[static_cast<size_t>(i)] = rem % dims[i];
    rem /= dims[i];
  }
}

std::vector<int64_t> NciPlan::output_shape(int64_t num_doppler, int64_t num_range) const {
  std::vector<int64_t> shape = preserved_dims;
  shape.push_back(num_doppler);
  shape.push_back(num_range);
  return shape;
}

NciPlan NciPlan::from_shape(const std::vector<int64_t>& shape, const std::vector<int64_t>& nci_axes,
                            const char* port_name, bool materialize_lookup) {
  if (shape.size() < 2) {
    std::ostringstream msg;
    msg << port_name << ": expected rank >= 2 (..., doppler, range), got rank " << shape.size();
    throw std::invalid_argument(msg.str());
  }

  NciPlan plan;
  plan.leading_dims.assign(shape.begin(), shape.end() - 2);
  const int64_t num_leading = static_cast<int64_t>(plan.leading_dims.size());
  plan.batch_in = product(plan.leading_dims);
  if (plan.batch_in <= 0) {
    throw std::invalid_argument(std::string(port_name) +
                                ": zero/negative-sized leading axis in input shape");
  }

  std::unordered_set<int64_t> seen;
  for (int64_t axis : nci_axes) {
    if (axis < 0 || axis >= num_leading) {
      std::ostringstream msg;
      msg << port_name << ": nci_axes index " << axis << " is out of range for " << num_leading
          << " leading axis";
      throw std::invalid_argument(msg.str());
    }
    if (!seen.insert(axis).second) {
      std::ostringstream msg;
      msg << port_name << ": duplicate nci_axes index " << axis;
      throw std::invalid_argument(msg.str());
    }
    plan.integrated_axes.push_back(axis);
  }
  std::sort(plan.integrated_axes.begin(), plan.integrated_axes.end());

  std::vector<bool> integrated(static_cast<size_t>(num_leading), false);
  for (int64_t axis : plan.integrated_axes) {
    integrated[static_cast<size_t>(axis)] = true;
    plan.integrated_dims.push_back(plan.leading_dims[static_cast<size_t>(axis)]);
  }

  for (int64_t i = 0; i < num_leading; ++i) {
    if (!integrated[static_cast<size_t>(i)]) {
      plan.preserved_dims.push_back(plan.leading_dims[static_cast<size_t>(i)]);
    }
  }

  plan.n_preserved = static_cast<int64_t>(plan.preserved_dims.size());
  plan.out_batch = product(plan.preserved_dims);
  plan.fold = product(plan.integrated_dims);
  if (plan.fold <= 0) {
    plan.fold = 1;
  }

  if (plan.fold > 1 && materialize_lookup) {
    plan.batch_in_table.resize(static_cast<size_t>(plan.out_batch * plan.fold));
    std::vector<int64_t> leading_coords(plan.leading_dims.size());
    std::vector<int64_t> preserved_coords(plan.preserved_dims.size());
    std::vector<int64_t> integrated_coords(plan.integrated_dims.size());
    for (int64_t b_out = 0; b_out < plan.out_batch; ++b_out) {
      decompose_row_major(b_out, plan.preserved_dims, preserved_coords);
      for (int64_t k = 0; k < plan.fold; ++k) {
        decompose_row_major(k, plan.integrated_dims, integrated_coords);
        size_t preserved_i = 0;
        size_t integrated_i = 0;
        for (size_t axis = 0; axis < plan.leading_dims.size(); ++axis) {
          if (integrated[axis]) {
            leading_coords[axis] = integrated_coords[integrated_i++];
          } else {
            leading_coords[axis] = preserved_coords[preserved_i++];
          }
        }
        plan.batch_in_table[static_cast<size_t>(b_out * plan.fold + k)] =
            compose_row_major(plan.leading_dims, leading_coords);
      }
    }
  }

  return plan;
}

}  // namespace holoscan::radar::detail
