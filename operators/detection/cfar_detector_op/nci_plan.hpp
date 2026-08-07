// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace holoscan::radar::detail {

// Row-major (C-order) decomposition of a linear index into per-axis
// coordinates. Shared by NciPlan construction and detection-index compaction.
void decompose_row_major(int64_t linear, const std::vector<int64_t>& dims,
                         std::vector<int64_t>& coords);

// Describes how leading axes of an rd_maps tensor are non-coherently
// integrated before CFAR. Integrated axes are summed (power domain); the
// remaining leading axes are preserved as independent CFAR batch slices.
struct NciPlan {
  std::vector<int64_t> leading_dims;       // all leading axes (..., D, R)
  std::vector<int64_t> preserved_dims;    // leading axes kept after NCI
  std::vector<int64_t> integrated_dims;    // sizes of integrated axes
  std::vector<int64_t> integrated_axes;    // sorted indices into leading_dims
  int64_t n_preserved{0};
  int64_t batch_in{1};   // product(leading_dims)
  int64_t out_batch{1};  // product(preserved_dims)
  int64_t fold{1};       // K = product(integrated_dims); 1 when nci_axes = []

  // For each output batch slice b in [0, out_batch) and fold index k in
  // [0, fold), batch_in_table[b * fold + k] is the linear index into the
  // collapsed (batch_in, doppler, range) power buffer.
  std::vector<int64_t> batch_in_table;

  // Full-rank output shape: preserved_dims + (num_doppler, num_range).
  std::vector<int64_t> output_shape(int64_t num_doppler,
                                    int64_t num_range) const;

  // Build a plan from the input tensor shape and nci_axes parameter.
  static NciPlan from_shape(const std::vector<int64_t>& shape,
                            const std::vector<int64_t>& nci_axes,
                            const char* port_name);
};

}  // namespace holoscan::radar::detail
