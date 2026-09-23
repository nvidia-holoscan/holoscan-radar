// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"

namespace holoscan::radar::detail {

// Host-planned mapping from TX-major/RX-minor physical channels to virtual
// elements. CUDA operators upload the applicable index vectors into their own
// ctor-sized device tables.
struct VirtualAssemblyPlan {
  OverlapPolicy policy{OverlapPolicy::kDedup};
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t num_virtual{0};
  std::vector<double> virtual_positions;
  std::vector<int64_t> src_channel;
  std::vector<int64_t> member_offsets;
  std::vector<int64_t> member_indices;
};

// Build p_virtual = p_tx + p_rx in deterministic TX-major/RX-minor order,
// applying the requested coincidence policy. `caller_name` prefixes
// validation errors so a shared implementation preserves operator identity.
VirtualAssemblyPlan build_virtual_plan(
    const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, int64_t num_tx, int64_t num_rx,
    OverlapPolicy policy, const char* caller_name);

ArrayGeometry make_array_geometry(const VirtualAssemblyPlan& plan,
                                  double wavelength);

}  // namespace holoscan::radar::detail
