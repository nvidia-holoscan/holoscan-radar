// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/virtual_array.hpp"

namespace holoscan::radar::detail {

// Kernel knobs for one tick. tx_positions, rx_positions, and overlap_policy
// are Params-only (static). wavelength is the live-merged scalar: the
// non-positive Params / override sentinel is rejected in resolve_params,
// so this is always a positive finite value after that call.
struct VirtualAssemblyStageConfig {
  std::vector<double> tx_positions;
  std::vector<double> rx_positions;
  double wavelength{0.0};
  OverlapPolicy overlap_policy{OverlapPolicy::kDedup};
};

// Contract input is exactly rank 4. Extra leading axes are not a module
// batch (the module named-axis invariant).
struct RdMapsLayout {
  int64_t num_tx{0};
  int64_t num_rx{0};
  int64_t num_doppler{0};
  int64_t num_range{0};
};

RdMapsLayout parse_rd_maps_layout(const holoscan::Tensor& tensor, const char* port_name);

std::vector<int64_t> assembled_rd_shape(const RdMapsLayout& in, int64_t num_virtual);

template <typename T>
void run_virtual_assembly_stage(const void* input, void* output, const RdMapsLayout& layout,
                                const VirtualAssemblyPlan& plan, const int64_t* d_src_channel,
                                const int64_t* d_member_offsets, const int64_t* d_member_indices,
                                cudaStream_t stream);

}  // namespace holoscan::radar::detail
