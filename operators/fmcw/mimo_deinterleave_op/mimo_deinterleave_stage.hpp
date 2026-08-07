// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/descriptors/tx_schedule.hpp"

namespace holoscan::radar::detail {

struct AdcCubeLayout {
  std::vector<int64_t> leading;
  int64_t batch{1};
  int64_t num_rx{0};
  int64_t slow_time{0};
  int64_t samples{0};
};

AdcCubeLayout parse_adc_cube_layout(const holoscan::Tensor& tensor,
                                    const char* port_name);

std::vector<int64_t> deinterleaved_adc_shape(const AdcCubeLayout& in,
                                             int64_t num_tx,
                                             int64_t chirps_per_tx);

void run_deinterleave_stage(const void* input, void* output,
                            const AdcCubeLayout& layout, int64_t num_tx,
                            int64_t chirps_per_tx, const int64_t* d_gather_map,
                            size_t elem_bytes, cudaStream_t stream);

}  // namespace holoscan::radar::detail
